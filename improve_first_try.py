"""
improve_first_try.py — Research-level inpainting via RePaint-style iterative resampling
             + negative prompt classifier-free guidance.

Core contributions vs. vanilla vanilla_first_try.py:
  1. RePaint harmonization loop: at each diffusion timestep t, composite the
     known (unmasked) latent region — re-noised to match the current noise level —
     back into the generated latent. This forces context propagation during
     diffusion rather than only at the final pixel compositing step.
     Reference: Lugmayr et al., "RePaint: Inpainting using Denoising Diffusion
     Probabilistic Models", CVPR 2022.
  2. Negative prompt CFG: instead of the implicit empty-string unconditional
     embedding used in vanilla SD, an explicit negative prompt directly modifies
     the score function ε = ε_neg + s*(ε_pos - ε_neg), pushing the output away
     from named artifact distributions.
  3. Custom inference loop: accesses pipe.scheduler, pipe.vae, pipe.unet, and
     pipe.text_encoder directly rather than calling pipe(...) as a black box.
     This is the only way to implement per-timestep harmonization in diffusers.

CLI is identical to vanilla_first_try.py plus --resample-steps / --negative-prompt flags.
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: str) -> StableDiffusionInpaintPipeline:
    model_id = "sd2-community/stable-diffusion-2-base"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_auth_token=True,
        safety_checker=None,
    )
    # Swap to DDIM scheduler so we can call add_noise with a specific timestep
    # and the algebra is straightforward (PNDM's multi-step buffer makes
    # re-noising harder to reason about).
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe


# ---------------------------------------------------------------------------
# Input preprocessing (identical to vanilla_first_try.py)
# ---------------------------------------------------------------------------

def preprocess_inputs(image_path: str, mask_path: str, target_size=(512, 512)):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    image = image.resize(target_size, Image.LANCZOS)
    mask = mask.resize(target_size, Image.NEAREST)

    # mask_rgb is kept for compatibility but we also return the L-mode mask
    # for latent-space masking.
    mask_rgb = mask.convert("RGB")
    return image, mask, mask_rgb


# ---------------------------------------------------------------------------
# Negative prompt selection
# ---------------------------------------------------------------------------

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, artifacts, bad anatomy, distorted face, "
    "watermark, noise, grain, overexposed, underexposed, mismatched colors, "
    "seam, boundary artifact, unnatural edge"
)


def select_negative_prompt(user_supplied: str | None) -> str:
    if user_supplied:
        return user_supplied
    return DEFAULT_NEGATIVE_PROMPT


# ---------------------------------------------------------------------------
# Latent encoding / decoding helpers
# ---------------------------------------------------------------------------

def encode_image_to_latent(pipe, image: Image.Image) -> torch.Tensor:
    """Encode a PIL image to the VAE latent space. Returns (1, 4, H/8, W/8)."""
    img_tensor = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(pipe.device)
    with torch.no_grad():
        latent = pipe.vae.encode(img_tensor).latent_dist.sample()
    latent = latent * pipe.vae.config.scaling_factor
    return latent


def decode_latent_to_image(pipe, latent: torch.Tensor) -> Image.Image:
    """Decode a VAE latent tensor to a PIL image."""
    latent = latent / pipe.vae.config.scaling_factor
    with torch.no_grad():
        decoded = pipe.vae.decode(latent).sample
    decoded = decoded.squeeze(0).permute(1, 2, 0).float()
    decoded = ((decoded + 1.0) * 127.5).clamp(0, 255).byte()
    return Image.fromarray(decoded.cpu().numpy())


# ---------------------------------------------------------------------------
# Text embedding helper
# ---------------------------------------------------------------------------

def get_text_embeddings(pipe, prompt: str, negative_prompt: str) -> torch.Tensor:
    """
    Returns stacked [negative, positive] embeddings for CFG.
    Shape: (2, seq_len, embed_dim).
    """
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    def _embed(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(pipe.device)
        with torch.no_grad():
            emb = text_encoder(input_ids)[0]
        return emb

    pos_emb = _embed(prompt)
    neg_emb = _embed(negative_prompt)
    return torch.cat([neg_emb, pos_emb], dim=0)  # (2, seq, dim)


# ---------------------------------------------------------------------------
# Mask preprocessing for latent space
# ---------------------------------------------------------------------------

def prepare_latent_mask(pipe, mask_L: Image.Image) -> torch.Tensor:
    """
    Downsample the binary mask to latent resolution (H/8, W/8) and binarize.
    Returns (1, 1, H/8, W/8) float32 tensor on pipe.device.
    1.0 = masked (generate), 0.0 = known (preserve).
    """
    latent_size = (mask_L.width // 8, mask_L.height // 8)
    mask_small = mask_L.resize(latent_size, Image.NEAREST)
    mask_np = (np.array(mask_small) > 127).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(pipe.device)
    return mask_tensor


# ---------------------------------------------------------------------------
# RePaint inference loop
# ---------------------------------------------------------------------------

def repaint_inpaint(
    pipe: StableDiffusionInpaintPipeline,
    image: Image.Image,
    mask_L: Image.Image,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    resample_steps: int,
    harmonize_start: float = 0.5,
) -> Image.Image:
    """
    Custom diffusion loop implementing RePaint-style harmonization with
    progressive harmonization gating.

    At every denoising timestep t:
      1. Standard UNet denoising step with CFG.
      2. Progressive RePaint harmonization: composite the known (unmasked)
         latent region — re-noised to noise level t-1 — back into the
         generated latent. Harmonization is GATED: it only activates after
         `harmonize_start` fraction of timesteps have elapsed. This allows
         the early (high-noise) steps to run freely so the model can form a
         new object identity, then blends the boundary in the late steps.
         Without gating, object replacement tasks produce ghosting because
         the original object's latent is composited back at every step.
      3. Resampling: re-noise x_{t-1} back to x_t and repeat U times.

    Args:
        pipe:             Loaded StableDiffusionInpaintPipeline (DDIM scheduler).
        image:            512×512 PIL RGB input image.
        mask_L:           512×512 PIL L-mode mask (white=inpaint, black=keep).
        prompt:           Positive text prompt.
        negative_prompt:  Negative text prompt for CFG.
        seed:             RNG seed.
        steps:            Number of DDIM denoising steps.
        guidance_scale:   CFG scale s.
        resample_steps:   Number of RePaint resampling jumps U per timestep.
        harmonize_start:  Fraction of total steps to skip before harmonization
                          begins. 0.0 = harmonize from the first step (original
                          RePaint, good for completion/inpainting). 0.5 = only
                          harmonize in the second half (good for replacement).
                          1.0 = no harmonization at all.
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(steps)
    timesteps = scheduler.timesteps  # descending, e.g. [980, 960, ..., 20]

    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # --- Encode original image to latent ---
    original_latent = encode_image_to_latent(pipe, image)  # (1,4,64,64)

    # --- Prepare latent mask ---
    latent_mask = prepare_latent_mask(pipe, mask_L)  # (1,1,64,64)

    # --- Text embeddings ---
    text_embeddings = get_text_embeddings(pipe, prompt, negative_prompt)  # (2,77,1024)

    # --- Initialize: start from pure noise ---
    noise = torch.randn(original_latent.shape, generator=generator, device=pipe.device)
    x_t = scheduler.add_noise(original_latent, noise, timesteps[0:1])

    # Clamp resample_steps to at least 1 so the UNet always runs
    inner_iters = max(1, resample_steps)

    # Timestep index at which harmonization activates
    harmonize_from = int(len(timesteps) * harmonize_start)

    # --- Main denoising loop ---
    for i, t in enumerate(timesteps):
        t_batch = torch.tensor([t], device=pipe.device, dtype=torch.long)

        # Inner resampling loop (RePaint: repeat U times per timestep)
        for u in range(inner_iters):
            # 1. UNet forward pass with CFG
            latent_model_input = torch.cat([x_t, x_t], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample

            # Classifier-free guidance
            noise_pred_neg, noise_pred_pos = noise_pred.chunk(2)
            noise_pred_guided = noise_pred_neg + guidance_scale * (noise_pred_pos - noise_pred_neg)

            # 2. Scheduler step: x_t -> x_{t-1}
            x_prev = scheduler.step(
                noise_pred_guided,
                t,
                x_t,
                generator=generator,
            ).prev_sample

            # 3. Progressive RePaint harmonization.
            #    Skip harmonization in the early high-noise steps so the model
            #    can freely form the new object's coarse structure.
            #    Only blend the boundary once we're past harmonize_from.
            if i >= harmonize_from:
                if i + 1 < len(timesteps):
                    t_prev = timesteps[i + 1]
                    t_prev_batch = torch.tensor([t_prev], device=pipe.device, dtype=torch.long)
                    known_noise = torch.randn(original_latent.shape, generator=generator, device=pipe.device, dtype=original_latent.dtype)
                    original_noised = scheduler.add_noise(original_latent, known_noise, t_prev_batch)
                else:
                    original_noised = original_latent

                x_prev = latent_mask * x_prev + (1.0 - latent_mask) * original_noised

            # 4. Resampling jump (if not last iteration of inner loop)
            if u < inner_iters - 1:
                resample_noise = torch.randn(x_prev.shape, generator=generator, device=pipe.device, dtype=x_prev.dtype)
                x_t = scheduler.add_noise(x_prev, resample_noise, t_batch)
            else:
                x_t = x_prev

    # --- Decode final latent ---
    result = decode_latent_to_image(pipe, x_t)
    return result


# ---------------------------------------------------------------------------
# Text overlay (render-and-blend)
# ---------------------------------------------------------------------------

def sample_text_color(image: Image.Image, mask_L: Image.Image) -> tuple:
    """
    Sample the median color from pixels just outside the mask boundary.
    This gives a data-driven estimate of the text color without hardcoding.
    Falls back to white (255, 255, 255) if no border pixels are found.
    """
    mask_np = np.array(mask_L)  # (H, W), 0-255
    img_np = np.array(image)    # (H, W, 3)

    # Dilate mask by a few pixels to get the immediate border region
    dilated = mask_L.filter(ImageFilter.MaxFilter(9))
    dilated_np = np.array(dilated)

    # Border = dilated minus original mask
    border = (dilated_np > 127) & (mask_np <= 127)
    if not border.any():
        return (255, 255, 255)

    border_pixels = img_np[border]  # (N, 3)
    median_color = tuple(int(np.median(border_pixels[:, c])) for c in range(3))
    return median_color


def render_text_overlay(
    inpainted: Image.Image,
    mask_L: Image.Image,
    text: str,
    font_size: int | None = None,
) -> Image.Image:
    """
    Render `text` into the masked region of `inpainted` using PIL.

    Steps:
      1. Compute the bounding box of the mask region.
      2. Sample text color from pixels bordering the mask.
      3. Fit the font size to fill the bounding box height (or use font_size if given).
      4. Render the text centered in the bounding box.
      5. Composite: text pixels replace the inpainted result only inside the mask.

    This guarantees correct glyph identity regardless of what the diffusion
    model generated. The diffusion output is kept for the background texture;
    only the foreground letter shapes are overridden.
    """
    mask_np = np.array(mask_L)  # (H, W)
    coords = np.argwhere(mask_np > 127)
    if coords.size == 0:
        return inpainted  # mask is empty, nothing to do

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    box_w = x1 - x0
    box_h = y1 - y0

    # Determine font size: fill ~80% of the box height
    target_font_size = font_size if font_size else max(8, int(box_h * 0.80))

    # Try to load a system font; fall back to PIL default if unavailable
    font = None
    candidate_fonts = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidate_fonts:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, target_font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    # Sample text color from the surrounding border pixels
    text_color = sample_text_color(inpainted, mask_L)

    # Render text onto a transparent layer
    overlay = Image.new("RGBA", inpainted.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Center text in the mask bounding box
    text_x = x0 + (box_w - text_w) // 2
    text_y = y0 + (box_h - text_h) // 2

    draw.text((text_x, text_y), text, font=font, fill=(*text_color, 255))

    # Composite: use the rendered text only where the mask is active
    result = inpainted.convert("RGBA")
    mask_rgba = mask_L.point(lambda p: 255 if p > 127 else 0)  # binary alpha
    result.paste(overlay, mask=mask_rgba)
    return result.convert("RGB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Improved inpainting: RePaint resampling + negative prompt CFG"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--mask", required=True, help="Path to mask (white=inpaint)")
    parser.add_argument("--prompt", required=True, help="Positive text prompt")
    parser.add_argument("--output_dir", default="output_improved",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50,
                        help="DDIM denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=9.0,
                        help="CFG guidance scale (higher = stronger prompt; default 9.0)")
    parser.add_argument("--resample-steps", type=int, default=2,
                        help="RePaint resampling jumps U per timestep (default 2; 1=no resampling, just harmonization)")
    parser.add_argument("--harmonize-start", type=float, default=0.5,
                        help="Fraction of denoising steps before harmonization activates. "
                             "0.0 = full RePaint (best for completion/inpainting). "
                             "0.5 = free generation for first half, then blend boundary (best for object replacement). "
                             "1.0 = no harmonization. (default: 0.5)")
    parser.add_argument("--negative-prompt", type=str, default=None,
                        help="Custom negative prompt (uses default artifact list if omitted)")
    parser.add_argument("--text-overlay", type=str, default=None,
                        help="Text to render into the masked region after diffusion "
                             "(e.g. --text-overlay 'D'). Color is sampled automatically "
                             "from the border pixels. Use for text reconstruction.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    pipe = load_model(device)

    print("Preprocessing inputs...")
    image, mask_L, mask_rgb = preprocess_inputs(args.image, args.mask)

    negative_prompt = select_negative_prompt(args.negative_prompt)
    print(f"Positive prompt : {args.prompt}")
    print(f"Negative prompt : {negative_prompt}")
    print(f"Guidance scale  : {args.guidance_scale}")
    print(f"Resample steps  : {args.resample_steps}")
    print(f"Harmonize start : {args.harmonize_start} ({int(args.harmonize_start * args.steps)}/{args.steps} steps free)")

    print("Running RePaint inpainting loop...")
    result = repaint_inpaint(
        pipe=pipe,
        image=image,
        mask_L=mask_L,
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        resample_steps=args.resample_steps,
        harmonize_start=args.harmonize_start,
    )

    if args.text_overlay:
        print(f"Applying text overlay: '{args.text_overlay}'")
        result = render_text_overlay(result, mask_L, args.text_overlay)

    clean_prompt = "".join([c if c.isalnum() else "_" for c in args.prompt])
    filename = f"{clean_prompt[:50]}_seed{args.seed}_U{args.resample_steps}.png"
    output_path = os.path.join(args.output_dir, filename)
    result.save(output_path)
    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    main()
