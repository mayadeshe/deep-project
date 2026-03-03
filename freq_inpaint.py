import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image

from utils.cli import preprocess_inputs, load_sd_pipeline


# ---------------------------------------------------------
# Helper: Low-pass filter via average pooling
# ---------------------------------------------------------

def low_pass_filter(tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Apply a box low-pass filter to a (B, C, H, W) tensor.
    Uses reflect padding to avoid border artifacts.
    """
    pad = kernel_size // 2
    padded = F.pad(tensor, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1, padding=0)


# ---------------------------------------------------------
# DDPM Inpainting with Frequency-Aware Blending
#
# Idea: decompose latents into low + high frequency components,
# then blend each band separately.
#
#   low_alpha  controls how strongly low frequencies (global structure/color)
#              are constrained to match the known region.
#   high_alpha controls how strongly high frequencies (edges/texture)
#              are constrained to match the known region.
#
# Setting low_alpha < high_alpha lets the network produce a globally
# coherent fill while keeping sharp edges from the original.
# ---------------------------------------------------------

@torch.no_grad()
def ddpm_inpaint_freq(
        pipe,
        image: Image.Image,
        mask: torch.Tensor,
        prompt: str,
        steps: int,
        guidance_scale: float,
        seed: int,
        low_alpha: float = 0.85,
        high_alpha: float = 1.0,
        kernel_size: int = 5,
) -> Image.Image:

    device = pipe.device
    generator = torch.Generator(device).manual_seed(seed)

    negative_prompt = "blurry, low quality, artifacts, seam, border, distorted, ugly, watermark"

    # Prepare masked image
    image_tensor = pipe.image_processor.preprocess(image).to(device)
    mask = mask.to(device)
    image_tensor = image_tensor * mask

    # Encode known region
    known_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator)
    known_latents *= pipe.vae.config.scaling_factor

    # Downsample mask to latent resolution
    mask_lat = torch.nn.functional.interpolate(
        mask, size=known_latents.shape[2:], mode="nearest"
    )

    # Initial pure noise
    latents = torch.randn(
        known_latents.shape,
        generator=generator,
        device=device,
        dtype=known_latents.dtype,
    )

    # Text embeddings with negative prompt CFG
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )
    text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])

    pipe.scheduler.set_timesteps(steps)

    num_steps = len(pipe.scheduler.timesteps)
    for i, t in enumerate(pipe.scheduler.timesteps):
        progress = (i + 1) / num_steps  # ramps 0 → 1 over denoising steps

        # Noisy version of known region at timestep t
        noise = torch.randn(
            known_latents.shape,
            generator=generator,
            device=device,
            dtype=known_latents.dtype,
        )
        noisy_known = pipe.scheduler.add_noise(known_latents, noise, t)

        # Frequency decomposition of noisy_known
        low_known = low_pass_filter(noisy_known, kernel_size)
        high_known = noisy_known - low_known

        # Frequency decomposition of current latents
        low_lat = low_pass_filter(latents, kernel_size)
        high_lat = latents - low_lat

        # Per-band blend inside known region — scaled by progress (near-0 early, full late)
        low_blend  = low_lat  + (low_alpha  * progress) * mask_lat * (low_known  - low_lat)
        high_blend = high_lat + (high_alpha * progress) * mask_lat * (high_known - high_lat)
        latents = low_blend + high_blend

        # Predict noise (CFG)
        latent_input = torch.cat([latents] * 2)
        noise_pred = pipe.unet(
            latent_input,
            t,
            encoder_hidden_states=text_embeddings,
        ).sample

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # Reverse diffusion step
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Final hard composite with clean known latents
    latents = (mask_lat * known_latents) + ((1 - mask_lat) * latents)

    # Decode
    latents /= pipe.vae.config.scaling_factor
    decoded = pipe.vae.decode(latents).sample
    image = pipe.image_processor.postprocess(decoded)[0]

    return image


# ---------------------------------------------------------
# CLI SCRIPT
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("DDPM Inpainting — Frequency-Aware Blending")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="output_freq")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low_alpha", type=float, default=0.85,
                        help="Blend strength for low-frequency (global structure) band.")
    parser.add_argument("--high_alpha", type=float, default=1.0,
                        help="Blend strength for high-frequency (edges/texture) band.")
    parser.add_argument("--kernel_size", type=int, default=5,
                        help="Kernel size for the box low-pass filter (odd number).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading diffusion model...")
    pipe = load_sd_pipeline(device)

    print("Preprocessing inputs...")
    image, mask = preprocess_inputs(args.image, args.mask)

    print("Running frequency-aware DDPM inpainting...")
    result = ddpm_inpaint_freq(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=args.prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        low_alpha=args.low_alpha,
        high_alpha=args.high_alpha,
        kernel_size=args.kernel_size,
    )

    out_path = os.path.join(args.output_dir, f"inpaint_seed{args.seed}.png")
    result.save(out_path)
    print(f"Saved result to: {out_path}")


if __name__ == "__main__":
    main()
