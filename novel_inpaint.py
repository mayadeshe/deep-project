import argparse
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt

from utils.cli import preprocess_inputs, load_sd_pipeline


# ---------------------------------------------------------
# Helper: Distance-Transform Soft Strength Map
# ---------------------------------------------------------

def make_soft_strength(mask_tensor: torch.Tensor, soft_zone_pixels: int) -> torch.Tensor:
    """
    Compute a soft blend strength based on distance from the mask boundary.

    Inside the mask, pixels far from the boundary get strength=1.0.
    Pixels near the boundary (within soft_zone_pixels) ramp down to 0.0.
    Outside the mask, strength is 0.0.

    Args:
        mask_tensor: (1, 1, H, W) float32, 1=keep, 0=inpaint
        soft_zone_pixels: width of the soft transition zone in pixels

    Returns:
        soft_strength: (1, 1, H, W) float32 in [0, 1]
    """
    mask_np = mask_tensor.squeeze().cpu().numpy()  # (H, W), 1=keep, 0=inpaint

    dist_inside = distance_transform_edt(mask_np)

    soft_strength = np.clip(dist_inside / soft_zone_pixels, 0.0, 1.0)
    soft_strength = soft_strength * mask_np

    return torch.from_numpy(soft_strength.astype(np.float32)).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------
# Novel DDPM Inpainting
# RePaint resampling + TrigoBlend + Boundary Softening
# ---------------------------------------------------------

@torch.no_grad()
def ddpm_inpaint_novel(
        pipe,
        image: Image.Image,
        mask: torch.Tensor,
        prompt: str,
        steps: int,
        guidance_scale: float,
        seed: int,
        resample_steps: int,
        soft_zone_pixels: int = 12,
) -> Image.Image:

    device = pipe.device
    generator = torch.Generator(device).manual_seed(seed)

    negative_prompt = "blurry, low quality, artifacts, seam, border, distorted, ugly, watermark"

    # Compute soft strength at pixel resolution before moving mask to device
    soft_strength_pix = make_soft_strength(mask, soft_zone_pixels)  # (1,1,H,W) on cpu

    # Prepare masked image
    image_tensor = pipe.image_processor.preprocess(image).to(device)
    mask = mask.to(device)
    image_tensor = image_tensor * mask

    # Encode known region
    known_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator)
    known_latents *= pipe.vae.config.scaling_factor

    lat_h, lat_w = known_latents.shape[2], known_latents.shape[3]

    # Downsample mask and soft_strength to latent resolution
    mask_lat = F.interpolate(mask, size=(lat_h, lat_w), mode="nearest")
    soft_strength_lat = F.interpolate(
        soft_strength_pix.to(device),
        size=(lat_h, lat_w),
        mode="bilinear",
        align_corners=False,
    )

    # Trigo blend on boundary-softened mask: convert soft strength to sin/cos
    theta = soft_strength_lat * mask_lat * (math.pi / 2.0)
    blend_sin = torch.sin(theta)
    blend_cos = torch.cos(theta)

    # Initial pure noise
    latents = torch.randn(
        known_latents.shape,
        generator=generator,
        device=device,
        dtype=known_latents.dtype,
    )

    # Text embeddings (CFG with negative prompt)
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )
    text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])

    pipe.scheduler.set_timesteps(steps)
    timesteps = list(pipe.scheduler.timesteps)

    for i, t in enumerate(timesteps):

        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0)

        for r in range(resample_steps):

            # Clamp known region at noise level t
            noise = torch.randn(
                known_latents.shape,
                generator=generator,
                device=device,
                dtype=known_latents.dtype,
            )
            noisy_known = pipe.scheduler.add_noise(known_latents, noise, t)

            # Trigo boundary blend: sin/cos tapers near mask edges
            latents = (blend_sin * noisy_known) + (blend_cos * latents)

            # Predict noise (CFG)
            latent_input = torch.cat([latents] * 2)
            noise_pred = pipe.unet(
                latent_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample

            noise_neg, noise_text = noise_pred.chunk(2)
            noise_pred = noise_neg + guidance_scale * (noise_text - noise_neg)

            # Reverse diffusion step
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

            # RePaint resampling jump
            if r < resample_steps - 1 and t_prev > 0:

                jump_noise = torch.randn(
                    known_latents.shape,
                    generator=generator,
                    device=device,
                    dtype=known_latents.dtype,
                )

                alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = pipe.scheduler.alphas_cumprod[t_prev]
                effective_alpha = alpha_prod_t / alpha_prod_t_prev
                effective_beta = torch.clamp(1.0 - effective_alpha, min=0.0, max=1.0)

                latents = torch.sqrt(1 - effective_beta) * latents + torch.sqrt(effective_beta) * jump_noise

    # Final trigo composite toward clean known latents
    latents = (blend_sin * known_latents) + (blend_cos * latents)

    # Decode
    latents /= pipe.vae.config.scaling_factor
    decoded = pipe.vae.decode(latents).sample
    image = pipe.image_processor.postprocess(decoded)[0]

    return image


# ---------------------------------------------------------
# CLI SCRIPT
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        "Novel DDPM Inpainting (RePaint + TrigoBlend + Boundary Softening)"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="output_novel_comparison")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resample-steps", type=int, default=5,
                        help="RePaint resampling iterations per timestep (r). Default: 5.")
    parser.add_argument("--soft-zone-pixels", type=int, default=12,
                        help="Pixel width of the soft transition zone at the mask boundary.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading diffusion model...")
    pipe = load_sd_pipeline(device)

    print("Preprocessing inputs...")
    image, mask = preprocess_inputs(args.image, args.mask)

    print(f"Running novel DDPM inpainting (RePaint + TrigoBlend + Boundary Softening)...")
    result = ddpm_inpaint_novel(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=args.prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        resample_steps=args.resample_steps,
        soft_zone_pixels=args.soft_zone_pixels,
    )

    out_path = os.path.join(args.output_dir, f"inpaint_seed{args.seed}.png")
    result.save(out_path)
    print(f"Saved result to: {out_path}")


if __name__ == "__main__":
    main()