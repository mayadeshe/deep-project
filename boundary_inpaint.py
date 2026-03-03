import argparse
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

    Inside the mask, pixels far from the boundary get strength=1.0 (hard constraint).
    Pixels near the boundary (within soft_zone_pixels) ramp down to 0.0.
    Outside the mask, strength is 0.0 (no constraint — inpainted region is free).

    Args:
        mask_tensor: (1, 1, H, W) float32, 1=keep, 0=inpaint
        soft_zone_pixels: width of the soft transition zone in pixels

    Returns:
        soft_strength: (1, 1, H, W) float32 in [0, 1]
    """
    mask_np = mask_tensor.squeeze().cpu().numpy()  # (H, W), 1=keep, 0=inpaint

    # Distance from boundary, measured inside the known region
    dist_inside   = distance_transform_edt(mask_np)         # >0 inside, 0 on boundary/outside

    # Ramp: 0 at boundary → 1 at soft_zone_pixels away from boundary (inside known region)
    soft_strength = np.clip(dist_inside / soft_zone_pixels, 0.0, 1.0)

    # Only apply softening within the known region (mask=1); outside stays 0
    soft_strength = soft_strength * mask_np

    return torch.from_numpy(soft_strength.astype(np.float32)).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------
# DDPM Inpainting with Distance-Transform Boundary Softening
# ---------------------------------------------------------

@torch.no_grad()
def ddpm_inpaint_boundary(
        pipe,
        image: Image.Image,
        mask: torch.Tensor,
        prompt: str,
        steps: int,
        guidance_scale: float,
        seed: int,
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

    for t in pipe.scheduler.timesteps:

        # Noisy version of known region at timestep t
        noise = torch.randn(
            known_latents.shape,
            generator=generator,
            device=device,
            dtype=known_latents.dtype,
        )
        noisy_known = pipe.scheduler.add_noise(known_latents, noise, t)

        # Soft boundary blend: strength tapers to 0 near mask edges
        latents = latents + soft_strength_lat * mask_lat * (noisy_known - latents)

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

    # Final soft composite toward clean known latents
    latents = latents + soft_strength_lat * mask_lat * (known_latents - latents)

    # Decode
    latents /= pipe.vae.config.scaling_factor
    decoded = pipe.vae.decode(latents).sample
    image = pipe.image_processor.postprocess(decoded)[0]

    return image


# ---------------------------------------------------------
# CLI SCRIPT
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("DDPM Inpainting — Distance-Transform Boundary Softening")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="output_boundary")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--soft_zone_pixels", type=int, default=12,
                        help="Pixel width of the soft transition zone at the mask boundary.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading diffusion model...")
    pipe = load_sd_pipeline(device)

    print("Preprocessing inputs...")
    image, mask = preprocess_inputs(args.image, args.mask)

    print("Running boundary-softening DDPM inpainting...")
    result = ddpm_inpaint_boundary(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=args.prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        soft_zone_pixels=args.soft_zone_pixels,
    )

    out_path = os.path.join(args.output_dir, f"inpaint_seed{args.seed}.png")
    result.save(out_path)
    print(f"Saved result to: {out_path}")


if __name__ == "__main__":
    main()
