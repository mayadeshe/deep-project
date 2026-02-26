import argparse
import math
import os
import torch
import torch.nn.functional as F
from PIL import Image

from utils.cli import preprocess_inputs, load_sd_pipeline


# ---------------------------------------------------------
# Mask schedule builder
# ---------------------------------------------------------

def build_mask_schedule(
        mask_hr: torch.Tensor,
        steps: int,
        max_radius: int,
        decay_steps: int,
) -> list:
    """Pre-compute one 64x64 dilated mask tensor per timestep.

    At timestep index i (0 = noisiest, steps-1 = cleanest):
        R_i = max_radius * 0.5 * (1 + cos(pi * i / decay_steps))  if i <= decay_steps
        R_i = 0                                                      otherwise

    Dilation (expanding the hole = eroding the keep region):
        M_dilated = 1 - max_pool2d(1 - M_HR, kernel_size=2*R+1, padding=R)

    The result is bilinearly downsampled to 64x64 latent resolution.
    """
    schedule = []
    for i in range(steps):
        if i <= decay_steps:
            R = int(round(max_radius * 0.5 * (1 + math.cos(math.pi * i / decay_steps))))
        else:
            R = 0

        if R > 0:
            k = 2 * R + 1
            # Erode the keep-mask (expand the hole)
            dilated_hr = 1.0 - F.max_pool2d(1.0 - mask_hr, kernel_size=k, stride=1, padding=R)
        else:
            dilated_hr = mask_hr.clone()

        # Downsample to latent resolution
        dilated_latent = F.interpolate(dilated_hr, size=(64, 64), mode="bilinear", align_corners=False)
        schedule.append(dilated_latent)

    return schedule


# ---------------------------------------------------------
# Novel DDPM inpainting sampler
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
        max_radius: int,
        decay_steps: int,
) -> Image.Image:
    """DDPM inpainting with cosine-decayed mask dilation.

    Novel contribution: at early (noisy) timesteps the mask is dilated so the
    model can anchor generated content into surrounding context. The dilation
    radius decays to zero via a cosine schedule, letting edges snap back to the
    true mask boundary for fine detail.
    """
    device = pipe.device
    generator = torch.Generator(device).manual_seed(seed)

    # Prepare masked image (hole pixels zeroed out)
    image_tensor = pipe.image_processor.preprocess(image).to(device)
    mask = mask.to(device)
    image_tensor = image_tensor * mask

    # Encode to latent space
    known_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator)
    known_latents *= pipe.vae.config.scaling_factor

    # Build the per-step dilated mask schedule at pixel resolution,
    # then downsample each entry to latent resolution.
    # mask shape: (1, 1, H, W) in pixel space
    mask_schedule = build_mask_schedule(mask, steps, max_radius, decay_steps)
    mask_schedule = [m.to(device=device, dtype=known_latents.dtype) for m in mask_schedule]

    # Undilated latent-space mask for final compositing
    mask_latent = F.interpolate(mask, size=known_latents.shape[2:], mode="bilinear", align_corners=False)
    mask_latent = mask_latent.to(device=device, dtype=known_latents.dtype)

    # Initial pure noise
    latents = torch.randn(known_latents.shape, generator=generator, device=device, dtype=known_latents.dtype)

    # Text embeddings
    negative_prompt = "blurry, low quality, artifacts, seam, border, distorted, ugly, watermark"
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
        # Use the dilated mask for this timestep
        mask_i = mask_schedule[i]

        # Clamp known region at noise level t using dilated mask
        noise = torch.randn(known_latents.shape, generator=generator, device=device, dtype=known_latents.dtype)
        noisy_known = pipe.scheduler.add_noise(known_latents, noise, t)
        latents = (mask_i * noisy_known) + ((1 - mask_i) * latents)

        # Predict noise (CFG)
        latent_input = torch.cat([latents] * 2)
        noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
        noise_neg, noise_text = noise_pred.chunk(2)
        noise_pred = noise_neg + guidance_scale * (noise_text - noise_neg)

        # Reverse sample
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Final composite with the original undilated latent mask
    latents = (mask_latent * known_latents) + ((1 - mask_latent) * latents)

    # Decode
    latents /= pipe.vae.config.scaling_factor
    decoded = pipe.vae.decode(latents).sample
    image = pipe.image_processor.postprocess(decoded)[0]

    return image


# ---------------------------------------------------------
# CLI SCRIPT
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Novel DDPM Inpainting (cosine-decayed mask dilation)")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="output_novel_inpaint")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-radius", type=int, default=10,
                        help="Maximum dilation radius in pixel space (512x512). Default: 10.")
    parser.add_argument("--decay-steps", type=int, default=35,
                        help="Number of timesteps over which the dilation decays to zero. Default: 35.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading diffusion model...")
    pipe = load_sd_pipeline(device)

    print("Preprocessing inputs...")
    image, mask = preprocess_inputs(args.image, args.mask)

    print(f"Running novel DDPM inpainting (max_radius={args.max_radius}, decay_steps={args.decay_steps})...")
    result = ddpm_inpaint_novel(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=args.prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        max_radius=args.max_radius,
        decay_steps=args.decay_steps,
    )

    out_path = os.path.join(args.output_dir, f"inpaint_seed{args.seed}.png")
    result.save(out_path)
    print(f"Saved result to: {out_path}")


if __name__ == "__main__":
    main()
