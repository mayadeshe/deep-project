import argparse
import os
import torch
from PIL import Image

from utils.cli import preprocess_inputs, load_sd_pipeline


# ---------------------------------------------------------
# DDPM Inpainting with Noise-Schedule-Annealed Constraint
#
# Idea: blend strength = alpha_bar_t ** gamma
#   At t=T (max noise): alpha_bar ≈ 0.005, blend ≈ 0.07  → weak constraint
#   At t=1 (clean):     alpha_bar ≈ 0.999, blend ≈ 0.9995 → near-hard constraint
#
# This lets the network explore freely early on and only tightly
# constrains the known region as denoising converges.
# ---------------------------------------------------------

@torch.no_grad()
def ddpm_inpaint_annealed(
        pipe,
        image: Image.Image,
        mask: torch.Tensor,
        prompt: str,
        steps: int,
        guidance_scale: float,
        seed: int,
        gamma: float = 0.5,
        resample_steps: int = 5,
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

    timesteps = list(pipe.scheduler.timesteps)
    for i, t in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0)

        for r in range(resample_steps):

            # Blend strength scales with alpha_bar_t^gamma
            alpha_bar_t = pipe.scheduler.alphas_cumprod[t].item()
            blend_alpha = alpha_bar_t ** gamma

            # Noisy version of known region at timestep t
            noise = torch.randn(
                known_latents.shape,
                generator=generator,
                device=device,
                dtype=known_latents.dtype,
            )
            noisy_known = pipe.scheduler.add_noise(known_latents, noise, t)

            # Annealed blend: weak at high noise, strong near clean
            latents = latents + blend_alpha * mask_lat * (noisy_known - latents)

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

            # RePaint jump-back (except on last resample iteration)
            if r < resample_steps - 1 and t_prev > 0:
                jump_noise = torch.randn(known_latents.shape, generator=generator, device=device, dtype=known_latents.dtype)
                alpha_prod_t      = pipe.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = pipe.scheduler.alphas_cumprod[t_prev]
                effective_alpha   = alpha_prod_t / alpha_prod_t_prev
                effective_beta    = torch.clamp(1.0 - effective_alpha, 0.0, 1.0)
                latents = torch.sqrt(1 - effective_beta) * latents + torch.sqrt(effective_beta) * jump_noise

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
    parser = argparse.ArgumentParser("DDPM Inpainting — Noise-Schedule-Annealed Constraint")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="output_annealed")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Exponent for annealing: blend = alpha_bar^gamma. "
                             "Higher = slower ramp-up of constraint.")
    parser.add_argument("--resample_steps", type=int, default=5,
                        help="Number of RePaint inner resample iterations per timestep.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading diffusion model...")
    pipe = load_sd_pipeline(device)

    print("Preprocessing inputs...")
    image, mask = preprocess_inputs(args.image, args.mask)

    print("Running annealed-constraint DDPM inpainting...")
    result = ddpm_inpaint_annealed(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=args.prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        gamma=args.gamma,
        resample_steps=args.resample_steps,
    )

    out_path = os.path.join(args.output_dir, f"inpaint_seed{args.seed}.png")
    result.save(out_path)
    print(f"Saved result to: {out_path}")


if __name__ == "__main__":
    main()
