import argparse
import os
import torch
from PIL import Image

from utils.cli import preprocess_inputs, load_sd_pipeline


# ---------------------------------------------------------
# DDPM inpainting with Soft Mask Bias
# --------------------------------------------------------

@torch.no_grad()
def ddpm_inpaint(
        pipe,
        image: Image.Image,
        mask: torch.Tensor,
        prompt: str,
        steps: int,
        guidance_scale: float,
        seed: int,
        mask_bias_strength: float = 0.7,
):
    device = pipe.device
    generator = torch.Generator(device).manual_seed(seed)

    # Prepare masked image
    image_tensor = pipe.image_processor.preprocess(image).to(device)
    mask = mask.to(device)
    image_tensor = image_tensor * mask

    # Encode
    known_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator)
    known_latents *= pipe.vae.config.scaling_factor

    # Downsample mask to latent resolution
    mask = torch.nn.functional.interpolate(
        mask,
        size=known_latents.shape[2:],
        mode="nearest"
    )

    # Initial pure noise
    latents = torch.randn(
        known_latents.shape,
        generator=generator,
        device=device,
        dtype=known_latents.dtype
    )

    # Text embeddings
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True
    )
    text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])

    pipe.scheduler.set_timesteps(steps)

    for t in pipe.scheduler.timesteps:

        # Generate noisy version of known region at timestep t
        noise = torch.randn(
            known_latents.shape,
            generator=generator,
            device=device,
            dtype=known_latents.dtype
        )
        noisy_known = pipe.scheduler.add_noise(known_latents, noise, t)

        # -------------------------------------------------
        # SOFT MASK BIAS (במקום hard replace)
        # z = z + alpha * mask * (noisy_known - z)
        # -------------------------------------------------
        latents = latents + mask_bias_strength * mask * (noisy_known - latents)

        # Predict noise
        latent_input = torch.cat([latents] * 2)
        noise_pred = pipe.unet(
            latent_input,
            t,
            encoder_hidden_states=text_embeddings
        ).sample

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # Reverse diffusion step
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Final soft blend toward clean known latents
    latents = latents + mask_bias_strength * mask * (known_latents - latents)

    # Decode
    latents /= pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents).sample
    image = pipe.image_processor.postprocess(image)[0]

    return image


# ---------------------------------------------------------
# CLI SCRIPT
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("DDPM Inpainting with Soft Mask Bias")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="output_ddpm_bias")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_bias_strength", type=float, default=0.7)  # ← חדש
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading vanilla diffusion model...")
    pipe = load_sd_pipeline(device)

    print("Preprocessing inputs...")
    image, mask = preprocess_inputs(args.image, args.mask)

    print("Running DDPM inpainting with soft mask bias...")
    result = ddpm_inpaint(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=args.prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        mask_bias_strength=args.mask_bias_strength,
    )

    out_path = os.path.join(args.output_dir, f"inpaint_seed{args.seed}.png")
    result.save(out_path)
    print(f"Saved result to: {out_path}")


if __name__ == "__main__":
    main()