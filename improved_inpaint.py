import argparse
import os
import torch
from PIL import Image
from torchvision.transforms.functional import gaussian_blur

from cliutils import preprocess_inputs, load_sd_pipeline


# ---------------------------------------------------------
# Improved DDPM inpainting sampler
# Adds: RePaint resampling + Gaussian soft mask blending
# ---------------------------------------------------------

@torch.no_grad()
def ddpm_inpaint_improved(
        pipe,
        image: Image.Image,
        mask: torch.Tensor,
        prompt: str,
        steps: int,
        guidance_scale: float,
        seed: int,
        resample_steps: int,
        blur_sigma: float,
        blur_kernel: int,
) -> Image.Image:

    device = pipe.device
    generator = torch.Generator(device).manual_seed(seed)

    # Prepare masked image
    image_tensor = pipe.image_processor.preprocess(image).to(device)
    mask = mask.to(device)
    image_tensor = image_tensor * mask

    # Encode
    known_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator)
    known_latents *= pipe.vae.config.scaling_factor

    # Soft mask: add gaussian blur in pixel space
    if blur_kernel > 1 and blur_sigma > 0.0:
        mask = gaussian_blur(mask, kernel_size=[blur_kernel], sigma=[blur_sigma])

    # Downsample mask to latent resolution
    mask = torch.nn.functional.interpolate(mask, size=known_latents.shape[2:], mode="bilinear", align_corners=False)

    # Initial pure noise
    latents = torch.randn(known_latents.shape, generator=generator, device=device, dtype=known_latents.dtype)

    # Text embeddings
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])

    pipe.scheduler.set_timesteps(steps)
    timesteps = list(pipe.scheduler.timesteps)

    for i, t in enumerate(timesteps):

        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0)

        for r in range(resample_steps):

            # Clamp known region at noise level t
            noise = torch.randn(known_latents.shape, generator=generator, device=device, dtype=known_latents.dtype)
            noisy_known = pipe.scheduler.add_noise(known_latents, noise, t)
            latents = (mask * noisy_known) + ((1 - mask) * latents)

            # Predict noise
            latent_input = torch.cat([latents] * 2)
            noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # Reverse sample
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

            # RePaint
            if r < resample_steps - 1 and t_prev > 0:

                jump_noise = torch.randn(known_latents.shape, generator=generator, device=device, dtype=known_latents.dtype)

                # Calculate effective beta
                alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = pipe.scheduler.alphas_cumprod[t_prev]
                effective_alpha = alpha_prod_t / alpha_prod_t_prev
                effective_beta = torch.clamp(1.0 - effective_alpha, min=0.0, max=1.0)

                # noise back the latents
                latents = torch.sqrt(1 - effective_beta) * latents + torch.sqrt(effective_beta) * jump_noise

    latents = (mask * known_latents) + ((1 - mask) * latents)

    # Decode the latents
    latents /= pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents).sample
    image = pipe.image_processor.postprocess(image)[0]

    return image


# ---------------------------------------------------------
# CLI SCRIPT
# ---------------------------------------------------------
def odd_kernel(value):
    k = int(value)
    if k % 2 == 0:
        raise argparse.ArgumentTypeError(f"--blur-kernel must be odd, got {k}")
    return k


def main():
    parser = argparse.ArgumentParser("Improved DDPM Inpainting (RePaint + soft mask)")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="output_improved_inpaint")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resample-steps", type=int, default=10,
                        help="RePaint resampling iterations per timestep (r). Default: 10.")
    parser.add_argument("--blur-sigma", type=float, default=1.5,
                        help="Gaussian sigma for soft latent mask blending. Default: 1.5.")
    parser.add_argument("--blur-kernel", type=odd_kernel, default=7,
                        help="Gaussian kernel size (must be odd). Default: 7.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading diffusion model...")
    pipe = load_sd_pipeline(device)

    print("Preprocessing inputs...")
    image, mask = preprocess_inputs(args.image, args.mask)

    print(f"Running improved DDPM inpainting (resample_steps={args.resample_steps}, "
          f"blur_sigma={args.blur_sigma}, blur_kernel={args.blur_kernel})...")
    result = ddpm_inpaint_improved(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=args.prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        resample_steps=args.resample_steps,
        blur_sigma=args.blur_sigma,
        blur_kernel=args.blur_kernel,
    )

    out_path = os.path.join(args.output_dir, f"inpaint_seed{args.seed}.png")
    result.save(out_path)
    print(f"Saved result to: {out_path}")


if __name__ == "__main__":
    main()
