import math
import torch

from utils import run_inpaint_cli


# ---------------------------------------------------------
# Improved DDPM inpainting sampler
# Adds: RePaint resampling + negative prompt CFG + TrigoBlend with SoftMask
# ---------------------------------------------------------

@torch.no_grad()
def ddpm_inpaint_improved(
        pipe,
        image,
        mask,
        prompt,
        steps,
        guidance_scale,
        seed,
        resample_steps,
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

    # Downsample mask to latent resolution for diffusion loop and to trigo points for trigo blend
    mask = torch.nn.functional.interpolate(mask, size=known_latents.shape[2:], mode="bilinear", align_corners=False)
    theta = mask * (math.pi / 2.0)
    mask_sin = torch.sin(theta)
    mask_cos = torch.cos(theta)

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

        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0)

        for r in range(resample_steps):

            # Clamp known region at noise level t
            noise = torch.randn(known_latents.shape, generator=generator, device=device, dtype=known_latents.dtype)
            noisy_known = pipe.scheduler.add_noise(known_latents, noise, t)
            latents = (mask_sin * noisy_known) + (mask_cos * latents)

            # Predict noise
            latent_input = torch.cat([latents] * 2)
            noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
            noise_neg, noise_text = noise_pred.chunk(2)
            noise_pred = noise_neg + guidance_scale * (noise_text - noise_neg)

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

    latents = (mask_sin * known_latents) + (mask_cos * latents) # Decode

    latents /= pipe.vae.config.scaling_factor
    decoded = pipe.vae.decode(latents).sample
    image = pipe.image_processor.postprocess(decoded)[0]

    return image


if __name__ == "__main__":
    run_inpaint_cli(
        "Improved DDPM Inpainting",
        ddpm_inpaint_improved,
        default_output_dir="output_sampling_refined_inpaint",
        extra_args=[
            ("--resample-steps", {"type": int, "default": 5}),
        ],
    )
