import math

import torch
import torch.nn.functional as F

from utils import (
    get_local_boundary_color_map,
    get_structural_edge_map,
    permute_noise_by_edges,
    run_inpaint_cli,
)


NEGATIVE_PROMPT = "blurry, low quality, artifacts, seam, border, distorted, ugly, watermark"


@torch.no_grad()
def ddpm_inpaint_final(
        pipe,
        image,
        mask,
        prompt,
        steps=50,
        guidance_scale=7.5,
        seed=42,
        resample_steps=5,
        lambda_grad=0.1,
        color_factor=0.2,
):
    device = pipe.device
    generator = torch.Generator(device).manual_seed(seed)

    image_tensor = pipe.image_processor.preprocess(image).to(device)
    mask = mask.to(device)[:, :1, :, :]
    image_tensor = image_tensor * mask

    edge_map  = F.interpolate(get_structural_edge_map(image_tensor, mask), size=(64, 64), mode="bilinear")
    color_map = get_local_boundary_color_map(image_tensor, mask)

    known_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator) * pipe.vae.config.scaling_factor

    # Two mask variants at latent resolution: soft (sin/cos for TrigoBlend) and hard (for LGG / final composite)
    mask_soft = F.interpolate(mask, size=known_latents.shape[2:], mode="bilinear", align_corners=False)
    theta     = mask_soft * (math.pi / 2.0)
    mask_sin  = torch.sin(theta)
    mask_cos  = torch.cos(theta)
    mask_hard = (mask_soft > 0.5).float()

    noise        = permute_noise_by_edges(torch.randn(known_latents.shape, device=device), edge_map)
    color_latent = pipe.vae.encode(color_map).latent_dist.sample(generator) * pipe.vae.config.scaling_factor
    latents      = (1 - color_factor) * noise + color_factor * color_latent

    pipe.scheduler.set_timesteps(steps)
    timesteps = list(pipe.scheduler.timesteps)

    latents = pipe.scheduler.add_noise(known_latents, latents, timesteps[0])

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=NEGATIVE_PROMPT,
    )
    text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])

    for i, t in enumerate(timesteps):

        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0)

        for r in range(resample_steps):

            step_noise  = torch.randn(known_latents.shape, generator=generator, device=device, dtype=known_latents.dtype)
            noisy_known = pipe.scheduler.add_noise(known_latents, step_noise, t)
            latents     = (mask_sin * noisy_known) + (mask_cos * latents)

            with torch.enable_grad():
                latents.requires_grad_(True)
                l_gx = latents[:, :, :, 1:]     - latents[:, :, :, :-1]
                l_gy = latents[:, :, 1:, :]     - latents[:, :, :-1, :]
                t_gx = noisy_known[:, :, :, 1:] - noisy_known[:, :, :, :-1]
                t_gy = noisy_known[:, :, 1:, :] - noisy_known[:, :, :-1, :]
                loss = F.mse_loss(l_gx * mask_hard[:, :, :, 1:], t_gx * mask_hard[:, :, :, 1:]) + \
                       F.mse_loss(l_gy * mask_hard[:, :, 1:, :], t_gy * mask_hard[:, :, 1:, :])
                grad = torch.autograd.grad(loss, latents)[0]
                latents = (latents - lambda_grad * grad).detach()

            latent_input = torch.cat([latents] * 2)
            noise_pred   = pipe.unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
            noise_neg, noise_text = noise_pred.chunk(2)
            noise_pred   = noise_neg + guidance_scale * (noise_text - noise_neg)

            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

            # RePaint jump-back between resamples
            if r < resample_steps - 1 and t_prev > 0:
                jump_noise = torch.randn(known_latents.shape, generator=generator, device=device, dtype=known_latents.dtype)
                alpha_prod_t      = pipe.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = pipe.scheduler.alphas_cumprod[t_prev]
                effective_alpha   = alpha_prod_t / alpha_prod_t_prev
                effective_beta    = torch.clamp(1.0 - effective_alpha, min=0.0, max=1.0)
                latents = torch.sqrt(1 - effective_beta) * latents + torch.sqrt(effective_beta) * jump_noise

    result_latents = (mask_hard * known_latents) + ((1 - mask_hard) * latents)
    result_latents = result_latents / pipe.vae.config.scaling_factor
    decoded = pipe.vae.decode(result_latents).sample
    return pipe.image_processor.postprocess(decoded)[0]


if __name__ == "__main__":
    run_inpaint_cli(
        "Final DDPM Inpainting",
        ddpm_inpaint_final,
        default_output_dir="output_full_inpaint",
        extra_args=[
            ("--resample-steps", {"type": int,   "default": 5}),
            ("--lambda_grad",    {"type": float, "default": 0.1}),
            ("--color_factor",   {"type": float, "default": 0.2}),
        ],
    )
