import torch
import torch.nn.functional as F

from utils import (
    get_local_boundary_color_map,
    get_structural_edge_map,
    permute_noise_by_edges,
    run_inpaint_cli,
)


@torch.no_grad()
def ddpm_structural_lgg_inpaint(
        pipe, image, mask, prompt, steps=50, guidance_scale=7.5, seed=42,
        lambda_grad=0.1, color_factor=0.2,
):
    device = pipe.device
    generator = torch.Generator(device).manual_seed(seed)

    image_tensor = pipe.image_processor.preprocess(image).to(device)
    mask = mask.to(device)[:, :1, :, :]
    image_tensor = image_tensor * mask

    edge_map = F.interpolate(get_structural_edge_map(image_tensor, mask), size=(64, 64), mode="bilinear")
    color_map = get_local_boundary_color_map(image_tensor, mask)

    known_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator) * pipe.vae.config.scaling_factor
    mask_latent = F.interpolate(mask, size=(64, 64), mode="nearest")

    noise = permute_noise_by_edges(torch.randn(known_latents.shape, device=device), edge_map)
    color_latent = pipe.vae.encode(color_map).latent_dist.sample(generator) * pipe.vae.config.scaling_factor
    latents = (1 - color_factor) * noise + color_factor * color_latent
    latents = pipe.scheduler.add_noise(known_latents, latents, pipe.scheduler.timesteps[0])

    prompt_embeds, neg_embeds = pipe.encode_prompt(prompt, device, 1, True)
    text_embeddings = torch.cat([neg_embeds, prompt_embeds])
    pipe.scheduler.set_timesteps(steps)

    for i, t in enumerate(pipe.scheduler.timesteps):
        noise_step = torch.randn(latents.shape, generator=generator, device=device)
        noisy_known = pipe.scheduler.add_noise(known_latents, noise_step, t)
        latents = (mask_latent * noisy_known) + ((1 - mask_latent) * latents)

        with torch.enable_grad():
            latents.requires_grad_(True)
            l_gx = latents[:, :, :, 1:] - latents[:, :, :, :-1]
            l_gy = latents[:, :, 1:, :] - latents[:, :, :-1, :]
            t_gx = noisy_known[:, :, :, 1:] - noisy_known[:, :, :, :-1]
            t_gy = noisy_known[:, :, 1:, :] - noisy_known[:, :, :-1, :]
            loss = F.mse_loss(l_gx * mask_latent[:, :, :, 1:], t_gx * mask_latent[:, :, :, 1:]) + \
                   F.mse_loss(l_gy * mask_latent[:, :, 1:, :], t_gy * mask_latent[:, :, 1:, :])
            grad = torch.autograd.grad(loss, latents)[0]
            latents = (latents - lambda_grad * grad).detach()

        latent_input = torch.cat([latents] * 2)
        noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    result_latents = (mask_latent * known_latents) + ((1 - mask_latent) * latents)
    result_img = \
        pipe.image_processor.postprocess(pipe.vae.decode(result_latents / pipe.vae.config.scaling_factor).sample)[0]

    return result_img


if __name__ == "__main__":
    run_inpaint_cli(
        "Structural + LGG DDPM Inpainting (Local Color Map)",
        ddpm_structural_lgg_inpaint,
        default_output_dir="output_local_color",
        extra_args=[
            ("--lambda_grad",  {"type": float, "default": 0.1}),
            ("--color_factor", {"type": float, "default": 0.2}),
        ],
    )
