import argparse
import os
import torch
import torch.nn.functional as F
from utils.cli import preprocess_inputs, load_sd_pipeline


@torch.no_grad()
def ddpm_lgg_inpaint(
        pipe, image, mask, prompt, steps=50, guidance_scale=7.5, seed=42,
        lambda_grad=0.1,
):
    """Vanilla DDPM inpainting + Latent Gradient Guidance (LGG) only."""
    device = pipe.device
    generator = torch.Generator(device).manual_seed(seed)

    # Prepare masked image
    image_tensor = pipe.image_processor.preprocess(image).to(device)
    mask = mask.to(device)[:, :1, :, :]
    image_tensor = image_tensor * mask

    # Encode known latents
    known_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator) * pipe.vae.config.scaling_factor
    mask_latent = F.interpolate(mask, size=(64, 64), mode="nearest")

    # Standard random noise initialization (no structural permutation, no color prior)
    latents = torch.randn(known_latents.shape, device=device, generator=generator, dtype=known_latents.dtype)

    # Text embeddings
    prompt_embeds, neg_embeds = pipe.encode_prompt(prompt, device, 1, True,
                                                   negative_prompt="blurry, smudge, bad quality")
    text_embeddings = torch.cat([neg_embeds, prompt_embeds])
    pipe.scheduler.set_timesteps(steps)

    for t in pipe.scheduler.timesteps:
        # Inject known region
        noise_step = torch.randn(latents.shape, generator=generator, device=device)
        noisy_known = pipe.scheduler.add_noise(known_latents, noise_step, t)
        latents = (mask_latent * noisy_known) + ((1 - mask_latent) * latents)

        # Latent Gradient Guidance (the only improvement)
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

        # UNet step
        latent_input = torch.cat([latents] * 2)
        noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Hard composite
    result_latents = (mask_latent * known_latents) + ((1 - mask_latent) * latents)
    result_img = pipe.image_processor.postprocess(
        pipe.vae.decode(result_latents / pipe.vae.config.scaling_factor).sample
    )[0]

    return result_img


def main():
    parser = argparse.ArgumentParser("LGG DDPM Inpainting (Ablation)")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="output_lgg_only")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_grad", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    pipe = load_sd_pipeline(device)
    image, mask = preprocess_inputs(args.image, args.mask)

    result = ddpm_lgg_inpaint(
        pipe, image, mask, args.prompt,
        steps=args.steps, guidance_scale=args.guidance_scale, seed=args.seed,
        lambda_grad=args.lambda_grad,
    )
    out_path = os.path.join(args.output_dir, f"inpaint_seed{args.seed}.png")
    result.save(out_path)
    print(f"Saved result to: {out_path}")


if __name__ == "__main__":
    main()
