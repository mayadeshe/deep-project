import argparse
import os
import torch
import torch.nn.functional as F
from utils.cli import preprocess_inputs, load_sd_pipeline


def get_structural_edge_map(image_tensor, mask):
    """Soft Edge Propagation — detect and propagate edges from known region"""
    gray = image_tensor.mean(dim=1, keepdim=True)

    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image_tensor.device).float().view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=image_tensor.device).float().view(1, 1, 3, 3)
    gx, gy = F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2)
    known_edges = mag * mask

    pooled_edges = F.max_pool2d(known_edges, kernel_size=15, stride=1, padding=7)

    kernel_size = 11
    sigma = 3.0
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=image_tensor.device).float()
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    gauss_kernel = (gauss.unsqueeze(1) * gauss.unsqueeze(0)).view(1, 1, kernel_size, kernel_size)

    edge_map = F.conv2d(pooled_edges, gauss_kernel, padding=kernel_size // 2)
    return edge_map


def permute_noise_by_edges(noise, edge_map):
    """Reorder noise amplitudes by edge strength"""
    b, c, h, w = noise.shape
    flat_noise = noise.view(b, c, -1)
    noise_idx = torch.argsort(flat_noise.abs(), dim=-1)
    flat_edges = edge_map.repeat(1, c, 1, 1).view(b, c, -1)
    edge_idx = torch.argsort(flat_edges, dim=-1)
    permuted_noise = torch.zeros_like(flat_noise)
    for i in range(b):
        for j in range(c):
            permuted_noise[i, j, edge_idx[i, j]] = flat_noise[i, j, noise_idx[i, j]]
    return permuted_noise.view(b, c, h, w)


@torch.no_grad()
def ddpm_structural_noise_inpaint(
        pipe, image, mask, prompt, steps=50, guidance_scale=7.5, seed=42,
):
    """Vanilla DDPM inpainting + structural noise permutation only."""
    device = pipe.device
    generator = torch.Generator(device).manual_seed(seed)

    # Prepare masked image
    image_tensor = pipe.image_processor.preprocess(image).to(device)
    mask = mask.to(device)[:, :1, :, :]
    image_tensor = image_tensor * mask

    # Compute edge map at latent resolution
    edge_map = F.interpolate(get_structural_edge_map(image_tensor, mask), size=(64, 64), mode="bilinear")

    # Encode known latents
    known_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator) * pipe.vae.config.scaling_factor
    mask_latent = F.interpolate(mask, size=(64, 64), mode="nearest")

    # Structural noise initialization (the only improvement)
    noise = torch.randn(known_latents.shape, device=device, generator=generator)
    latents = permute_noise_by_edges(noise, edge_map)

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
    parser = argparse.ArgumentParser("Structural Noise DDPM Inpainting (Ablation)")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="output_structural_noise")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    pipe = load_sd_pipeline(device)
    image, mask = preprocess_inputs(args.image, args.mask)

    result = ddpm_structural_noise_inpaint(
        pipe, image, mask, args.prompt,
        steps=args.steps, guidance_scale=args.guidance_scale, seed=args.seed,
    )
    out_path = os.path.join(args.output_dir, f"inpaint_seed{args.seed}.png")
    result.save(out_path)
    print(f"Saved result to: {out_path}")


if __name__ == "__main__":
    main()
