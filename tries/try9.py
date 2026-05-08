import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
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


def get_local_boundary_color_map(image_tensor, mask):
    """
    Build a local boundary color map using EDT (Euclidean Distance Transform)
    and nearest-neighbor assignment from scipy.

    For each pixel in the inpaint hole, assigns the color of the nearest
    boundary pixel — producing a spatially-varying color prior instead of
    a single global average.

    Args:
        image_tensor: [1, 3, H, W] tensor, known region (hole zeroed out)
        mask: [1, 1, H, W] tensor, 1=keep, 0=inpaint

    Returns:
        color_map: [1, 3, H, W] tensor — local color map for the hole region
    """
    from scipy.ndimage import distance_transform_edt

    # Get mask as numpy (H, W), 1=known, 0=hole
    mask_np = mask[0, 0].cpu().numpy()  # (H, W)

    # Compute boundary: known pixels adjacent to hole pixels
    kernel = np.ones((3, 3))
    from scipy.ndimage import binary_dilation
    dilated_hole = binary_dilation(mask_np == 0, structure=kernel)
    boundary_np = (mask_np == 1) & dilated_hole  # known pixels next to hole

    # EDT from hole pixels to nearest boundary pixel
    # indices gives coordinates of nearest boundary pixel for each position
    _, indices = distance_transform_edt(~boundary_np, return_distances=True, return_indices=True)

    # Build color map: for each pixel, look up the color at its nearest boundary pixel
    img_np = image_tensor[0].cpu().numpy()  # (3, H, W)
    color_map_np = np.zeros_like(img_np)
    for c in range(3):
        color_map_np[c] = img_np[c][indices[0], indices[1]]

    color_map = torch.from_numpy(color_map_np).unsqueeze(0).to(image_tensor.device)
    return color_map


@torch.no_grad()
def ddpm_structural_lgg_inpaint(
        pipe, image, mask, prompt, steps=50, guidance_scale=7.5, seed=42,
        lambda_grad=0.1, color_factor=0.2
):
    device = pipe.device
    generator = torch.Generator(device).manual_seed(seed)

    # 1. Sterile preparation — zero out the hole to prevent leakage
    image_tensor = pipe.image_processor.preprocess(image).to(device)
    mask = mask.to(device)[:, :1, :, :]

    # Physical reset of hidden region
    image_tensor = image_tensor * mask

    edge_map = F.interpolate(get_structural_edge_map(image_tensor, mask), size=(64, 64), mode="bilinear")

    # Local boundary color map (replaces global average)
    color_map = get_local_boundary_color_map(image_tensor, mask)

    # Encode known latents
    known_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator) * pipe.vae.config.scaling_factor
    mask_latent = F.interpolate(mask, size=(64, 64), mode="nearest")

    # 2. Structural noise initialization with local color prior
    noise = permute_noise_by_edges(torch.randn(known_latents.shape, device=device), edge_map)
    color_latent = pipe.vae.encode(color_map).latent_dist.sample(generator) * pipe.vae.config.scaling_factor
    latents = (1 - color_factor) * noise + color_factor * color_latent
    latents = pipe.scheduler.add_noise(known_latents, latents, pipe.scheduler.timesteps[0])

    prompt_embeds, neg_embeds = pipe.encode_prompt(prompt, device, 1, True,
                                                   negative_prompt="blurry, smudge, bad quality")
    text_embeddings = torch.cat([neg_embeds, prompt_embeds])
    pipe.scheduler.set_timesteps(steps)

    for i, t in enumerate(pipe.scheduler.timesteps):
        # Inject known region
        noise_step = torch.randn(latents.shape, generator=generator, device=device)
        noisy_known = pipe.scheduler.add_noise(known_latents, noise_step, t)
        latents = (mask_latent * noisy_known) + ((1 - mask_latent) * latents)

        # 3. Latent Gradient Guidance (LGG)
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

    # 4. Hard composite — NO Gaussian blur, NO information leakage
    result_latents = (mask_latent * known_latents) + ((1 - mask_latent) * latents)
    result_img = \
        pipe.image_processor.postprocess(pipe.vae.decode(result_latents / pipe.vae.config.scaling_factor).sample)[0]

    return result_img


def main():
    parser = argparse.ArgumentParser("Structural + LGG DDPM Inpainting (Local Color Map)")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="output_local_color")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_sd_pipeline(device)
    image, mask = preprocess_inputs(args.image, args.mask)

    result = ddpm_structural_lgg_inpaint(pipe, image, mask, args.prompt)
    result.save(os.path.join(args.output_dir, "local_color_result.png"))


if __name__ == "__main__":
    main()
