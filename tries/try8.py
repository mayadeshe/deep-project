import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from utils.cli import preprocess_inputs, load_sd_pipeline

def get_structural_edge_map(image_tensor, mask):
    gray = image_tensor.mean(dim=1, keepdim=True)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image_tensor.device).float().view(1,1,3,3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=image_tensor.device).float().view(1,1,3,3)
    gx, gy = F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx**2 + gy**2)
    known_edges = mag * mask
    edge_map = F.max_pool2d(known_edges, kernel_size=21, stride=1, padding=10)
    return edge_map

def permute_noise_by_edges(noise, edge_map):
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
def ddpm_structural_lgg_inpaint(
    pipe, image, mask, prompt, steps=50, guidance_scale=7.5, seed=42,
    lambda_grad=0.1, color_factor=0.2
):
    device = pipe.device
    generator = torch.Generator(device).manual_seed(seed)

    image_tensor = pipe.image_processor.preprocess(image).to(device)
    mask = mask.to(device)[:, :1, :, :]
    image_tensor = image_tensor * mask
    edge_map = F.interpolate(get_structural_edge_map(image_tensor, mask), size=(64, 64), mode="bilinear")

    kernel = torch.ones((1, 1, 3, 3), device=device)
    boundary = (F.conv2d(mask, kernel, padding=1) > 0).float() - mask
    avg_color = (image_tensor * boundary).sum(dim=(2,3), keepdim=True) / (boundary.sum(dim=(2,3), keepdim=True) + 1e-6)

    known_latents = pipe.vae.encode(image_tensor * mask).latent_dist.sample(generator) * pipe.vae.config.scaling_factor
    mask_latent = F.interpolate(mask, size=(64, 64), mode="nearest")

    noise = permute_noise_by_edges(torch.randn(known_latents.shape, device=device), edge_map)
    avg_latent = pipe.vae.encode(avg_color.repeat(1,1,512,512)).latent_dist.sample(generator) * pipe.vae.config.scaling_factor
    latents = (1 - color_factor) * noise + color_factor * avg_latent
    latents = pipe.scheduler.add_noise(known_latents, latents, pipe.scheduler.timesteps[0])

    prompt_embeds, neg_embeds = pipe.encode_prompt(prompt, device, 1, True, negative_prompt="blurry, smudge, bad quality")
    text_embeddings = torch.cat([neg_embeds, prompt_embeds])
    pipe.scheduler.set_timesteps(steps)

    for i, t in enumerate(pipe.scheduler.timesteps):
        noise_step = torch.randn(latents.shape, generator=generator, device=device)
        noisy_known = pipe.scheduler.add_noise(known_latents, noise_step, t)
        latents = (mask_latent * noisy_known) + ((1 - mask_latent) * latents)

        with torch.enable_grad():
            latents.requires_grad_(True)
            l_gx, l_gy = latents[:,:,:,1:] - latents[:,:,:,:-1], latents[:,:,1:,:] - latents[:,:,:-1,:]
            t_gx, t_gy = noisy_known[:,:,:,1:] - noisy_known[:,:,:,:-1], noisy_known[:,:,1:,:] - noisy_known[:,:,:-1,:]
            loss = F.mse_loss(l_gx * mask_latent[:,:,:,1:], t_gx * mask_latent[:,:,:,1:]) + \
                   F.mse_loss(l_gy * mask_latent[:,:,1:,:], t_gy * mask_latent[:,:,1:,:])
            grad = torch.autograd.grad(loss, latents)[0]
            latents = (latents - lambda_grad * grad).detach()

        latent_input = torch.cat([latents] * 2)
        noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    latents = (mask_latent * known_latents) + ((1 - mask_latent) * latents)
    result_img = pipe.image_processor.postprocess(pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample)[0]

    mask_blurred = pipe.image_processor.postprocess(mask)[0].convert("L").filter(ImageFilter.GaussianBlur(radius=11))
    return Image.composite(image.convert("RGB"), result_img.convert("RGB"), mask_blurred)

def main():
    parser = argparse.ArgumentParser("Structural + LGG DDPM Inpainting")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="output_final")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_sd_pipeline(device)
    image, mask = preprocess_inputs(args.image, args.mask)

    result = ddpm_structural_lgg_inpaint(pipe, image, mask, args.prompt)
    result.save(os.path.join(args.output_dir, "final_hybrid_result.png"))

if __name__ == "__main__":
    main()
