import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from utils.cli import preprocess_inputs, load_sd_pipeline


def get_local_boundary_color_map(image_tensor, mask):
    """
    Build a local boundary color map using EDT (Euclidean Distance Transform).
    For each pixel in the inpaint hole, assigns the color of the nearest
    boundary pixel.

    Args:
        image_tensor: [1, 3, H, W] tensor, known region (hole zeroed out)
        mask: [1, 1, H, W] tensor, 1=keep, 0=inpaint

    Returns:
        color_map: [1, 3, H, W] tensor — local color map for the hole region
    """
    from scipy.ndimage import distance_transform_edt, binary_dilation

    mask_np = mask[0, 0].cpu().numpy()

    kernel = np.ones((3, 3))
    dilated_hole = binary_dilation(mask_np == 0, structure=kernel)
    boundary_np = (mask_np == 1) & dilated_hole

    _, indices = distance_transform_edt(~boundary_np, return_distances=True, return_indices=True)

    img_np = image_tensor[0].cpu().numpy()
    color_map_np = np.zeros_like(img_np)
    for c in range(3):
        color_map_np[c] = img_np[c][indices[0], indices[1]]

    color_map = torch.from_numpy(color_map_np).unsqueeze(0).to(image_tensor.device)
    return color_map


@torch.no_grad()
def ddpm_local_color_inpaint(
        pipe, image, mask, prompt, steps=50, guidance_scale=7.5, seed=42,
        color_factor=0.2,
):
    """Vanilla DDPM inpainting + local boundary color map only."""
    device = pipe.device
    generator = torch.Generator(device).manual_seed(seed)

    # Prepare masked image
    image_tensor = pipe.image_processor.preprocess(image).to(device)
    mask = mask.to(device)[:, :1, :, :]
    image_tensor = image_tensor * mask

    # Local boundary color map (the only improvement)
    color_map = get_local_boundary_color_map(image_tensor, mask)

    # Encode known latents
    known_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator) * pipe.vae.config.scaling_factor
    mask_latent = F.interpolate(mask, size=(64, 64), mode="nearest")

    # Noise initialization with color prior blended in
    noise = torch.randn(known_latents.shape, device=device, generator=generator)
    color_latent = pipe.vae.encode(color_map).latent_dist.sample(generator) * pipe.vae.config.scaling_factor
    latents = (1 - color_factor) * noise + color_factor * color_latent

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
    parser = argparse.ArgumentParser("Local Color Map DDPM Inpainting (Ablation)")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="output_local_color_only")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--color_factor", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    pipe = load_sd_pipeline(device)
    image, mask = preprocess_inputs(args.image, args.mask)

    result = ddpm_local_color_inpaint(
        pipe, image, mask, args.prompt,
        steps=args.steps, guidance_scale=args.guidance_scale, seed=args.seed,
        color_factor=args.color_factor,
    )
    out_path = os.path.join(args.output_dir, f"inpaint_seed{args.seed}.png")
    result.save(out_path)
    print(f"Saved result to: {out_path}")


if __name__ == "__main__":
    main()
