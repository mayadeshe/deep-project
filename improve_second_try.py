import argparse
import os
import torch
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------
def load_layered_model(device: str):
    model_id = "sd2-community/stable-diffusion-2-base"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=False)
    return pipe


@torch.no_grad()
def layered_ddpm_inpaint(
        pipe,
        image: Image.Image,
        mask: torch.Tensor,
        prompt: str,
        num_layers: int = 10,
        steps_per_layer=None,
        guidance_scale_per_layer=None,
        seed: int = 42,
):
    device = pipe.device
    generator = torch.Generator(device).manual_seed(seed)

    # Preprocess image
    image_tensor = pipe.image_processor.preprocess(image).to(device)
    mask = mask.to(device)

    # Encode image to latent
    known_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator)
    known_latents *= pipe.vae.config.scaling_factor

    # Compute distance transform for hole
    hole = (mask == 0).cpu().numpy()[0, 0]
    distance = ndi.distance_transform_edt(hole)

    # Create layers (exclude known pixels in first layer)
    max_dist = distance.max()
    layer_bins = np.linspace(0, max_dist, num_layers + 1)
    layer_masks = []
    for i in range(num_layers):
        low = layer_bins[i] if i > 0 else 1e-6
        layer = (distance >= low) & (distance < layer_bins[i + 1])
        layer_mask = torch.from_numpy(layer.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        layer_mask = torch.nn.functional.interpolate(
            layer_mask, size=known_latents.shape[-2:], mode="nearest"
        ).to(device)
        layer_masks.append(layer_mask)

    # Initialize latent with noise
    latents = torch.randn(
        known_latents.shape,
        generator=generator,
        device=device,
        dtype=known_latents.dtype,
    )

    # Encode prompt for CFG
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Ensure steps_per_layer and guidance_scale_per_layer are lists
    if isinstance(steps_per_layer, int):
        steps_per_layer = [steps_per_layer] * num_layers
    if isinstance(guidance_scale_per_layer, float):
        guidance_scale_per_layer = [guidance_scale_per_layer] * num_layers

    # Downsample mask to latent resolution
    completed_mask = torch.nn.functional.interpolate(
        mask.clone(), size=known_latents.shape[-2:], mode="nearest"
    )

    # Layered DDPM sampling
    for idx, layer_mask in enumerate(layer_masks):
        pipe.scheduler.set_timesteps(steps_per_layer[idx])
        timesteps = pipe.scheduler.timesteps

        for t in timesteps:
            noise = torch.randn_like(known_latents)
            alpha_bar = pipe.scheduler.alphas_cumprod[t]
            sqrt_alpha_bar = alpha_bar.sqrt()
            sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()
            noisy_known = sqrt_alpha_bar * known_latents + sqrt_one_minus_alpha_bar * noise

            effective_mask = torch.clamp(completed_mask, 0.0, 1.0)
            latents = effective_mask * noisy_known + (1 - effective_mask) * latents

            latent_input = torch.cat([latents] * 2)
            noise_pred = pipe.unet(
                latent_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale_per_layer[idx] * (noise_text - noise_uncond)

            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Update completed mask
        completed_mask = torch.clamp(completed_mask + layer_mask, 0.0, 1.0)

    latents /= pipe.vae.config.scaling_factor
    out_image = pipe.vae.decode(latents).sample
    out_image = pipe.image_processor.postprocess(out_image)[0]
    return out_image


def preprocess_inputs(image_path, mask_path, size=(512, 512)):
    image = Image.open(image_path).convert("RGB").resize(size, Image.LANCZOS)
    mask = Image.open(mask_path).convert("L").resize(size, Image.NEAREST)
    mask = torch.from_numpy((np.array(mask) > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return image, mask


def main():
    parser = argparse.ArgumentParser("Layered DDPM Inpainting")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="output_layered_ddpm")
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--steps_per_layer", type=str, default="15",
                        help="Comma-separated list of steps per layer")
    parser.add_argument("--guidance_scale_per_layer", type=str, default="7.5",
                        help="Comma-separated list of CFG scales per layer")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading diffusion model...")
    pipe = load_layered_model(device)

    print("Preprocessing inputs...")
    image, mask = preprocess_inputs(args.image, args.mask)

    # Parse per-layer lists
    steps_per_layer = [int(x) for x in args.steps_per_layer.split(",")]
    guidance_per_layer = [float(x) for x in args.guidance_scale_per_layer.split(",")]

    if len(steps_per_layer) != args.num_layers:
        steps_per_layer = (steps_per_layer * args.num_layers)[:args.num_layers]
    if len(guidance_per_layer) != args.num_layers:
        guidance_per_layer = (guidance_per_layer * args.num_layers)[:args.num_layers]

    print("Running layered DDPM inpainting...")
    result = layered_ddpm_inpaint(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=args.prompt,
        num_layers=args.num_layers,
        steps_per_layer=steps_per_layer,
        guidance_scale_per_layer=guidance_per_layer,
        seed=args.seed
    )

    out_path = os.path.join(args.output_dir, f"layered_inpaint_seed{args.seed}.png")
    result.save(out_path)
    print(f"Saved result to: {out_path}")


if __name__ == "__main__":
    main()