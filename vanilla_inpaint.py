import argparse
import os
import numpy as np
import torch
from PIL import Image

from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------
def load_vanilla_model(device: str) -> StableDiffusionPipeline:
    model_id = "sd2-community/stable-diffusion-2-base"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        use_auth_token=True,
    )

    # DDIM used only as a convenient DDPM-style scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=False)
    return pipe


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
def preprocess_inputs(image_path, mask_path, size=(512, 512)):
    image = Image.open(image_path).convert("RGB").resize(size, Image.LANCZOS)
    if mask_path.endswith(".pt"):
        mask_tensor = torch.load(mask_path, map_location="cpu").float()
        # Normalize to [0,1] if needed, then threshold
        if mask_tensor.max() > 1.0:
            mask_tensor = mask_tensor / 255.0
        # Ensure shape is (1, 1, H, W)
        while mask_tensor.dim() < 4:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask = torch.nn.functional.interpolate(mask_tensor, size=size[::-1], mode="nearest")
        mask = (mask > 0.5).float()
    else:
        mask = Image.open(mask_path).convert("L").resize(size, Image.NEAREST)
        # mask: 1 = keep, 0 = inpaint
        mask = torch.from_numpy(
            (np.array(mask) > 127).astype(np.float32)
        ).unsqueeze(0).unsqueeze(0)

    return image, mask


# ---------------------------------------------------------
# Vanilla DDPM inpainting sampler
# ---------------------------------------------------------
@torch.no_grad()
def ddpm_inpaint(
        pipe,
        image: Image.Image,
        mask: torch.Tensor,
        prompt: str,
        steps: int,
        guidance_scale: float,
        seed: int,
):
    device = pipe.device
    generator = torch.Generator(device).manual_seed(seed)

    #Prepare
    image_tensor = pipe.image_processor.preprocess(image).to(device)
    mask = mask.to(device)
    image_tensor = image_tensor * mask

    known_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator)
    known_latents *= pipe.vae.config.scaling_factor

    # Downsample mask to latent resolution
    mask = torch.nn.functional.interpolate(mask, size=known_latents.shape[2:], mode="nearest")

    # Initial pure noise for the very first step
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

    for t in pipe.scheduler.timesteps:

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

        # Clamp again at noise level t-1
        noise = torch.randn(known_latents.shape, generator=generator, device=device, dtype=known_latents.dtype)
        noisy_known = pipe.scheduler.add_noise(known_latents, noise, t-1)
        latents = (mask * noisy_known) + ((1 - mask) * latents)

    latents = (mask * known_latents) + ((1 - mask) * latents)

    # Decode the latents
    latents /= pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents).sample
    image = pipe.image_processor.postprocess(image)[0]

    return image


# ---------------------------------------------------------
# CLI SCRIPT
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Vanilla DDPM Inpainting")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="output_ddpm")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading vanilla diffusion model...")
    pipe = load_vanilla_model(device)

    print("Preprocessing inputs...")
    image, mask = preprocess_inputs(args.image, args.mask)

    print("Running DDPM inpainting...")
    result = ddpm_inpaint(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=args.prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    out_name = f"inpaint_seed{args.seed}.png"
    out_path = os.path.join(args.output_dir, out_name)
    result.save(out_path)

    print(f"Saved result to: {out_path}")


if __name__ == "__main__":
    main()