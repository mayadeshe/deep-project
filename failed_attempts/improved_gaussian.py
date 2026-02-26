import argparse
import os
import torch
from PIL import Image
from torchvision.transforms.functional import gaussian_blur

from cliutils import preprocess_inputs, load_sd_pipeline


# ---------------------------------------------------------
# DDPM inpainting + Gaussian soft mask blending
# (No RePaint, no resampling)
# ---------------------------------------------------------

@torch.no_grad()
def ddpm_inpaint_softmask(
        pipe,
        image: Image.Image,
        mask: torch.Tensor,
        prompt: str,
        steps: int,
        guidance_scale: float,
        seed: int,
        blur_sigma: float,
        blur_kernel: int,
):
    device = pipe.device
    generator = torch.Generator(device).manual_seed(seed)

    # ---------------------------------
    # Prepare masked image
    # ---------------------------------
    image_tensor = pipe.image_processor.preprocess(image).to(device)
    mask = mask.to(device)
    image_tensor = image_tensor * mask

    # ---------------------------------
    # Encode known region
    # ---------------------------------
    known_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator)
    known_latents *= pipe.vae.config.scaling_factor

    # ---------------------------------
    # 🔥 Gaussian soft mask (NEW PART)
    # ---------------------------------
    if blur_kernel > 1 and blur_sigma > 0.0:
        mask = gaussian_blur(mask, kernel_size=blur_kernel, sigma=blur_sigma)

    # Downsample mask smoothly to latent resolution
    mask = torch.nn.functional.interpolate(
        mask,
        size=known_latents.shape[2:],
        mode="bilinear",
        align_corners=False,
    )

    # ---------------------------------
    # Initial noise
    # ---------------------------------
    latents = torch.randn(
        known_latents.shape,
        generator=generator,
        device=device,
        dtype=known_latents.dtype
    )

    # ---------------------------------
    # Text embeddings
    # ---------------------------------
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])

    pipe.scheduler.set_timesteps(steps)

    # ---------------------------------
    # DDPM loop (unchanged)
    # ---------------------------------
    for t in pipe.scheduler.timesteps:

        # Clamp known region at noise level t
        noise = torch.randn(
            known_latents.shape,
            generator=generator,
            device=device,
            dtype=known_latents.dtype,
        )
        noisy_known = pipe.scheduler.add_noise(known_latents, noise, t)

        # Soft blending happens here
        latents = (mask * noisy_known) + ((1 - mask) * latents)

        # Predict noise
        latent_input = torch.cat([latents] * 2)
        noise_pred = pipe.unet(
            latent_input,
            t,
            encoder_hidden_states=text_embeddings
        ).sample

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # Reverse step
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Final clamp
    latents = (mask * known_latents) + ((1 - mask) * latents)

    # ---------------------------------
    # Decode
    # ---------------------------------
    latents /= pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents).sample
    image = pipe.image_processor.postprocess(image)[0]

    return image


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def odd_kernel(value):
    k = int(value)
    if k % 2 == 0:
        raise argparse.ArgumentTypeError(f"--blur-kernel must be odd, got {k}")
    return k


def main():
    parser = argparse.ArgumentParser("DDPM Inpainting + Soft Gaussian Mask")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="output_softmask")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)

    # Only new parameters
    parser.add_argument("--blur-sigma", type=float, default=1.5)
    parser.add_argument("--blur-kernel", type=odd_kernel, default=7)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading diffusion model...")
    pipe = load_sd_pipeline(device)

    print("Preprocessing inputs...")
    image, mask = preprocess_inputs(args.image, args.mask)

    print("Running DDPM + soft mask...")
    result = ddpm_inpaint_softmask(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=args.prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        blur_sigma=args.blur_sigma,
        blur_kernel=args.blur_kernel,
    )

    out_path = os.path.join(args.output_dir, f"inpaint_seed{args.seed}.png")
    result.save(out_path)
    print(f"Saved result to: {out_path}")


if __name__ == "__main__":
    main()