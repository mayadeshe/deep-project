# Diffusion-Based Inpainting
#
# Setup (conda):
#   conda env create -f environment.yaml
#   conda activate inpainting-env
#
# Usage:
#   python inpaint.py --image photo.jpg --mask mask.png --prompt "a wooden bench" --output result.png
#
# Requirements (if installing manually with pip):
#   pip install diffusers==0.25.0 transformers==4.36.0 accelerate==0.25.0 torch==2.1.0 torchvision==0.16.0 Pillow==10.1.0

import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


def load_model(device: str) -> StableDiffusionInpaintPipeline:
    """Load StableDiffusion 2 base inpainting pipeline onto the given device."""
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-base",
        torch_dtype=torch.float32,  # float16 is unreliable on MPS
        safety_checker=None,        # disabled to reduce RAM usage
    )
    pipe = pipe.to(device)
    return pipe


def preprocess_inputs(image_path: str, mask_path: str, target_size=(512, 512)):
    """
    Load and resize image and mask to target_size.

    - Image is resized with LANCZOS (high-quality downsampling).
    - Mask is resized with NEAREST (preserves hard binary edges, avoids gray fringe).
    - Mask is converted to RGB as required by StableDiffusionInpaintPipeline.

    Returns:
        (image_rgb, mask_rgb): tuple of PIL Images
    """
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    image = image.resize(target_size, Image.LANCZOS)
    mask = mask.resize(target_size, Image.NEAREST)

    mask_rgb = mask.convert("RGB")

    return image, mask_rgb


def run_inpainting(
    pipe: StableDiffusionInpaintPipeline,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
) -> Image.Image:
    """
    Run inpainting inference.

    The generator is created on the same device as the pipeline to avoid
    device mismatch errors on MPS.
    """
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return result.images[0]


def main():
    parser = argparse.ArgumentParser(
        description="Text-conditioned inpainting with Stable Diffusion 2 base (MPS/CPU)"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--mask", required=True, help="Path to mask image (white = inpaint region)")
    parser.add_argument("--prompt", required=True, help="Text prompt describing the inpainted region")
    parser.add_argument("--output", default="output.png", help="Output file path (default: output.png)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps (default: 50)")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (default: 7.5)",
    )
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    pipe = load_model(device)

    print("Preprocessing inputs...")
    image, mask = preprocess_inputs(args.image, args.mask)

    print(f"Running inpainting (steps={args.steps}, guidance_scale={args.guidance_scale}, seed={args.seed})...")
    result = run_inpainting(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=args.prompt,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
    )

    result.save(args.output)
    print(f"Saved result to: {args.output}")


if __name__ == "__main__":
    main()
