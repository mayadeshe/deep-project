import argparse
import torch
import os
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

def load_model(device: str) -> StableDiffusionInpaintPipeline:
    """Load StableDiffusion 2 base into an inpainting pipeline."""
    model_id = "sd2-community/stable-diffusion-2-base"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_auth_token=True,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    return pipe

def preprocess_inputs(image_path: str, mask_path: str, target_size=(512, 512)):
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
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # כאן השמות חייבים להתאים למה ש-InpaintPipeline מצפה
    result = pipe(
        prompt=prompt,
        image=image,            # התמונה המקורית
        mask_image=mask,        # המסכה
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return result.images[0]

def main():
    parser = argparse.ArgumentParser(description="Inpainting with SD2 Base")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="output_vanila/vase_result.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    args = parser.parse_args()

    # יצירת תיקיית פלט אם היא לא קיימת
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    pipe = load_model(device)

    print("Preprocessing inputs...")
    image, mask = preprocess_inputs(args.image, args.mask)

    print(f"Running inpainting (steps={args.steps})...")
    result = run_inpainting(pipe, image, mask, args.prompt, args.seed, args.steps, args.guidance_scale)

    result.save(args.output)
    print(f"Saved result to: {args.output}")

if __name__ == "__main__":
    main()