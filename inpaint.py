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
    parser = argparse.ArgumentParser(description="Inpainting with SD2 Base")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--prompt", required=True)
    # נשאיר את output כתיקייה כברירת מחדל
    parser.add_argument("--output_dir", default="output_vanila")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    args = parser.parse_args()

    # יצירת שם קובץ מתוך הפרומפט
    # ננקה תווים שעלולים להיות בעייתיים בשמות קבצים (כמו רווחים או סימני פיסוק)
    clean_prompt = "".join([c if c.isalnum() else "_" for c in args.prompt])
    # נגביל את האורך כדי שלא יהיה שם קובץ ארוך מדי
    filename = f"{clean_prompt[:50]}_seed{args.seed}.png"
    full_output_path = os.path.join(args.output_dir, filename)

    # יצירת תיקיית פלט אם היא לא קיימת
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    pipe = load_model(device)

    print("Preprocessing inputs...")
    image, mask = preprocess_inputs(args.image, args.mask)

    print(f"Running inpainting for prompt: '{args.prompt}'")
    result = run_inpainting(pipe, image, mask, args.prompt, args.seed, args.steps, args.guidance_scale)

    # שמירה לנתיב החדש שיצרנו מהפרומפט
    result.save(full_output_path)
    print(f"Saved result to: {full_output_path}")

if __name__ == "__main__":
    main()