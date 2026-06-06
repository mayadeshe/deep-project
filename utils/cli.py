import argparse
import os

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDPMScheduler


MODEL_ID = "sd2-community/stable-diffusion-2-base"

_BASE_ARGS = [
    ("--image",          {"required": True}),
    ("--mask",           {"required": True}),
    ("--prompt",         {"required": True}),
    ("--steps",          {"type": int,   "default": 50}),
    ("--guidance_scale", {"type": float, "default": 7.5}),
    ("--seed",           {"type": int,   "default": 42}),
]


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def preprocess_inputs(image_path, mask_path, size=(512, 512)):
    image = Image.open(image_path).convert("RGB").resize(size, Image.LANCZOS)
    if mask_path.endswith(".pt"):
        mask_tensor = torch.load(mask_path, map_location="cpu").float()
        if mask_tensor.max() > 1.0:
            mask_tensor = mask_tensor / 255.0
        while mask_tensor.dim() < 4:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask = torch.nn.functional.interpolate(mask_tensor, size=size[::-1], mode="nearest")
        mask = (mask > 0.5).float()
    else:
        mask_pil = Image.open(mask_path).convert("L").resize(size, Image.NEAREST)
        mask = torch.from_numpy(
            (np.array(mask_pil) > 127).astype(np.float32)
        ).unsqueeze(0).unsqueeze(0)
    return image, mask


def load_sd_pipeline(device):
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=False)
    return pipe


def run_inpaint_cli(name, sampler, default_output_dir, extra_args=None):
    parser = argparse.ArgumentParser(name)
    for flag, kwargs in _BASE_ARGS:
        parser.add_argument(flag, **kwargs)
    parser.add_argument("--output_dir", default=default_output_dir)
    extra_keys = []
    for flag, kwargs in (extra_args or []):
        parser.add_argument(flag, **kwargs)
        extra_keys.append(flag.lstrip("-").replace("-", "_"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    print("Loading diffusion model...")
    pipe = load_sd_pipeline(device)

    print("Preprocessing inputs...")
    image, mask = preprocess_inputs(args.image, args.mask)

    sampler_kwargs = {k: getattr(args, k) for k in extra_keys}
    print(f"Running {name}...")
    result = sampler(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=args.prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        **sampler_kwargs,
    )

    out_path = os.path.join(args.output_dir, f"inpaint_seed{args.seed}.png")
    result.save(out_path)
    print(f"Saved result to: {out_path}")
    return out_path
