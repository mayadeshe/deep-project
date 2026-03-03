import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDPMScheduler


MODEL_ID = "sd2-community/stable-diffusion-2-base"

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
