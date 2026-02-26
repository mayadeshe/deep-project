import numpy as np
import torch
from PIL import Image


def prepare_mnist_mask(mnist_image, size=(512, 512)):
    mask_pil = mnist_image.resize(size, Image.NEAREST)
    mask_np = np.array(mask_pil)
    mask_binary = (mask_np <= 127).astype(np.float32)
    return torch.from_numpy(mask_binary).unsqueeze(0).unsqueeze(0)


def prepare_coco_image(image_path, size=(512, 512)):
    return Image.open(image_path).convert("RGB").resize(size, Image.LANCZOS)


def apply_mask_for_display(image, mask_tensor):
    img_np = np.array(image).copy()
    mask_np = mask_tensor.squeeze().numpy()
    img_np[mask_np == 0] = 0
    return Image.fromarray(img_np)
