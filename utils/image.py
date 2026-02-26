import numpy as np
import torch
from PIL import Image


def prepare_mnist_mask(mnist_image, size=(512, 512)):
    """
    Convert an MNIST PIL image to an inpainting mask.
    Digit pixels (bright > 127) become the inpainted region.

    Convention: 1 = keep, 0 = inpaint.

    Returns
    -------
    torch.Tensor of shape (1, 1, H, W), dtype float32
    """
    mask_pil = mnist_image.resize(size, Image.NEAREST)
    mask_np = np.array(mask_pil)
    mask_binary = (mask_np <= 127).astype(np.float32)
    return torch.from_numpy(mask_binary).unsqueeze(0).unsqueeze(0)


def prepare_coco_image(image_path, size=(512, 512)):
    """Load a COCO image from disk and resize to *size*. Returns PIL RGB image."""
    return Image.open(image_path).convert("RGB").resize(size, Image.LANCZOS)


def apply_mask_for_display(image, mask_tensor):
    """Black out the inpainted region for visualization. Returns PIL image."""
    img_np = np.array(image).copy()
    mask_np = mask_tensor.squeeze().numpy()  # (H, W), 1=keep, 0=inpaint
    img_np[mask_np == 0] = 0
    return Image.fromarray(img_np)
