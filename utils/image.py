import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt


def prepare_mnist_mask(mnist_image, size=(512, 512), thickness=1.5):
    """
    Convert an MNIST PIL image to an inpainting mask using EDT-based upscaling.
    Digit pixels (bright > 127) become the inpainted region.

    Upsamples the continuous distance field via bicubic interpolation then
    thresholds, producing smooth anti-aliased digit silhouettes instead of
    blocky nearest-neighbour pixels.

    Convention: 1 = keep, 0 = inpaint.

    Parameters
    ----------
    thickness : float
        Distance threshold in 28×28 pixel units. Default 1.5 expands each
        digit stroke by ~27px at 512×512 output.

    Returns
    -------
    torch.Tensor of shape (1, 1, H, W), dtype float32
    """
    # Binarize: digit pixels (bright >127) = 1, background = 0
    mask_np = (np.array(mnist_image) > 127).astype(np.float32)

    # EDT on inverted mask: distance from background to nearest digit pixel
    dist_field = distance_transform_edt(1.0 - mask_np)

    # Upsample continuous distance field with bicubic interpolation
    dist_tensor = torch.from_numpy(dist_field).unsqueeze(0).unsqueeze(0).float()
    highres_dist = F.interpolate(dist_tensor, size=size, mode='bicubic', align_corners=False)

    # Threshold: pixels within `thickness` of a digit pixel become inpaint region (0)
    # Convention: 1=keep, 0=inpaint — digit region is 0
    inpaint_region = (highres_dist <= thickness).float()
    return 1.0 - inpaint_region


def prepare_coco_image(image_path, size=(512, 512)):
    """Load a COCO image from disk and resize to *size*. Returns PIL RGB image."""
    return Image.open(image_path).convert("RGB").resize(size, Image.LANCZOS)


def apply_mask_for_display(image, mask_tensor):
    """Black out the inpainted region for visualization. Returns PIL image."""
    img_np = np.array(image).copy()
    mask_np = mask_tensor.squeeze().numpy()  # (H, W), 1=keep, 0=inpaint
    img_np[mask_np == 0] = 0
    return Image.fromarray(img_np)
