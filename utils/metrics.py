import json
import os

import lpips as lpips_lib
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from tqdm.auto import tqdm

from utils.image import apply_mask_for_display  # noqa: F401 (re-exported for convenience)

_lpips_model = None


def get_lpips_model(device="cpu"):
    """Return a cached LPIPS AlexNet model (initialized once per process)."""
    global _lpips_model
    if _lpips_model is None:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
            _lpips_model = lpips_lib.LPIPS(net="alex", version="0.1").to(device)
        _lpips_model.eval()
    return _lpips_model.to(device)


def compute_masked_metrics(original, inpainted, mask_tensor, device="cpu"):
    """
    Compute SSIM, PSNR, LPIPS on the inpainted region only.

    Parameters
    ----------
    original, inpainted : PIL Images (RGB, same size)
    mask_tensor         : (1,1,H,W) float32 tensor — 1=keep, 0=inpaint
    device              : torch device string for LPIPS

    Returns
    -------
    dict with keys "ssim", "psnr", "lpips"
    """
    orig_np = np.array(original).astype(np.float64)
    inp_np  = np.array(inpainted).astype(np.float64)
    mask_np = mask_tensor.squeeze().numpy()
    inpaint_mask = (mask_np == 0)

    if not inpaint_mask.any():
        return {"ssim": 1.0, "psnr": float("inf"), "lpips": 0.0}

    # ---- Bounding box of inpainted region ----
    rows = np.any(inpaint_mask, axis=1)
    cols = np.any(inpaint_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    crop_orig = orig_np[rmin:rmax + 1, cmin:cmax + 1]
    crop_inp  = inp_np[rmin:rmax + 1, cmin:cmax + 1]

    # ---- PSNR on masked pixels only ----
    masked_orig = orig_np[inpaint_mask]
    masked_inp  = inp_np[inpaint_mask]
    mse = np.mean((masked_orig - masked_inp) ** 2)
    psnr_val = 10.0 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")

    # ---- SSIM on bounding-box crop ----
    min_dim = min(crop_orig.shape[0], crop_orig.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    win_size = max(win_size, 3)

    ssim_val = ssim_fn(
        crop_orig, crop_inp,
        data_range=255.0, channel_axis=2, win_size=win_size,
    )

    # ---- LPIPS on bounding-box crop ----
    crop_orig_f = crop_orig.astype(np.float32)
    crop_inp_f  = crop_inp.astype(np.float32)

    orig_t = torch.from_numpy(crop_orig_f).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    inp_t  = torch.from_numpy(crop_inp_f).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

    lpips_model = get_lpips_model(device)
    with torch.no_grad():
        lpips_val = lpips_model(orig_t.to(device), inp_t.to(device)).item()

    return {"ssim": ssim_val, "psnr": psnr_val, "lpips": lpips_val}


def run_metrics(inpainted_dir, masks_dir, originals_dir, n_images, captions=None, device="cpu"):
    """
    Load saved results from disk and compute per-image metrics.

    Parameters
    ----------
    inpainted_dir : str — directory with 0000.png … inpainted images
    masks_dir     : str — directory with 0000.pt  … mask tensors
    originals_dir : str — directory with 0000.png + 0000.json
    n_images      : int — number of images to process
    captions      : dict or None — optional {img_id: [cap, …]} (not used if JSON present)
    device        : str — torch device for LPIPS

    Returns
    -------
    list of dicts: {idx, original, inpainted, mask, caption, ssim, psnr, lpips}
    """
    results = []
    desc = f"Metrics [{os.path.basename(inpainted_dir)}]"
    for i in tqdm(range(n_images), desc=desc):
        orig_path    = os.path.join(originals_dir, f"{i:04d}.png")
        inp_path     = os.path.join(inpainted_dir, f"{i:04d}.png")
        mask_path    = os.path.join(masks_dir,     f"{i:04d}.pt")
        caption_path = os.path.join(originals_dir, f"{i:04d}.json")

        if not all(os.path.exists(p) for p in [orig_path, inp_path, mask_path]):
            print(f"Stopping at image {i} (files not found). Run the inpainting loop first.")
            break

        orig_img = Image.open(orig_path)
        inp_img  = Image.open(inp_path)
        mask_t   = torch.load(mask_path, weights_only=True)

        caption = ""
        if os.path.exists(caption_path):
            with open(caption_path) as f:
                caption = json.load(f).get("caption", "")

        metrics = compute_masked_metrics(orig_img, inp_img, mask_t, device=device)
        results.append({
            "idx": i,
            "original": orig_img,
            "inpainted": inp_img,
            "mask": mask_t,
            "caption": caption,
            **metrics,
        })

    return results
