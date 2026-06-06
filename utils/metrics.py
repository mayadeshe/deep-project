import json
import os
import warnings
import lpips as lpips_lib
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, sobel
from skimage.metrics import structural_similarity as ssim_fn
from tqdm.auto import tqdm

# Each entry: (key, display_name, xlabel, precision, higher_is_better)
METRIC_SPECS = [
    ("ssim",                   "SSIM",           "SSIM",                                  ".4f", True),
    ("psnr",                   "PSNR",           "PSNR (dB)",                             ".2f", True),
    ("lpips",                  "LPIPS",          "LPIPS (lower=better)",                  ".4f", False),
    ("seam_score",             "Seam Score",     "Boundary Discontinuity (lower=better)", ".4f", False),
    ("color_distance",         "Color Distance", "Color Distance (lower=better)",         ".4f", False),
    ("gradient_variance_diff", "Grad Var Diff",  "Gradient Variance Diff (lower=better)", ".4f", False),
]

_lpips_model = None

def get_lpips_model(device="cpu"):
    global _lpips_model
    if _lpips_model is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
            _lpips_model = lpips_lib.LPIPS(net="alex", version="0.1").to(device)
        _lpips_model.eval()
    return _lpips_model.to(device)


def compute_masked_metrics(original, inpainted, mask_tensor, device="cpu"):
    orig_np = np.array(original).astype(np.float64)
    inp_np  = np.array(inpainted).astype(np.float64)
    mask_np = mask_tensor.squeeze().numpy()
    inpaint_mask = (mask_np == 0)
    inpaint_mask_bool = inpaint_mask.astype(bool)

    if not inpaint_mask.any():
        return {
            "ssim": 1.0, "psnr": float("inf"), "lpips": 0.0,
            "seam_score": 0.0, "color_distance": 0.0, "gradient_variance_diff": 0.0,
        }

    # Tight bounding box around the masked region (used by SSIM only)
    rows = np.any(inpaint_mask, axis=1)
    cols = np.any(inpaint_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    crop_orig = orig_np[rmin:rmax + 1, cmin:cmax + 1]
    crop_inp  = inp_np[rmin:rmax + 1, cmin:cmax + 1]

    # --- PSNR: masked pixels only, vs ground truth ---
    masked_orig = orig_np[inpaint_mask_bool]
    masked_inp  = inp_np[inpaint_mask_bool]
    mse = np.mean((masked_orig - masked_inp) ** 2)
    psnr_val = 10.0 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")

    # --- SSIM: tight bbox crop, vs ground truth ---
    min_dim = min(crop_orig.shape[0], crop_orig.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    win_size = max(win_size, 3)
    ssim_val = ssim_fn(
        crop_orig, crop_inp,
        data_range=255.0, channel_axis=2, win_size=win_size,
    )

    # --- LPIPS: whole 512x512 image, vs ground truth ---
    # Whole-image scope because the AlexNet/VGG receptive field demands more
    # spatial context than a tight mask bbox can reliably provide.
    orig_t = torch.from_numpy(orig_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    inp_t  = torch.from_numpy(inp_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    lpips_model = get_lpips_model(device)
    with torch.no_grad():
        lpips_val = lpips_model(orig_t.to(device), inp_t.to(device)).item()

    # --- Color Distance: per-pixel L2 RGB distance, vs ground truth, masked pixels only ---
    # Matches the report formula  E_{p in H}[|| I_gen(p) - I_gt(p) ||_2].
    diff = orig_np[inpaint_mask_bool] - inp_np[inpaint_mask_bool]   # shape (N, 3)
    color_distance = float(np.mean(np.linalg.norm(diff, axis=1)))

    # --- Grad Var Diff: |Var(grad I_gen) - Var(grad I_gt)| over masked pixels ---
    # Sobel gradient magnitude on grayscale, evaluated only on masked pixels.
    gray_inp  = inp_np.mean(axis=2)
    gray_orig = orig_np.mean(axis=2)
    gmag_inp  = np.sqrt(sobel(gray_inp,  axis=1) ** 2 + sobel(gray_inp,  axis=0) ** 2)
    gmag_orig = np.sqrt(sobel(gray_orig, axis=1) ** 2 + sobel(gray_orig, axis=0) ** 2)
    gradient_variance_diff = float(abs(
        gmag_inp[inpaint_mask_bool].var() - gmag_orig[inpaint_mask_bool].var()
    ))

    # --- Seam Score: mean L1 grad magnitude on 2-pixel ring outside the mask ---
    dilated = binary_dilation(inpaint_mask_bool, iterations=2)
    boundary_mask = dilated & ~inpaint_mask_bool
    gx = np.abs(np.diff(inp_np.mean(axis=2), axis=1))
    gy = np.abs(np.diff(inp_np.mean(axis=2), axis=0))
    grad_mag = np.zeros_like(inp_np[:, :, 0])
    grad_mag[:, :-1] += gx
    grad_mag[:-1, :] += gy
    seam_score = float(grad_mag[boundary_mask].mean()) if boundary_mask.any() else 0.0

    return {
        "ssim": ssim_val,
        "psnr": psnr_val,
        "lpips": lpips_val,
        "seam_score": seam_score,
        "color_distance": color_distance,
        "gradient_variance_diff": gradient_variance_diff,
    }


def run_metrics(inpainted_dir, masks_dir, originals_dir, n_images, captions=None, device="cpu"):
    results = []
    for i in tqdm(range(n_images), desc=f"Metrics [{os.path.basename(inpainted_dir)}]"):
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
