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

    if not inpaint_mask.any():
        return {"ssim": 1.0, "psnr": float("inf"), "lpips": 0.0}

    rows = np.any(inpaint_mask, axis=1)
    cols = np.any(inpaint_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    crop_orig = orig_np[rmin:rmax + 1, cmin:cmax + 1]
    crop_inp  = inp_np[rmin:rmax + 1, cmin:cmax + 1]

    masked_orig = orig_np[inpaint_mask]
    masked_inp  = inp_np[inpaint_mask]
    mse = np.mean((masked_orig - masked_inp) ** 2)
    psnr_val = 10.0 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")

    min_dim = min(crop_orig.shape[0], crop_orig.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    win_size = max(win_size, 3)

    ssim_val = ssim_fn(
        crop_orig, crop_inp,
        data_range=255.0, channel_axis=2, win_size=win_size,
    )

    crop_orig_f = crop_orig.astype(np.float32)
    crop_inp_f  = crop_inp.astype(np.float32)

    orig_t = torch.from_numpy(crop_orig_f).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    inp_t  = torch.from_numpy(crop_inp_f).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

    lpips_model = get_lpips_model(device)
    with torch.no_grad():
        lpips_val = lpips_model(orig_t.to(device), inp_t.to(device)).item()

    # --- Metric: Boundary Discontinuity (Seam Smoothness) ---
    # Measures pixel intensity jump along the mask boundary (lower = smoother seam)
    inpaint_mask_bool = inpaint_mask.astype(bool)
    dilated = binary_dilation(inpaint_mask_bool, iterations=2)
    boundary_outside = dilated & ~inpaint_mask_bool  # pixels just outside mask
    boundary_inside = inpaint_mask_bool & ~binary_dilation(~inpaint_mask_bool, iterations=2)  # pixels just inside mask
    # Simpler approach: dilate mask edge and measure gradient magnitude there
    edge_band = binary_dilation(inpaint_mask_bool, iterations=2) ^ binary_dilation(~inpaint_mask_bool, iterations=2) & inpaint_mask_bool
    # Use the boundary between inside and outside
    boundary_ring = dilated ^ inpaint_mask_bool  # 2px ring outside the mask
    boundary_ring_inside = inpaint_mask_bool ^ binary_dilation(~inpaint_mask_bool.astype(bool), iterations=2).astype(bool) & inpaint_mask_bool

    # Compute gradient magnitude on the inpainted image at the boundary
    grad_x = np.abs(np.diff(inp_np.mean(axis=2), axis=1))
    grad_y = np.abs(np.diff(inp_np.mean(axis=2), axis=0))
    # Pad to original size
    grad_mag = np.zeros_like(inp_np[:, :, 0])
    grad_mag[:, :-1] += grad_x
    grad_mag[:-1, :] += grad_y

    # Boundary = 2px dilation minus the mask itself (ring just outside)
    boundary_mask = dilated & ~inpaint_mask_bool
    if boundary_mask.any():
        seam_score = grad_mag[boundary_mask].mean()
    else:
        seam_score = 0.0

    # --- Metric: Lighting/Color Distance ---
    # Mean color difference between inside and outside the mask (lower = better match)
    if inpaint_mask_bool.any() and (~inpaint_mask_bool).any():
        mean_inside = inp_np[inpaint_mask_bool].mean(axis=0)   # shape (3,)
        mean_outside = inp_np[~inpaint_mask_bool].mean(axis=0)
        color_distance = np.linalg.norm(mean_inside - mean_outside)
    else:
        color_distance = 0.0

    # --- Metric: Gradient Variance (Texture Continuity) ---
    # Compare gradient variance inside mask vs immediate surroundings (lower diff = better)
    gray_inp = inp_np.mean(axis=2)
    sobel_x = sobel(gray_inp, axis=1)
    sobel_y = sobel(gray_inp, axis=0)
    grad_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Surrounding band: dilate mask by 5px, subtract original mask
    surround = binary_dilation(inpaint_mask_bool, iterations=5) & ~inpaint_mask_bool
    if inpaint_mask_bool.any() and surround.any():
        var_inside = grad_magnitude[inpaint_mask_bool].var()
        var_surround = grad_magnitude[surround].var()
        gradient_variance_diff = abs(var_inside - var_surround)
    else:
        gradient_variance_diff = 0.0

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
