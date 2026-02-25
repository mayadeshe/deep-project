import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.stats import gaussian_kde

from utils.image import apply_mask_for_display


def print_stats_table(results, label=""):
    """Print Mean / Median / Std for SSIM, PSNR, LPIPS."""
    ssim_v  = np.array([r["ssim"]  for r in results])
    psnr_v  = np.array([r["psnr"]  for r in results])
    lpips_v = np.array([r["lpips"] for r in results])

    print(f"\n{'=' * 55}")
    print(f"  {label} — {len(results)} images (inpainted region only)")
    print(f"{'=' * 55}")
    print(f"{'Metric':<10} {'Mean':>10} {'Median':>10} {'Std':>10}")
    print(f"{'-' * 55}")
    print(f"{'SSIM':<10} {ssim_v.mean():>10.4f} {np.median(ssim_v):>10.4f} {ssim_v.std():>10.4f}")
    print(f"{'PSNR':<10} {psnr_v.mean():>10.2f} {np.median(psnr_v):>10.2f} {psnr_v.std():>10.2f}")
    print(f"{'LPIPS':<10} {lpips_v.mean():>10.4f} {np.median(lpips_v):>10.4f} {lpips_v.std():>10.4f}")
    print(f"{'=' * 55}")

    return ssim_v, psnr_v, lpips_v


def plot_kde_single(results, out_path, title="Inpainting — Metric Distributions"):
    """KDE distributions for SSIM / PSNR / LPIPS for one method. Saves PNG."""
    ssim_v  = np.array([r["ssim"]  for r in results])
    psnr_v  = np.array([r["psnr"]  for r in results])
    lpips_v = np.array([r["lpips"] for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for ax, vals, xlabel, subplot_title in [
        (axes[0], ssim_v,  "SSIM",                "SSIM Distribution"),
        (axes[1], psnr_v,  "PSNR (dB)",           "PSNR Distribution"),
        (axes[2], lpips_v, "LPIPS (lower=better)", "LPIPS Distribution"),
    ]:
        kde = gaussian_kde(vals, bw_method="scott")
        x = np.linspace(vals.min() - 0.05 * np.ptp(vals),
                        vals.max() + 0.05 * np.ptp(vals), 500)
        y = kde(x)
        ax.plot(x, y, color="steelblue", linewidth=2,
                label=f"mean={vals.mean():.3f}")
        ax.fill_between(x, y, alpha=0.20, color="steelblue")
        ax.axvline(vals.mean(), color="steelblue", linestyle="--", linewidth=1.2)
        ax.set_title(subplot_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")


def plot_kde_overlay(results_a, results_b, label_a, label_b, out_path,
                     color_a="steelblue", color_b="darkorange"):
    """Overlaid KDE for two methods side-by-side. Saves PNG."""
    def _get_arrays(results):
        return (
            np.array([r["ssim"]  for r in results]),
            np.array([r["psnr"]  for r in results]),
            np.array([r["lpips"] for r in results]),
        )

    ssim_a, psnr_a, lpips_a = _get_arrays(results_a)
    ssim_b, psnr_b, lpips_b = _get_arrays(results_b)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for ax, vals_a, vals_b, xlabel, subplot_title in [
        (axes[0], ssim_a,  ssim_b,  "SSIM",                "SSIM Distribution"),
        (axes[1], psnr_a,  psnr_b,  "PSNR (dB)",           "PSNR Distribution"),
        (axes[2], lpips_a, lpips_b, "LPIPS (lower=better)", "LPIPS Distribution"),
    ]:
        for vals, label, color in [
            (vals_a, label_a, color_a),
            (vals_b, label_b, color_b),
        ]:
            kde = gaussian_kde(vals, bw_method="scott")
            lo = min(vals_a.min(), vals_b.min()) - 0.05 * max(np.ptp(vals_a), np.ptp(vals_b))
            hi = max(vals_a.max(), vals_b.max()) + 0.05 * max(np.ptp(vals_a), np.ptp(vals_b))
            x = np.linspace(lo, hi, 500)
            y = kde(x)
            ax.plot(x, y, color=color, linewidth=2,
                    label=f"{label} (mean={vals.mean():.3f})")
            ax.fill_between(x, y, alpha=0.20, color=color)
            ax.axvline(vals.mean(), color=color, linestyle="--", linewidth=1.2)
        ax.set_title(subplot_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.suptitle(
        f"{label_a} vs {label_b} — Inpainting Metric Distributions",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")


def show_top10(results, inpainted_dir=None, originals_dir=None, masks_dir=None,
               sort_by="ssim", ascending=True, out_path=None):
    """
    10×3 grid: Original | Masked | Inpainted.

    If inpainted_dir / originals_dir / masks_dir are provided the images are
    loaded from disk; otherwise they are taken from the ``results`` dicts
    (which store PIL images directly when produced by run_metrics()).

    Parameters
    ----------
    sort_by   : metric key to sort by ("ssim", "psnr", "lpips")
    ascending : True → worst first (lowest SSIM), False → best first
    out_path  : if given, saves the figure as PNG
    """
    sorted_r = sorted(results, key=lambda r: r[sort_by], reverse=not ascending)
    top10 = sorted_r[:10]

    direction = "lowest" if ascending else "highest"
    fig, axes = plt.subplots(10, 3, figsize=(15, 50))
    fig.suptitle(
        f"Top 10 — {direction} {sort_by.upper()}",
        fontsize=16, y=1.0,
    )

    for row, r in enumerate(top10):
        idx = r["idx"]

        if originals_dir is not None:
            orig_img = Image.open(os.path.join(originals_dir, f"{idx:04d}.png"))
        else:
            orig_img = r["original"]

        if inpainted_dir is not None:
            inp_img = Image.open(os.path.join(inpainted_dir, f"{idx:04d}.png"))
        else:
            inp_img = r["inpainted"]

        if masks_dir is not None:
            mask_t = torch.load(os.path.join(masks_dir, f"{idx:04d}.pt"),
                                weights_only=True)
        else:
            mask_t = r["mask"]

        masked_vis = apply_mask_for_display(orig_img, mask_t)

        axes[row, 0].imshow(orig_img)
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 0].set_ylabel(
            f"#{idx}\nSSIM={r['ssim']:.3f}\nPSNR={r['psnr']:.1f}\nLPIPS={r['lpips']:.3f}",
            fontsize=9, rotation=0, labelpad=70, va="center",
        )

        axes[row, 1].imshow(masked_vis)
        axes[row, 1].set_title("Masked" if row == 0 else "")
        axes[row, 1].set_xlabel(f'"{r["caption"]}"', fontsize=8, labelpad=6)

        axes[row, 2].imshow(inp_img)
        axes[row, 2].set_title("Inpainted" if row == 0 else "")

        for col in range(3):
            axes[row, col].axis("off")
        axes[row, 1].xaxis.set_visible(True)
        axes[row, 1].tick_params(bottom=False, labelbottom=True)
        axes[row, 1].set_xticks([])

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.show()
