import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, distance_transform_edt


def get_structural_edge_map(image_tensor, mask):
    gray = image_tensor.mean(dim=1, keepdim=True)

    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image_tensor.device).float().view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=image_tensor.device).float().view(1, 1, 3, 3)
    gx, gy = F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2)
    known_edges = mag * mask

    pooled_edges = F.max_pool2d(known_edges, kernel_size=15, stride=1, padding=7)

    kernel_size = 11
    sigma = 3.0
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=image_tensor.device).float()
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    gauss_kernel = (gauss.unsqueeze(1) * gauss.unsqueeze(0)).view(1, 1, kernel_size, kernel_size)

    return F.conv2d(pooled_edges, gauss_kernel, padding=kernel_size // 2)


def permute_noise_by_edges(noise, edge_map):
    b, c, h, w = noise.shape
    flat_noise = noise.view(b, c, -1)
    noise_idx = torch.argsort(flat_noise.abs(), dim=-1)
    flat_edges = edge_map.repeat(1, c, 1, 1).view(b, c, -1)
    edge_idx = torch.argsort(flat_edges, dim=-1)
    permuted_noise = torch.zeros_like(flat_noise)
    for i in range(b):
        for j in range(c):
            permuted_noise[i, j, edge_idx[i, j]] = flat_noise[i, j, noise_idx[i, j]]
    return permuted_noise.view(b, c, h, w)


def get_local_boundary_color_map(image_tensor, mask):
    mask_np = mask[0, 0].cpu().numpy()

    kernel = np.ones((3, 3))
    dilated_hole = binary_dilation(mask_np == 0, structure=kernel)
    boundary_np = (mask_np == 1) & dilated_hole

    _, indices = distance_transform_edt(~boundary_np, return_distances=True, return_indices=True)

    img_np = image_tensor[0].cpu().numpy()
    color_map_np = np.zeros_like(img_np)
    for c in range(3):
        color_map_np[c] = img_np[c][indices[0], indices[1]]

    return torch.from_numpy(color_map_np).unsqueeze(0).to(image_tensor.device)
