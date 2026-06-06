import os

import numpy as np
import torch


def setup_eval_dirs(results_dir, seed):
    originals_dir = os.path.join(results_dir, "originals")
    inpainted_dir = os.path.join(results_dir, "inpainted")
    masks_dir     = os.path.join(results_dir, "masks")

    for d in (originals_dir, inpainted_dir, masks_dir):
        os.makedirs(d, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    return originals_dir, inpainted_dir, masks_dir
