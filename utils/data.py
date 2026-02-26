import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torchvision
from tqdm.auto import tqdm

IMG_URL  = "http://images.cocodataset.org/zips/val2017.zip"
ANN_URL  = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

_EXCLUDE_SUPERCATS = {"person", "animal"}

_pbar = None


def _reporthook(block_num, block_size, total_size):
    global _pbar
    if _pbar is None:
        _pbar = tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024)
    downloaded = block_num * block_size
    _pbar.n = min(downloaded, total_size)
    _pbar.refresh()
    if downloaded >= total_size:
        _pbar.close()
        _pbar = None


def load_data(data_root, seed=42, n_images=1000):
    coco_root    = Path(data_root) / "coco"
    coco_img_dir = coco_root / "val2017"
    ann_file     = coco_root / "annotations" / "captions_val2017.json"

    coco_root.mkdir(parents=True, exist_ok=True)

    if not coco_img_dir.is_dir():
        img_zip = coco_root / "val2017.zip"
        print("Downloading COCO val2017 images (~1 GB)...")
        urllib.request.urlretrieve(IMG_URL, img_zip, _reporthook)
        print("Extracting images...")
        with zipfile.ZipFile(img_zip) as z:
            z.extractall(coco_root)
        img_zip.unlink()
        print("Done.")
    else:
        print(f"COCO images already present: {coco_img_dir}")

    if not ann_file.exists():
        ann_zip = coco_root / "annotations_trainval2017.zip"
        print("Downloading COCO annotations (~240 MB)...")
        urllib.request.urlretrieve(ANN_URL, ann_zip, _reporthook)
        print("Extracting annotations...")
        with zipfile.ZipFile(ann_zip) as z:
            z.extractall(coco_root)
        ann_zip.unlink()
        print("Done.")
    else:
        print(f"COCO annotations already present: {ann_file}")

    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
    captions = {}
    for ann in coco_data["annotations"]:
        captions.setdefault(ann["image_id"], []).append(ann["caption"])

    inst_file = coco_root / "annotations" / "instances_val2017.json"
    with open(inst_file, "r") as f:
        inst_data = json.load(f)
    exclude_cat_ids = {
        c["id"] for c in inst_data["categories"]
        if c["supercategory"] in _EXCLUDE_SUPERCATS
    }
    excluded_image_ids = {
        ann["image_id"] for ann in inst_data["annotations"]
        if ann["category_id"] in exclude_cat_ids
    }

    valid_ids = [
        img_id for img_id, caps in captions.items()
        if (coco_img_dir / id_to_filename[img_id]).exists()
        and img_id not in excluded_image_ids
    ]
    print(f"Valid images (excluding people/animals): {len(valid_ids)}")

    rng = np.random.RandomState(seed)
    selected_ids = rng.choice(valid_ids, size=n_images, replace=False).tolist()

    mnist = torchvision.datasets.MNIST(root=str(data_root), train=False, download=True)
    mnist_indices = rng.choice(len(mnist), size=n_images, replace=False)

    print(f"MNIST dataset size: {len(mnist)}")
    print(f"Sampled {n_images} image-mask pairs.")
    return selected_ids, id_to_filename, captions, mnist, mnist_indices
