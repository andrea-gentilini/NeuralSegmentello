import os
import random

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset


def random_coarse(mask, rng=None):
    rng = random if rng is None else rng
    # Convert to uint8 and ensure contiguous
    mask = np.ascontiguousarray((mask * 255).astype(np.uint8))

    # band around the edge
    contour_band = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    contour_band = cv2.subtract(contour_band, mask)

    # Get coordinates where noise can be placed (in edge band)
    edge_coords = np.column_stack(np.where(contour_band > 0))

    if len(edge_coords) == 0:
        return (mask > 0).astype(np.uint8)

    noise = np.zeros_like(mask, dtype=np.uint8)

    # Add random circles only around the contour area
    num_circles = rng.randint(15, 40)
    for _ in range(num_circles):
        y, x = edge_coords[rng.randint(0, len(edge_coords) - 1)]
        radius = rng.randint(3, 10)
        cv2.circle(noise, (x, y), radius, 255, -1)

    # Combine noise with original mask
    mask = cv2.bitwise_or(mask, noise)

    # Random dilation
    dilate_kernel_size = rng.randint(7, 15)
    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size)
    )
    mask = cv2.dilate(mask, kernel_dilate, iterations=rng.randint(1, 2))

    # Random erosion
    erode_kernel_size = rng.randint(5, dilate_kernel_size - 1)
    kernel_erode = cv2.getStructuringElement(
        rng.choice([cv2.MORPH_ELLIPSE, cv2.MORPH_RECT, cv2.MORPH_CROSS]),
        (erode_kernel_size, erode_kernel_size),
    )
    mask = cv2.erode(mask, kernel_erode, iterations=1)

    # Gaussian blur + threshold for soft borders
    blur = cv2.GaussianBlur(mask, (5, 5), sigmaX=rng.uniform(2.0, 4.5))
    _, coarse_mask = cv2.threshold(blur, rng.randint(60, 120), 1, cv2.THRESH_BINARY)

    return coarse_mask.astype(np.uint8)


def random_coarse_paintbrush(mask, rng=None):
    rng = random if rng is None else rng
    mask = np.ascontiguousarray((mask > 0).astype(np.uint8) * 255)

    band_width = rng.randint(15, 40)
    contour_band = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_width, band_width)),
        iterations=1,
    )
    contour_band = cv2.subtract(contour_band, mask)
    edge_coords = np.column_stack(np.where(contour_band > 0))

    if len(edge_coords) == 0:
        return (mask > 0).astype(np.uint8)

    noise = np.zeros_like(mask, np.uint8)
    num_circles = rng.randint(8, 18)
    for _ in range(num_circles):
        y, x = edge_coords[rng.randrange(len(edge_coords))]
        radius = rng.randint(band_width // 2, band_width)
        cv2.circle(noise, (x, y), radius, 255, -1)

    mask = cv2.bitwise_or(mask, noise)

    close_size = rng.randint(band_width // 2, band_width)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    blur = cv2.GaussianBlur(mask, (0, 0), sigmaX=rng.uniform(6.0, 12.0))
    _, coarse = cv2.threshold(blur, 127, 1, cv2.THRESH_BINARY)

    return coarse.astype(np.uint8)


def random_coarse_erode(mask, rng=None):
    rng = random if rng is None else rng
    """
    erode
    """
    mask = np.ascontiguousarray((mask > 0).astype(np.uint8) * 255)

    band_width = rng.randint(30, 40)
    contour_band = cv2.erode(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_width, band_width)),
        iterations=1,
    )
    contour_band = cv2.subtract(mask, contour_band)
    edge_coords = np.column_stack(np.where(contour_band > 0))

    if len(edge_coords) == 0:
        return (mask > 0).astype(np.uint8)

    noise = np.zeros_like(mask, np.uint8)
    num_circles = rng.randint(2, 6)
    for _ in range(num_circles):
        y, x = edge_coords[rng.randrange(len(edge_coords))]
        radius = rng.randint(band_width // 4, band_width // 2)
        cv2.circle(noise, (x, y), radius, 255, -1)

    mask = cv2.bitwise_and(mask, cv2.bitwise_not(noise))

    open_size = rng.randint(band_width // 2, band_width)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    blur = cv2.GaussianBlur(mask, (0, 0), sigmaX=rng.uniform(6.0, 12.0))
    _, coarse = cv2.threshold(blur, 127, 1, cv2.THRESH_BINARY)

    return coarse.astype(np.uint8)


def random_coarse_distanza(mask, sigma_noise=12, blur_sigma=6.0, bias=12, np_rng=None):
    mask = (mask > 0).astype(np.uint8)

    d_in = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    d_out = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    signed = d_in - d_out

    np_rng = np.random if np_rng is None else np_rng
    noise = np_rng.standard_normal(mask.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), blur_sigma)

    d_hat = signed + sigma_noise * noise - bias
    coarse = (d_hat > 0).astype(np.uint8)

    return coarse


class CoarseMaskDataset(Dataset):
    """
    Dataset for loading images and their corresponding coarse masks.
    Args:
        dataset_dir (str): Directory containing the dataset.
        transform_type (str): Type of transformation to apply to the masks.
        mode (str): Mode for loading images ("gray" or "rgb").
        max_masks (int): Maximum number of masks to load per image.
        image_gradient (bool): Whether to compute and include image gradient in the output.

    Returns on __getitem__:
        input_tensor (torch.Tensor): Input tensor containing the coarse mask and image.
            Structure is [coarse_mask, image (RGB -> 3 channels, gray -> 1 channel), (optional) gradient].
            So that coarse_mask is always at pos [0], image at [1:-1], and gradient at [-1].
        gt_mask (torch.Tensor): Ground truth mask tensor.
    """

    def __init__(
        self,
        dataset_dir: str,
        transform_type: str = "erode",
        mode: str = "gray",
        max_masks: int = 1,
        image_gradient: bool = False,
        seed: int | None = None,
        image_ids: list[int] | None = None,
    ):
        self.img_dir = os.path.join(dataset_dir, "train2014")
        self.ann_file = os.path.join(dataset_dir, "annotations", "instances_train2014.json")
        self.seed = seed
        if max_masks < 1:
            raise ValueError("max_masks must be positive")
        self.transform_type = transform_type
        self.coco = COCO(self.ann_file)
        self.mode = mode  # "gray", "rgb"
        self.max_masks = max_masks
        self.image_gradient = image_gradient
        all_ids = sorted(self.coco.imgs.keys())
        self.ids = [img_id for img_id in all_ids if len(self.coco.getAnnIds(imgIds=img_id)) > 0]
        if image_ids is not None:
            requested = set(image_ids)
            if not requested.issubset(self.ids):
                raise ValueError("Requested image IDs are missing or have no annotations")
            self.ids = sorted(requested)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        rng = random if self.seed is None else random.Random(self.seed + img_id)
        np_rng = np.random if self.seed is None else np.random.default_rng(self.seed + img_id)
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        # Load and convert image
        if self.mode == "gray":
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)
        elif self.mode == "rgb":
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("wrong mode")

        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        if not anns:
            raise ValueError(f"No annotations found for image id {img_id}")

        # Select annotations
        selected_anns = sorted(anns, key=lambda ann: ann["area"], reverse=True)[: self.max_masks]

        gt_mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        for ann in selected_anns:
            gt_mask = np.maximum(gt_mask, self.coco.annToMask(ann))

        gt_mask = np.clip(gt_mask, 0, 1)

        transform_type = (
            rng.choice(self.transform_type)
            if isinstance(self.transform_type, (list, tuple))
            else self.transform_type
        )
        transforms = {
            "v1": random_coarse,
            "erode": random_coarse_erode,
            "paintbrush": random_coarse_paintbrush,
        }
        if transform_type == "dist":
            coarse_mask = random_coarse_distanza(gt_mask, np_rng=np_rng)
        elif transform_type in transforms:
            coarse_mask = transforms[transform_type](gt_mask, rng=rng)
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
        # Never substitute another source image: doing so crosses subset boundaries.
        # Empty/very small coarse masks remain valid, explicitly scored examples.
        # Convert to tensor
        image_tensor = torch.from_numpy(image)
        gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).float()
        coarse_mask = torch.from_numpy(coarse_mask).unsqueeze(0).float()

        image_tensor = image_tensor.float() / 255.0
        if self.mode == "rgb":
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        elif self.mode == "gray":
            image_tensor = image_tensor.unsqueeze(0)  # HW -> 1 x H x W

        # Optional: compute and stack gradient
        if self.image_gradient:
            if self.mode == "rgb":
                gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray_img = image

            # Compute gradient magnitude using Sobel
            grad_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_mag = grad_mag / (grad_mag.max() + 1e-8)  # Normalize to [0, 1]
            grad_tensor = torch.from_numpy(grad_mag).unsqueeze(0).float()  # 1 x H x W

            # Put coarse first, then gradient, then image
            input_tensor = torch.cat([coarse_mask, image_tensor, grad_tensor], dim=0)
        else:
            # Put coarse first, then image
            input_tensor = torch.cat([coarse_mask, image_tensor], dim=0)

        return input_tensor, gt_mask


class SimpleCoarseMaskDataset(Dataset):
    """
    Simplified Dataset for test experiments.
    Expects precomputed numpy arrays for coarse masks, images, and ground truth masks.

    Directory structure:
        dataset_dir/
            coarse/input_coarse_mask_{i}.npy
            image/input_image_{i}.npy
            gt/ground_truth_mask_{i}.npy

    Args:
        dataset_dir (str): Root directory containing 'coarse', 'image', and 'gt' subdirectories.
        mode (str): "gray" or "rgb" for the input image mode.
        image_gradient (bool): Whether to compute and include image gradient in the input tensor.

    Returns:
        input_tensor (torch.Tensor): Tensor with structure [coarse_mask, image, (optional) gradient].
        gt_mask (torch.Tensor): Ground truth mask.
    """

    def __init__(self, dataset_dir: str, mode: str = "gray", image_gradient: bool = False):
        self.coarse_dir = os.path.join(dataset_dir, "coarse")
        self.image_dir = os.path.join(dataset_dir, "image")
        self.gt_dir = os.path.join(dataset_dir, "gt")
        self.mode = mode
        self.image_gradient = image_gradient

        # Extract indices from available files
        self.file_indices = sorted(
            [
                int(f.split("_")[-1].split(".")[0])
                for f in os.listdir(self.coarse_dir)
                if f.startswith("input_coarse_mask_") and f.endswith(".npy")
            ]
        )

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        index = self.file_indices[idx]

        # Load data
        coarse_mask = np.load(os.path.join(self.coarse_dir, f"input_coarse_mask_{index}.npy"))
        image = np.load(os.path.join(self.image_dir, f"input_image_{index}.npy"))
        gt_mask = np.load(os.path.join(self.gt_dir, f"ground_truth_mask_{index}.npy"))

        coarse_mask = (coarse_mask > 0).astype(np.float32)
        gt_mask = (gt_mask > 0).astype(np.float32)
        image = np.asarray(image)
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
            raise ValueError("Image must contain finite values in [0, 1] or [0, 255]")
        normalized = image.astype(np.float32)
        if np.issubdtype(image.dtype, np.integer) or normalized.max() > 1:
            normalized /= 255.0
        if self.mode == "gray":
            if normalized.ndim == 3:
                normalized = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)
            image_tensor = torch.from_numpy(np.ascontiguousarray(normalized)).unsqueeze(0)
            gray_img = normalized
        elif self.mode == "rgb":
            if normalized.ndim == 2:
                normalized = np.repeat(normalized[..., None], 3, axis=2)
            image_tensor = torch.from_numpy(np.ascontiguousarray(normalized)).permute(2, 0, 1)
            gray_img = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError("Mode must be 'gray' or 'rgb'")
        if coarse_mask.shape != gt_mask.shape or coarse_mask.shape != normalized.shape[:2]:
            raise ValueError("Image, coarse mask and target mask must share spatial dimensions")
        coarse_mask_tensor = torch.from_numpy(coarse_mask).unsqueeze(0)
        gt_mask_tensor = torch.from_numpy(gt_mask).unsqueeze(0)
        channels = [coarse_mask_tensor, image_tensor]
        if self.image_gradient:
            grad_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            magnitude /= magnitude.max() + 1e-8
            channels.append(torch.from_numpy(magnitude).unsqueeze(0))
        input_tensor = torch.cat(channels, dim=0)

        return input_tensor, gt_mask_tensor


def collate_fn(batch):
    x_list, y_list = zip(*batch)
    x = torch.stack(x_list)  # [B, 3, H, W]
    y = torch.stack(y_list)  # [B, 1, H, W]
    return x, y


# fake one-image dataset
class SingleSampleDataset(Dataset):
    def __init__(self, item):
        self.item = item

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return self.item


def split_image_ids(image_ids, validation_fraction=0.1, seed=42):
    """Split unique source images; all annotations for an image stay together."""
    ids = sorted(set(image_ids))
    if not 0 < validation_fraction < 1 or len(ids) < 2:
        raise ValueError(
            "Need at least two images and a validation fraction strictly between 0 and 1"
        )
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_valid = max(1, min(len(ids) - 1, int(len(ids) * validation_fraction)))
    return sorted(ids[n_valid:]), sorted(ids[:n_valid])
