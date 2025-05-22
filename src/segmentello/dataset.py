import os
import random

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset


def random_coarse(mask):
    # Convert to uint8 and ensure contiguous
    mask = np.ascontiguousarray((mask * 255).astype(np.uint8))

    # band around the edge
    contour_band = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    contour_band = cv2.subtract(contour_band, mask)

    # Get coordinates where noise can be placed (in edge band)
    edge_coords = np.column_stack(np.where(contour_band > 0))

    if len(edge_coords) == 0:
        return mask  # Return the original mask if no edge is detected

    noise = np.zeros_like(mask, dtype=np.uint8)

    # Add random circles only around the contour area
    num_circles = random.randint(15, 40)
    for _ in range(num_circles):
        y, x = edge_coords[random.randint(0, len(edge_coords) - 1)]
        radius = random.randint(3, 10)
        cv2.circle(noise, (x, y), radius, 255, -1)

    # Combine noise with original mask
    mask = cv2.bitwise_or(mask, noise)

    # Random dilation
    dilate_kernel_size = random.randint(7, 15)
    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size)
    )
    mask = cv2.dilate(mask, kernel_dilate, iterations=random.randint(1, 2))

    # Random erosion
    erode_kernel_size = random.randint(5, dilate_kernel_size - 1)
    kernel_erode = cv2.getStructuringElement(
        random.choice([cv2.MORPH_ELLIPSE, cv2.MORPH_RECT, cv2.MORPH_CROSS]),
        (erode_kernel_size, erode_kernel_size),
    )
    mask = cv2.erode(mask, kernel_erode, iterations=1)

    # Gaussian blur + threshold for soft borders
    blur = cv2.GaussianBlur(mask, (5, 5), sigmaX=random.uniform(2.0, 4.5))
    _, coarse_mask = cv2.threshold(blur, random.randint(60, 120), 1, cv2.THRESH_BINARY)

    return coarse_mask.astype(np.uint8)


def random_coarse_paintbrush(mask):
    mask = np.ascontiguousarray((mask > 0).astype(np.uint8) * 255)

    band_width = random.randint(15, 40)
    contour_band = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_width, band_width)),
        iterations=1,
    )
    contour_band = cv2.subtract(contour_band, mask)
    edge_coords = np.column_stack(np.where(contour_band > 0))

    if len(edge_coords) == 0:
        return mask  # Return the original mask if no edge is detected

    noise = np.zeros_like(mask, np.uint8)
    num_circles = random.randint(8, 18)
    for _ in range(num_circles):
        y, x = edge_coords[random.randrange(len(edge_coords))]
        radius = random.randint(band_width // 2, band_width)
        cv2.circle(noise, (x, y), radius, 255, -1)

    mask = cv2.bitwise_or(mask, noise)

    close_size = random.randint(band_width // 2, band_width)
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    blur = cv2.GaussianBlur(mask, (0, 0), sigmaX=random.uniform(6.0, 12.0))
    _, coarse = cv2.threshold(blur, 127, 1, cv2.THRESH_BINARY)

    return coarse.astype(np.uint8)


def random_coarse_erode(mask):
    """
    erode
    """
    mask = np.ascontiguousarray((mask > 0).astype(np.uint8) * 255)

    band_width = random.randint(30, 40)
    contour_band = cv2.erode(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_width, band_width)),
        iterations=1,
    )
    contour_band = cv2.subtract(mask, contour_band)
    edge_coords = np.column_stack(np.where(contour_band > 0))

    if len(edge_coords) == 0:
        return mask  # Return the original mask if no edge is detected

    noise = np.zeros_like(mask, np.uint8)
    num_circles = random.randint(2, 6)
    for _ in range(num_circles):
        y, x = edge_coords[random.randrange(len(edge_coords))]
        radius = random.randint(band_width // 4, band_width // 2)
        cv2.circle(noise, (x, y), radius, 255, -1)

    mask = cv2.bitwise_and(mask, cv2.bitwise_not(noise))

    open_size = random.randint(band_width // 2, band_width)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    blur = cv2.GaussianBlur(mask, (0, 0), sigmaX=random.uniform(6.0, 12.0))
    _, coarse = cv2.threshold(blur, 127, 1, cv2.THRESH_BINARY)

    return coarse.astype(np.uint8)


def random_coarse_distanza(mask, sigma_noise=12, blur_sigma=6.0, bias=12):
    mask = (mask > 0).astype(np.uint8)

    d_in = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    d_out = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    signed = d_in - d_out

    noise = np.random.randn(*mask.shape).astype(np.float32)
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
    ):
        self.img_dir = os.path.join(dataset_dir, "train2014")
        self.ann_file = os.path.join(
            dataset_dir, "annotations", "instances_train2014.json"
        )
        self.transform_type = transform_type
        self.coco = COCO(self.ann_file)
        self.mode = mode  # "gray", "rgb"
        self.max_masks = max_masks
        self.image_gradient = image_gradient
        all_ids = list(self.coco.imgs.keys())
        self.ids = [
            img_id for img_id in all_ids if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        # Load and convert image
        if self.mode == "gray":
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        elif self.mode == "rgb":
            image = cv2.imread(img_path)
            image = image[:, :, ::-1]  # BGR to RGB
        else:
            raise ValueError("wrong mode")

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        if not anns:
            raise ValueError(f"No annotations found for image id {img_id}")

        # Select annotations
        selected_anns = sorted(anns, key=lambda ann: ann["area"], reverse=True)[
            : self.max_masks
        ]

        gt_mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        for ann in selected_anns:
            gt_mask += self.coco.annToMask(ann)

        gt_mask = np.clip(gt_mask, 0, 1)

        if isinstance(self.transform_type, list):
            # Randomly select a transform type from the list
            self.transform_type = random.choice(self.transform_type)

        # Apply coarse transform
        if self.transform_type == "v1":
            coarse_mask = random_coarse(gt_mask)
        elif self.transform_type == "erode":
            coarse_mask = random_coarse_erode(gt_mask)
        elif self.transform_type == "paintbrush":
            coarse_mask = random_coarse_paintbrush(gt_mask)
        elif self.transform_type == "dist":
            coarse_mask = random_coarse_distanza(gt_mask)
        else:
            raise ValueError("Unknown transform type")

        # Skip if the coarse mask is too small (zero or very small area)
        if np.sum(coarse_mask) < 50:
            return self.__getitem__((idx + 1) % len(self))
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
