import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff


# Modules
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class SobelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        sobel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        target = target.float()

        # Ensure Sobel filters are on the same device as pred
        sobel_x = self.sobel_x.to(pred.device)
        sobel_y = self.sobel_y.to(pred.device)

        grad_pred_x = F.conv2d(pred, sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, sobel_y, padding=1)
        grad_target_x = F.conv2d(target, sobel_x, padding=1)
        grad_target_y = F.conv2d(target, sobel_y, padding=1)

        grad_pred = torch.sqrt(grad_pred_x**2 + grad_pred_y**2 + 1e-8)
        grad_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + 1e-8)

        return F.mse_loss(grad_pred, grad_target)


class AttentionBlock(nn.Module):
    def __init__(self, g_channels, x_channels, intermediate_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, intermediate_channels, kernel_size=1),
            nn.BatchNorm2d(intermediate_channels),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, intermediate_channels, kernel_size=1),
            nn.BatchNorm2d(intermediate_channels),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Align shapes explicitly
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(
                g1, size=x1.shape[2:], mode="bilinear", align_corners=False
            )

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# Metrics
def compute_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    pred_mask = (pred_mask > 0).float()
    true_mask = (true_mask > 0).float()
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    iou = (
        intersection / union
        if union > 0
        else torch.tensor(1.0)
        if intersection == 0
        else torch.tensor(0.0)
    )
    return iou


def compute_pixel_accuracy(
    pred_mask: torch.Tensor, true_mask: torch.Tensor
) -> torch.Tensor:
    pred_mask = (pred_mask > 0).float()
    true_mask = (true_mask > 0).float()
    correct = (pred_mask == true_mask).float().sum()
    total = true_mask.numel()
    return correct / total


def extract_boundary(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    padding = kernel_size // 2
    eroded = F.max_pool2d(1 - mask.unsqueeze(0), kernel_size, stride=1, padding=padding)
    boundary = mask - (1 - eroded.squeeze(0).squeeze(0))
    return (boundary > 0).float()


def compute_boundary_iou(
    pred_mask: torch.Tensor, true_mask: torch.Tensor, kernel_size: int = 3
) -> torch.Tensor:
    pred_mask = (pred_mask > 0).float()
    true_mask = (true_mask > 0).float()

    pred_boundary = extract_boundary(pred_mask, kernel_size)
    true_boundary = extract_boundary(true_mask, kernel_size)

    intersection = (pred_boundary * true_boundary).sum()
    union = pred_boundary.sum() + true_boundary.sum() - intersection
    iou = (
        intersection / union
        if union > 0
        else torch.tensor(1.0)
        if intersection == 0
        else torch.tensor(0.0)
    )
    return iou


def mask_to_numpy_coords(mask: torch.Tensor) -> np.ndarray:
    return mask.nonzero(as_tuple=False).cpu().numpy()


def compute_hausdorff_distance(
    pred_mask: torch.Tensor, true_mask: torch.Tensor
) -> torch.Tensor:
    pred_mask = (pred_mask > 0).float()
    true_mask = (true_mask > 0).float()

    pred_coords = mask_to_numpy_coords(pred_mask)
    true_coords = mask_to_numpy_coords(true_mask)

    if pred_coords.size == 0 and true_coords.size == 0:
        return torch.tensor(0.0)  # no structure in either
    if pred_coords.size == 0 or true_coords.size == 0:
        return torch.tensor(float("inf"))  # one is empty

    hd_forward = directed_hausdorff(pred_coords, true_coords)[0]
    hd_backward = directed_hausdorff(true_coords, pred_coords)[0]
    return torch.tensor(max(hd_forward, hd_backward))
