from data.config import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class DoubleConv(nn.Module):
  """
  A helper module that performs two convolutional operations:
  Conv -> ReLU -> Conv -> ReLU
  """
  def __init__(self, in_channels, out_channels):
    super(DoubleConv, self).__init__()
    self.net = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
    )

  def forward(self, x):
    return self.net(x)




class AttentionBlock(nn.Module):
    def __init__(self, g_channels, x_channels, intermediate_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, intermediate_channels, kernel_size=1),
            nn.BatchNorm2d(intermediate_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, intermediate_channels, kernel_size=1),
            nn.BatchNorm2d(intermediate_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Align shapes explicitly
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi



class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.attentions = nn.ModuleList()

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, 2, 2))
            self.ups.append(DoubleConv(feature*2, feature))
            self.attentions.append(AttentionBlock(g_channels=feature, x_channels=feature, intermediate_channels=feature//2))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skip = []
        for down in self.downs:
            x = down(x)
            skip.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip = skip[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)  # Upsample first
            attn = self.attentions[i//2](g=x, x=skip[i//2])  # Apply attention AFTER upsample
            if x.shape != attn.shape:
                x = F.interpolate(x, size=attn.shape[2:])
            x = torch.cat([attn, x], dim=1)
            x = self.ups[i+1](x)


        return self.final_conv(x)
    





class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class SobelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        target = target.float()

        grad_pred_x = F.conv2d(pred, self.sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, self.sobel_y, padding=1)
        grad_target_x = F.conv2d(target, self.sobel_x, padding=1)
        grad_target_y = F.conv2d(target, self.sobel_y, padding=1)

        grad_pred = torch.sqrt(grad_pred_x**2 + grad_pred_y**2 + 1e-8)
        grad_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + 1e-8)

        return F.mse_loss(grad_pred, grad_target)


class RefinementLoss(nn.Module):
    """
    Loss for refining coarse masks to match GT masks.
    Encourages:
    - recovering missing GT (coarse=0, GT=1),
    - removing coarse hallucinations (coarse=1, GT=0),
    - penalizing hallucinated new regions (coarse=0, GT=0).
    """
    def __init__(self, weights=REFINEMENT_PENALTY, threshold=0.5):
        super().__init__()
        # Default weights
        self.weights = weights
        # Threshold for coarse mask values to be converted to 0 or 1
        self.threshold = threshold

    def forward(self, preds, gt, coarse):
        preds = torch.sigmoid(preds)
        gt = gt.float()
        
        # Apply threshold to coarse mask to classify as 0 or 1
        coarse = (coarse > self.threshold).float()

        # Masks
        recover_mask = (gt == 1) & (coarse == 0)
        delete_mask = (gt == 0) & (coarse == 1)
        hallucinate_mask = (gt == 0) & (coarse == 0)
        soft_penalty_mask = (gt == 0) & (coarse == 1)

        # Reshape masks to match preds
        recover_mask = recover_mask.squeeze(0).expand_as(preds)  # Add batch and channel dimension
        delete_mask = delete_mask.squeeze(0).expand_as(preds)
        hallucinate_mask = hallucinate_mask.squeeze(0).expand_as(preds)
        soft_penalty_mask = soft_penalty_mask.squeeze(0).expand_as(preds)

        loss = 0.0
        if recover_mask.any():
            loss += self.weights['recover'] * F.binary_cross_entropy(preds[recover_mask], gt[recover_mask])
        if delete_mask.any():
            loss += self.weights['delete'] * F.binary_cross_entropy(preds[delete_mask], gt[delete_mask])
        if hallucinate_mask.any():
            loss += self.weights['hallucinate'] * F.binary_cross_entropy(preds[hallucinate_mask], gt[hallucinate_mask])
        if soft_penalty_mask.any():
            loss += self.weights['soft_penalty'] * F.binary_cross_entropy(preds[soft_penalty_mask], gt[soft_penalty_mask])

        return loss


def compute_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    pred_mask = (pred_mask > 0).float()
    true_mask = (true_mask > 0).float()
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    iou = intersection / union if union > 0 else torch.tensor(1.0) if intersection == 0 else torch.tensor(0.0)
    return iou


class Coarse2FineUNetSmall(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=1, lr=1e-3,
                 starting_loss_weights=None, refinement_penalty=None,
                 learnable_weights=True):
        super().__init__()
        self.model = UNet(in_channels=in_channels, out_channels=out_channels)

        # Loss functions
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.boundary = SobelLoss()
        self.refine = RefinementLoss(weights=refinement_penalty)

        # Loss weights
        default_weights = starting_loss_weights or [0.3, 0.2, 0.2, 0.3]
        if learnable_weights:
            self.loss_weights = nn.Parameter(torch.tensor(default_weights), requires_grad=True)
        else:
            self.register_buffer("loss_weights", torch.tensor(default_weights))  # static weights

        self.learnable_weights = learnable_weights
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, preds, targets, coarse):
        # Individual losses
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        boundary_loss = self.boundary(preds, targets)
        refinement_loss = self.refine(preds, targets, coarse)

        # Normalize weights using softmax
        weights = F.softmax(self.loss_weights, dim=0)
        total_loss = (
            weights[0] * bce_loss +
            weights[1] * dice_loss +
            weights[2] * boundary_loss +
            weights[3] * refinement_loss
        )

        # Logging component losses
        self.log("loss_bce", bce_loss, prog_bar=True)
        self.log("loss_dice", dice_loss, prog_bar=True)
        self.log("loss_boundary", boundary_loss, prog_bar=True)
        self.log("loss_refine", refinement_loss, prog_bar=True)
        #   for name, weight in zip(ORDER_LOSS_WEIGHTS, weights):
        #       self.log(f"loss_weight_{name}", weight.item(), prog_bar=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: [B, 3, H, W], y: [B, 1, H, W]
        x = x.to(self.device)
        y = y.to(self.device)

        coarse = x[:, 0:1, :, :]        # [B, 1, H, W]
        logits = self(x)                # [B, 1, H, W]

        loss = self.compute_loss(logits, y, coarse)

        self.log("train_loss", loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch  # x: [B, 3, H, W], y: [B, 1, H, W]
        x = x.to(self.device)
        y = y.to(self.device)

        coarse = x[:, 0:1, :, :]        # [B, 1, H, W]
        preds = self(x)                 # [B, 1, H, W]

        loss = self.compute_loss(preds, y, coarse)

        preds_class = (torch.sigmoid(preds) > 0.5).float()
        targets = y.float()

        # IoU per ogni sample nel batch
        ious = [compute_iou(preds_class[i, 0], targets[i, 0]) for i in range(x.size(0))]
        batch_iou = torch.stack(ious).mean()

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_iou', batch_iou, on_epoch=True, prog_bar=True)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
