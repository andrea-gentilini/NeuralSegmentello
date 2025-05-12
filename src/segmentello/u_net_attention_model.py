from data.config import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


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
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
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


def compute_iou(
    pred_mask: torch.Tensor, 
    true_mask: torch.Tensor
) -> torch.Tensor:
    """
    Intersection over Union for a binary segmentation mask.

    Returns a torch.Tensor with only one iou float number
    """
    # Ensure binary
    pred_mask = (pred_mask > 0).float()
    true_mask = (true_mask > 0).float()

    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    iou = intersection / union if union > 0 else torch.tensor(1.0) if intersection == 0 else torch.tensor(0.0)
    return iou


class Coarse2FineUNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet(in_channels=2, out_channels=1)
        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def compute_loss(self, preds, targets):
        """
        helper function to compute same loss in the training and in the 
        validation step
        """
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return 0.5 * bce_loss + 0.5 * dice_loss


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x_list, y_list = batch
        losses = []
        for x, y in zip(x_list, y_list):
            x = x.unsqueeze(0).to(self.device)
            y = y.unsqueeze(0).to(self.device)
            logits = self(x)
            loss = self.compute_loss(logits, x)
            losses.append(loss)
        total_loss = torch.stack(losses).mean()
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss


    def validation_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        targets = masks.squeeze(1).float()
        preds = preds.squeeze(1)

        loss = self.compute_loss(preds, targets)

        # Compute IoU
        preds_class = (torch.sigmoid(preds) > 0.5).float()  # binarize logits after sigmoid
        ious = []
        for i in range(images.size(0)):
            iou = compute_iou(preds_class[i], targets[i])
            ious.append(iou)
        batch_iou = torch.mean(torch.stack(ious))

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_iou', batch_iou, on_epoch=True, prog_bar=True)
        return loss


    def test_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        preds_class = torch.argmax(preds, dim=1)

        ious = []
        for i in range(images.size(0)):
            iou = compute_iou(preds_class[i], masks[i].squeeze(0))
            ious.append(iou)

        batch_iou = torch.mean(torch.stack(ious))
        self.log('test_iou', batch_iou, on_step=False, on_epoch=True, prog_bar=True)
        return batch_iou


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        return optimizer