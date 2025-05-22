import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.config import LR
from data.utils import (
    AttentionBlock,
    DiceLoss,
    DoubleConv,
    SobelLoss,
    compute_boundary_iou,
    compute_hausdorff_distance,
    compute_iou,
    compute_pixel_accuracy,
)


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
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.ups.append(DoubleConv(feature * 2, feature))
            self.attentions.append(
                AttentionBlock(
                    g_channels=feature,
                    x_channels=feature,
                    intermediate_channels=feature // 2,
                )
            )

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
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
            attn = self.attentions[i // 2](
                g=x, x=skip[i // 2]
            )  # Apply attention AFTER upsample
            if x.shape != attn.shape:
                x = F.interpolate(x, size=attn.shape[2:])
            x = torch.cat([attn, x], dim=1)
            x = self.ups[i + 1](x)

        return self.final_conv(x)


class Coarse2FineUNetAttention(pl.LightningModule):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        lr=LR,
        loss_weights=None,
        losses=["bce", "dice"],
    ):
        """
        losses: {"bce","dice","boundary"}
        """
        super().__init__()
        loss_to_class = {
            "bce": nn.BCEWithLogitsLoss(),
            "dice": DiceLoss(),
            "boundary": SobelLoss(),
        }
        self.model = UNet(in_channels=in_channels, out_channels=out_channels)
        self.losses = [loss_to_class[loss] for loss in losses]
        self.default_weights = loss_weights or [1 / len(losses)] * (len(losses))
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, preds, y):
        return sum(
            [w * loss(preds, y) for w, loss in zip(self.default_weights, self.losses)]
        )

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: [B, 3, H, W], y: [B, 1, H, W]
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self(x)  # [B, 1, H, W]
        loss = self.compute_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        preds = self(x)
        loss = self.compute_loss(preds, y)
        self.log("val_loss", loss, prog_bar=True)

        preds_class = (preds > 0.5).float()
        ious = []
        accuracies = []
        boundary_ious = []
        hausdorff_dists = []

        for i in range(x.size(0)):
            pred_i = preds_class[i]
            true_i = y[i]

            ious.append(compute_iou(pred_i, true_i))
            accuracies.append(compute_pixel_accuracy(pred_i, true_i))
            boundary_ious.append(compute_boundary_iou(pred_i, true_i))
            hausdorff_dists.append(compute_hausdorff_distance(pred_i, true_i))

        mean_iou = torch.stack(ious).mean()
        mean_acc = torch.stack(accuracies).mean()
        mean_biou = torch.stack(boundary_ious).mean()
        mean_hd = torch.stack(hausdorff_dists).mean()

        self.log("val_iou", mean_iou, prog_bar=True)
        self.log("val_accuracy", mean_acc, prog_bar=True)
        self.log("val_boundary_iou", mean_biou, prog_bar=False)
        self.log("val_hausdorff", mean_hd, prog_bar=False)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
