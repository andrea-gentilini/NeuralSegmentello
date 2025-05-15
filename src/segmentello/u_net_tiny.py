import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from u_net_attention_model import compute_iou


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


class UNetMini(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[16, 32]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # Downsampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Upsampling
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, 2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx//2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](x)

        return self.final_conv(x)


class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1.):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        return 1 - ((2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth))


class Coarse2FineTiny(pl.LightningModule):
    def __init__(self, lr=1e-3, features=[16, 32]):
        super().__init__()
        self.model = UNetMini(in_channels=3, out_channels=1, features=features)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        preds = self(x)
        loss = self.bce(preds, y) + self.dice(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        preds = self(x)
        loss = self.bce(preds, y) + self.dice(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        # Compute IoU for the batch
        preds_class = (preds > 0.5).float()  # Convert predictions to binary values
        ious = [compute_iou(preds_class[i], y[i]) for i in range(x.size(0))]
        batch_iou = torch.stack(ious).mean()  # Compute mean IoU for the batch
        self.log("val_iou", batch_iou, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
