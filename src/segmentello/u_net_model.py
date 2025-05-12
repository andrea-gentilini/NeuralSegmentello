from data.config import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


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


class UNet(nn.Module):
  """
  U-Net architecture for image segmentation.
  """
  def __init__(self, in_channels=2, out_channels=1):
    super(UNet, self).__init__()

    # Encoder
    self.down1 = DoubleConv(in_channels, 64)
    self.pool1 = nn.MaxPool2d(2)
    self.down2 = DoubleConv(64, 128)
    self.pool2 = nn.MaxPool2d(2)
    self.down3 = DoubleConv(128, 256)
    self.pool3 = nn.MaxPool2d(2)
    self.down4 = DoubleConv(256, 512)
    self.pool4 = nn.MaxPool2d(2)

    # Bottleneck
    self.bottleneck = DoubleConv(512, 1024)

    # Decoder
    self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
    self.dec1 = DoubleConv(1024, 512)
    self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.dec2 = DoubleConv(512, 256)
    self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.dec3 = DoubleConv(256, 128)
    self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.dec4 = DoubleConv(128, 64)

    self.final = nn.Conv2d(64, out_channels, kernel_size=1)

  def forward(self, x):
    # Encoder
    c1 = self.down1(x)
    p1 = self.pool1(c1)

    c2 = self.down2(p1)
    p2 = self.pool2(c2)

    c3 = self.down3(p2)
    p3 = self.pool3(c3)

    c4 = self.down4(p3)
    p4 = self.pool4(c4)

    # Bottleneck
    bn = self.bottleneck(p4)

    # Decoder
    up1 = self.up1(bn)
    merge1 = torch.cat([up1, c4], dim=1)
    c5 = self.dec1(merge1)

    up2 = self.up2(c5)
    merge2 = torch.cat([up2, c3], dim=1)
    c6 = self.dec2(merge2)

    up3 = self.up3(c6)
    merge3 = torch.cat([up3, c2], dim=1)
    c7 = self.dec3(merge3)

    up4 = self.up4(c7)
    merge4 = torch.cat([up4, c1], dim=1)
    c8 = self.dec4(merge4)

    out = self.final(c8)
    return out
  

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





class UNetLightning(pl.LightningModule):
  def __init__(self):
    super(UNetLightning, self).__init__()
    self.save_hyperparameters()
    self.model = UNet()
    # self.loss_fn = nn.CrossEntropyLoss()
    self.loss_fn = nn.BCEWithLogitsLoss()

  def forward(self, x):
    return self.model(x)

  # def training_step(self, batch, batch_idx):
  #   images, masks = batch
  #   preds = self(images)
  #   loss = self.loss_fn(preds, masks.squeeze(1))
  #   self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
  #   return loss

  # def validation_step(self, batch, batch_idx):
  #   images, masks = batch
  #   preds = self(images)
  #   loss = self.loss_fn(preds, masks.squeeze(1))

  #   # Compute IoU
  #   preds_class = torch.argmax(preds, dim=1)  # (N, H, W)
  #   ious = []
  #   for i in range(images.size(0)):
  #     iou = compute_iou(preds_class[i], masks[i].squeeze(0))
  #     ious.append(iou)
  #   batch_iou = torch.mean(torch.stack(ious))

  #   # Log metrics
  #   self.log('val_loss', loss, on_epoch=True, prog_bar=True)
  #   self.log('val_iou', batch_iou, on_epoch=True, prog_bar=True)
  #   return loss

  def training_step(self, batch, batch_idx):
    images, masks = batch
    preds = self(images)
    targets = masks.squeeze(1).float()   # shape (N, H, W)
    preds = preds.squeeze(1)             # shape (N, H, W)
    loss = self.loss_fn(preds, targets)
    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
      images, masks = batch
      preds = self(images)
      targets = masks.squeeze(1).float()
      preds = preds.squeeze(1)

      loss = self.loss_fn(preds, targets)

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
  

def main() -> None:
  # TODO add dataloader part, maybe in separate files?

  pl.seed_everything(SEED)

  checkpoint_callback = ModelCheckpoint(
    dirpath=MODEL_CHECKPOINT_DIR,
    filename="best-checkpoint",
    save_top_k=SAVE_TOP_K,
    verbose=True,
    monitor=MONITOR_METRIC,
    mode="min"
  )

  early_stop_callback = EarlyStopping(
    monitor=MONITOR_METRIC,
    patience=10,
    verbose=True,
    mode="min"
  )

  trainer = pl.Trainer(
    max_epochs=EPOCHS,
    callbacks=[checkpoint_callback, early_stop_callback],
    accelerator="auto",
    devices=1,
    log_every_n_steps=5
  )

  model = UNetLightning()
  # trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
  main()