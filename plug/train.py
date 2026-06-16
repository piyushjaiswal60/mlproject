# ============================================================
# FILE: train.py
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os

from plug.unet_model import get_model
from plug.dataset    import get_loaders


# ── Loss: BCE + Dice combined ─────────────────────────────────
class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.bce    = nn.BCELoss()
        self.smooth = smooth

    def dice_loss(self, pred, target):
        pred   = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        inter  = (pred * target).sum()
        dice   = (2 * inter + self.smooth) / \
                 (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

    def forward(self, pred, target):
        return self.bce(pred, target) + \
               self.dice_loss(pred, target)


def iou_score(pred_bin, target):
    """Mean IoU over batch."""
    inter = (pred_bin * target).sum((1,2,3))
    union = (pred_bin + target - pred_bin*target).sum((1,2,3))
    return ((inter + 1e-6) / (union + 1e-6)).mean().item()


def train_one_epoch(model, loader, optimizer,
                    loss_fn, device):
    model.train()
    total_loss, total_iou = 0.0, 0.0

    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss  = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()

        pred_bin = (preds > 0.5).float()
        total_loss += loss.item()
        total_iou  += iou_score(pred_bin, masks)

    n = len(loader)
    return total_loss/n, total_iou/n


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_iou = 0.0, 0.0

    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)
        preds = model(imgs)
        loss  = loss_fn(preds, masks)

        pred_bin = (preds > 0.5).float()
        total_loss += loss.item()
        total_iou  += iou_score(pred_bin, masks)

    n = len(loader)
    return total_loss/n, total_iou/n


def train(data_dir     = 'data/',
          output_dir   = 'checkpoints/',
          img_size     = (256, 768),
          epochs       = 80,
          batch_size   = 4,
          lr           = 1e-3):

    os.makedirs(output_dir, exist_ok=True)
    device = ('cuda' if torch.cuda.is_available()
              else 'mps'
              if torch.backends.mps.is_available()
              else 'cpu')
    print(f"Using device: {device}")

    trn_loader, val_loader = get_loaders(
        data_dir, img_size, batch_size=batch_size
    )
    print(f"Train: {len(trn_loader.dataset)} images")
    print(f"Val:   {len(val_loader.dataset)} images")

    model     = get_model(device)
    loss_fn   = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min',
                                   patience=10, factor=0.5)

    history    = {'trn_loss':[], 'val_loss':[],
                  'trn_iou':[],  'val_iou':[]}
    best_iou   = 0.0
    best_path  = os.path.join(output_dir, 'best_model.pth')

    print(f"\n{'Epoch':>5} {'TrnLoss':>9} "
          f"{'ValLoss':>9} {'TrnIoU':>8} {'ValIoU':>8}")
    print('─' * 50)

    for epoch in range(1, epochs+1):
        trn_loss, trn_iou = train_one_epoch(
            model, trn_loader, optimizer, loss_fn, device
        )
        val_loss, val_iou = validate(
            model, val_loader, loss_fn, device
        )
        scheduler.step(val_loss)

        history['trn_loss'].append(trn_loss)
        history['val_loss'].append(val_loss)
        history['trn_iou'].append(trn_iou)
        history['val_iou'].append(val_iou)

        print(f"{epoch:5d} {trn_loss:9.4f} "
              f"{val_loss:9.4f} {trn_iou:8.4f} "
              f"{val_iou:8.4f}")

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch':      epoch,
                'model_state':model.state_dict(),
                'val_iou':    val_iou,
                'img_size':   img_size,
            }, best_path)
            print(f"  ★ Saved best (IoU={val_iou:.4f})")

        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            torch.save({
                'epoch':      epoch,
                'model_state':model.state_dict(),
            }, os.path.join(output_dir,
                            f'checkpoint_ep{epoch}.pth'))

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['trn_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss (BCE+Dice)')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history['trn_iou'], label='Train')
    axes[1].plot(history['val_iou'], label='Val')
    axes[1].set_title('IoU Score')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                             'training_curves.png'),
                dpi=130)
    plt.show()
    print(f"\nBest model saved: {best_path}")
    print(f"Best val IoU: {best_iou:.4f}")


if __name__ == '__main__':
    train(
        data_dir   = 'data/',
        output_dir = 'checkpoints/',
        img_size   = (256, 768),
        epochs     = 80,
        batch_size = 4,
        lr         = 1e-3,
    )