# ============================================================
# FILE: dataset.py
# ============================================================
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PlugDataset(Dataset):
    """
    Folder structure expected:
      data/
        images/   frame_0000.png  frame_0010.png ...
        masks/    frame_0000.png  frame_0010.png ...
                  (masks: 0=background, 255=plug)
    """
    def __init__(self, img_dir, mask_dir,
                 img_size=(256, 768),
                 augment=True):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.img_size  = img_size  # (H, W)
        self.augment   = augment
        self.filenames = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith(('.png', '.jpg'))
        ])

        self.aug_pipe = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3, p=0.7),
            A.GaussNoise(var_limit=(5, 30), p=0.4),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=5,
                p=0.5),
        ])

        self.to_tensor = A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Load image (grayscale)
        img = cv2.imread(
            os.path.join(self.img_dir, fname),
            cv2.IMREAD_GRAYSCALE
        )
        # Load mask (grayscale, 0 or 255)
        mask = cv2.imread(
            os.path.join(self.mask_dir, fname),
            cv2.IMREAD_GRAYSCALE
        )
        mask = (mask > 127).astype(np.uint8) * 255

        # Augment
        if self.augment:
            aug = self.aug_pipe(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']

        # Resize + normalize + to tensor
        result = self.to_tensor(image=img, mask=mask)
        img_t  = result['image'].float()        # (1,H,W)
        mask_t = (result['mask'] > 127).float() # (H,W)
        mask_t = mask_t.unsqueeze(0)            # (1,H,W)

        return img_t, mask_t


def get_loaders(data_dir, img_size=(256, 768),
                val_split=0.15, batch_size=4):
    """Split data and return train/val loaders."""
    img_dir  = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')

    all_files = sorted(os.listdir(img_dir))
    n_val     = max(1, int(len(all_files) * val_split))
    val_files = all_files[-n_val:]
    trn_files = all_files[:-n_val]

    # Temporarily patch filenames
    trn_ds = PlugDataset(img_dir, mask_dir,
                          img_size, augment=True)
    val_ds = PlugDataset(img_dir, mask_dir,
                          img_size, augment=False)
    trn_ds.filenames = trn_files
    val_ds.filenames = val_files

    trn_loader = DataLoader(trn_ds, batch_size=batch_size,
                             shuffle=True,  num_workers=2,
                             pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                             shuffle=False, num_workers=2,
                             pin_memory=True)
    return trn_loader, val_loader