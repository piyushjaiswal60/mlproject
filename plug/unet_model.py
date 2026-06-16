# ============================================================
# FILE: unet_model.py
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two conv layers with BatchNorm and ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    U-Net for binary segmentation:
      Input:  (B, 1, H, W)  grayscale
      Output: (B, 1, H, W)  sigmoid probability map
              > 0.5 = plug region
    """
    def __init__(self, features=(32, 64, 128, 256)):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        in_ch = 1
        for f in features:
            self.encoders.append(DoubleConv(in_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = f

        # ── Bottleneck ────────────────────────────────────────
        self.bottleneck = DoubleConv(features[-1],
                                      features[-1] * 2)

        # ── Decoder ──────────────────────────────────────────
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        in_ch = features[-1] * 2
        for f in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, f, 2, stride=2)
            )
            self.decoders.append(DoubleConv(f * 2, f))
            in_ch = f

        # ── Output ───────────────────────────────────────────
        self.final = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        # Encode
        skip_conns = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skip_conns.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # Decode
        for up, dec, skip in zip(self.upconvs,
                                   self.decoders,
                                   reversed(skip_conns)):
            x    = up(x)
            # Handle odd input sizes
            if x.shape != skip.shape:
                x = F.interpolate(x,
                                  size=skip.shape[2:],
                                  mode='bilinear',
                                  align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return torch.sigmoid(self.final(x))


def get_model(device='cpu'):
    model = UNet(features=(32, 64, 128, 256))
    return model.to(device)