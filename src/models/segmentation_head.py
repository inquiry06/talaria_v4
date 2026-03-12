"""
TALARIA Dual-Branch Segmentation Head (nnU-Net decoder style).

T-Branch: 종양(tumor) segmentation — deep_feat 기반
N-Branch: 림프절(lymph node) segmentation — shallow_feat 기반 (공간 detail 중요)

두 branch 모두 skip connection을 활용한 UNet-style decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class UpBlock(nn.Module):
    """Upsample + skip concat + DoubleConv (nnU-Net decoder block)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if size mismatch (odd spatial dims)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SegBranch(nn.Module):
    """
    Single segmentation branch (T or N).

    Decoder from deep_feat back to full resolution using skip connections.
    skips = [f0(32), f1(64), f2(128), f3(256), f4(320)]
    deep_feat = f4 (320 ch, 1/16 res)

    Decoder stages:
        f4(320) + f3(256) → 256
        256     + f2(128) → 128
        128     + f1(64)  → 64
        64      + f0(32)  → 32
        32 → Conv(1) → logit
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.up1 = UpBlock(320, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up3 = UpBlock(128, 64,  64)
        self.up4 = UpBlock(64,  32,  32)
        self.head = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, deep_feat: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            deep_feat: (B, 320, D/16, H/16, W/16)
            skips:     [f0, f1, f2, f3, f4]
        Returns:
            logit: (B, num_classes, D, H, W)
        """
        f0, f1, f2, f3, _ = skips
        x = self.up1(deep_feat, f3)
        x = self.up2(x, f2)
        x = self.up3(x, f1)
        x = self.up4(x, f0)
        return self.head(x)


class DualBranchSegHead(nn.Module):
    """
    Dual-branch segmentation head:
        T-Branch: tumor segmentation
        N-Branch: lymph node segmentation (FocalTverskyLoss 적용)

    두 branch는 구조는 같지만 weight를 공유하지 않는다.
    N-Branch는 shallow_feat에서 추가 spatial prior를 받는다.
    """

    def __init__(self):
        super().__init__()
        self.t_branch = SegBranch(num_classes=1)
        self.n_branch = SegBranch(num_classes=1)

        # N-Branch spatial prior: shallow_feat(128ch)에서 attention map 생성
        self.n_spatial_prior = nn.Sequential(
            nn.Conv3d(128, 64, 1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, shallow_feat: torch.Tensor, deep_feat: torch.Tensor,
                skips: List[torch.Tensor]):
        """
        Args:
            shallow_feat: (B, 128, D/4,  H/4,  W/4)
            deep_feat:    (B, 320, D/16, H/16, W/16)
            skips:        [f0, f1, f2, f3, f4]
        Returns:
            t_logit: (B, 1, D, H, W)
            n_logit: (B, 1, D, H, W)
        """
        t_logit = self.t_branch(deep_feat, skips)

        # N-branch: shallow spatial prior로 attention 보강
        spatial_prior = self.n_spatial_prior(shallow_feat)  # (B,1,D/4,H/4,W/4)
        spatial_prior = F.interpolate(
            spatial_prior, size=skips[0].shape[2:],
            mode='trilinear', align_corners=False
        )
        # skip f0에 spatial prior 곱해서 lymph node 영역 강조
        skips_n = list(skips)
        skips_n[0] = skips_n[0] * spatial_prior

        n_logit = self.n_branch(deep_feat, skips_n)

        return t_logit, n_logit


if __name__ == '__main__':
    from encoder import TALARIAEncoder
    enc  = TALARIAEncoder()
    head = DualBranchSegHead()

    vol = torch.randn(2, 1, 96, 96, 96)
    shallow, deep, skips = enc(vol)
    t_logit, n_logit = head(shallow, deep, skips)
    print(f"t_logit: {t_logit.shape}")
    print(f"n_logit: {n_logit.shape}")
