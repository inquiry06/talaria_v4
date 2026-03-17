"""
TALARIA Dual-Branch Segmentation Head (nnU-Net decoder style).

T-Branch: 종양(tumor) segmentation — deep_feat 기반
N-Branch: 림프절(lymph node) segmentation
          Attention Gates (Oktay et al., 2018)로 skip feature를 re-weight
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
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Additive Attention Gate (Oktay et al., 2018).

    gating signal g (coarser, decoder) + skip signal x (finer, encoder)
    -> attention coefficient alpha in [0,1] per spatial location
    -> attended skip = x * alpha

        q_g   = W_g * g      (1x1x1 conv, InstanceNorm)
        q_x   = W_x * x      (1x1x1 conv, stride=2, InstanceNorm)
        psi   = ReLU(q_g + q_x)
        alpha = Sigmoid(W_psi * psi)
        out   = x * upsample(alpha)

    Args:
        F_g:   gating signal channels (decoder, coarser)
        F_l:   skip feature channels  (encoder, finer)
        F_int: intermediate channels  (typically F_l // 2)
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, bias=False),
            nn.InstanceNorm3d(F_int, affine=True),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=2, bias=False),
            nn.InstanceNorm3d(F_int, affine=True),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, bias=False),
            nn.InstanceNorm3d(1, affine=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: (B, F_g, Dg, Hg, Wg)  gating signal (decoder, coarser)
            x: (B, F_l, Dx, Hx, Wx)  skip feature  (encoder, finer)
        Returns:
            (B, F_l, Dx, Hx, Wx)  x re-weighted by attention coefficient
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            x1 = F.interpolate(x1, size=g1.shape[2:], mode='trilinear', align_corners=False)
        alpha = self.psi(self.relu(g1 + x1))
        alpha = F.interpolate(alpha, size=x.shape[2:], mode='trilinear', align_corners=False)
        return x * alpha


class SegBranch(nn.Module):
    """Standard UNet-style segmentation branch (T-Branch)."""

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.up1  = UpBlock(320, 256, 256)
        self.up2  = UpBlock(256, 128, 128)
        self.up3  = UpBlock(128, 64,  64)
        self.up4  = UpBlock(64,  32,  32)
        self.head = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, deep_feat: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        f0, f1, f2, f3, _ = skips
        x = self.up1(deep_feat, f3)
        x = self.up2(x, f2)
        x = self.up3(x, f1)
        x = self.up4(x, f0)
        return self.head(x)


class DualBranchSegHead(nn.Module):
    """
    Dual-branch segmentation head:

    T-Branch: standard SegBranch (no attention gate)
    N-Branch: Attention Gate decoder (Oktay et al., 2018)
              각 upsampling 단계에서 AG가 encoder skip을 re-weight하여
              sub-10mm 림프절 후보 신호를 선택적으로 증폭

    AG 배치 (N-Branch):
        AG1: g=deep_feat(320, 1/16), x=f3(256, 1/8)
        AG2: g=x1(256, 1/8),         x=f2(128, 1/4)
        AG3: g=x2(128, 1/4),         x=f1(64,  1/2)
        AG4: g=x3(64,  1/2),         x=f0(32,  1/1)
    """

    def __init__(self):
        super().__init__()

        # T-Branch
        self.t_branch = SegBranch(num_classes=1)

        # N-Branch: Attention Gates + decoder
        self.n_ag1 = AttentionGate(F_g=320, F_l=256, F_int=128)
        self.n_ag2 = AttentionGate(F_g=256, F_l=128, F_int=64)
        self.n_ag3 = AttentionGate(F_g=128, F_l=64,  F_int=32)
        self.n_ag4 = AttentionGate(F_g=64,  F_l=32,  F_int=16)

        self.n_up1  = UpBlock(320, 256, 256)
        self.n_up2  = UpBlock(256, 128, 128)
        self.n_up3  = UpBlock(128, 64,  64)
        self.n_up4  = UpBlock(64,  32,  32)
        self.n_head = nn.Conv3d(32, 1, kernel_size=1)

    def _n_forward(self, deep_feat: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        f0, f1, f2, f3, _ = skips

        f3_att = self.n_ag1(g=deep_feat, x=f3)
        x1     = self.n_up1(deep_feat, f3_att)

        f2_att = self.n_ag2(g=x1, x=f2)
        x2     = self.n_up2(x1, f2_att)

        f1_att = self.n_ag3(g=x2, x=f1)
        x3     = self.n_up3(x2, f1_att)

        f0_att = self.n_ag4(g=x3, x=f0)
        x4     = self.n_up4(x3, f0_att)

        return self.n_head(x4)

    def forward(self, shallow_feat: torch.Tensor, deep_feat: torch.Tensor,
                skips: List[torch.Tensor]):
        """
        Args:
            shallow_feat: (B, 128, D/4, H/4, W/4)  -- AG로 대체되어 미사용
            deep_feat:    (B, 320, D/16, H/16, W/16)
            skips:        [f0(32), f1(64), f2(128), f3(256), f4(320)]
        Returns:
            t_logit: (B, 1, D, H, W)
            n_logit: (B, 1, D, H, W)
        """
        t_logit = self.t_branch(deep_feat, skips)
        n_logit = self._n_forward(deep_feat, skips)
        return t_logit, n_logit


if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from src.models.encoder import TALARIAEncoder

    enc  = TALARIAEncoder()
    head = DualBranchSegHead()
    vol  = torch.randn(2, 1, 96, 96, 96)
    shallow, deep, skips = enc(vol)
    t_logit, n_logit = head(shallow, deep, skips)
    print(f"t_logit: {t_logit.shape}")
    print(f"n_logit: {n_logit.shape}")
    ag_params = sum(p.numel() for m in [head.n_ag1, head.n_ag2, head.n_ag3, head.n_ag4]
                    for p in m.parameters())
    print(f"AG total params: {ag_params:,}")
