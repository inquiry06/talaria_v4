"""
TALARIA Encoder: nnU-Net backbone (TotalSegmentator pretrained).

TotalSegmentator는 nnU-Net 기반으로 117개 장기를 segmentation하도록
학습된 모델이다. 여기서는 그 encoder (nnU-Net PlainConvUNet의 encoder 부분)를
backbone으로 가져와서 T-branch / N-branch / classification head에 feature를 공급한다.

Flow:
    CT (B, 1, D, H, W)
    → nnU-Net encoder (TotalSegmentator pretrained, fine-tune)
    → multi-scale features:
        shallow_feat: low-level spatial features  (→ N-Branch)
        deep_feat:    high-level semantic features (→ T-Branch + Cls Head)
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ConvNormAct(nn.Module):
    """Conv3d → InstanceNorm3d → LeakyReLU (nnU-Net standard block)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConvBlock(nn.Module):
    """Two ConvNormAct blocks (standard nnU-Net encoder stage)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, stride=stride)
        self.conv2 = ConvNormAct(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class TALARIAEncoder(nn.Module):
    """
    nnU-Net-style encoder backbone initialized from TotalSegmentator weights.

    Architecture (matches TotalSegmentator's nnU-Net encoder):
        Stage 0: Conv(1→32)   stride=1  → f0  (full res)
        Stage 1: Conv(32→64)  stride=2  → f1  (1/2)
        Stage 2: Conv(64→128) stride=2  → f2  (1/4)  ← shallow_feat → N-Branch
        Stage 3: Conv(128→256) stride=2 → f3  (1/8)
        Stage 4: Conv(256→320) stride=2 → f4  (1/16) ← deep_feat → T-Branch + Cls

    Returns:
        shallow_feat: (B, 128, D/4,  H/4,  W/4)
        deep_feat:    (B, 320, D/16, H/16, W/16)
        skips:        [f0, f1, f2, f3, f4]  for segmentation decoder
    """

    CHANNELS = [32, 64, 128, 256, 320]

    def __init__(
        self,
        in_channels: int = 1,
        channels: Optional[List[int]] = None,
    ):
        super().__init__()
        ch = channels or self.CHANNELS

        self.stage0 = DoubleConvBlock(in_channels, ch[0], stride=1)
        self.stage1 = DoubleConvBlock(ch[0], ch[1], stride=2)
        self.stage2 = DoubleConvBlock(ch[1], ch[2], stride=2)
        self.stage3 = DoubleConvBlock(ch[2], ch[3], stride=2)
        self.stage4 = DoubleConvBlock(ch[3], ch[4], stride=2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.InstanceNorm3d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        f0 = self.stage0(x)
        f1 = self.stage1(f0)
        f2 = self.stage2(f1)   # shallow_feat
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)   # deep_feat
        return f2, f4, [f0, f1, f2, f3, f4]

    def load_totalsegmentator_weights(self, strict: bool = False):
        """
        TotalSegmentator pretrained weight를 로드한다.
        실패해도 random init으로 계속 진행한다.
        """
        try:
            from totalsegmentator.libs import download_pretrained_weights
            from totalsegmentator.config import get_weights_dir
            import os, glob

            task_id = 291
            weights_dir = get_weights_dir()
            download_pretrained_weights(task_id)

            candidates = glob.glob(
                os.path.join(weights_dir, '**', 'fold_0', 'checkpoint_final.pth'),
                recursive=True
            )
            if not candidates:
                raise FileNotFoundError("TotalSegmentator checkpoint not found")

            ckpt_path = candidates[0]
            ckpt = torch.load(ckpt_path, map_location='cpu')
            state = ckpt.get('state_dict', ckpt)

            enc_state = {
                k.split('encoder.')[-1]: v
                for k, v in state.items() if 'encoder' in k
            }
            missing, unexpected = self.load_state_dict(enc_state, strict=strict)
            print(f"[TALARIAEncoder] TotalSegmentator weights loaded — "
                  f"missing: {len(missing)}, unexpected: {len(unexpected)}")

        except Exception as e:
            print(f"[TALARIAEncoder] Warning: {e}")
            print("  Continuing with random initialization.")


if __name__ == '__main__':
    enc = TALARIAEncoder(in_channels=1)
    vol = torch.randn(2, 1, 96, 96, 96)
    shallow, deep, skips = enc(vol)
    print(f"shallow: {shallow.shape}")
    print(f"deep:    {deep.shape}")
    print(f"skips:   {[list(s.shape) for s in skips]}")
