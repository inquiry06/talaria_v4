"""
TALARIA Reconstruction Decoder for self-supervised pretraining.
Reconstructs masked 3D patches from encoder features (Phase 1).

Architecture:
    encoder tokens (B, N, E)
        → linear projection
        → reshape to 3D feature map
        → transposed conv upsampling
        → reconstructed volume (B, 1, D, H, W)
"""

import torch
import torch.nn as nn
from einops import rearrange


class ReconstructionDecoder(nn.Module):
    """
    Lightweight decoder that reconstructs the original 3D CT volume
    from patch-level encoder tokens.

    Args:
        embed_dim:  encoder output dimension
        patch_size: patch size used in PatchEmbed3D (must match encoder)
        in_channels: number of input channels of original volume (1 for CT)
        decoder_dim: intermediate decoder channel width
    """

    def __init__(
        self,
        embed_dim: int = 192,
        patch_size: int = 8,
        in_channels: int = 1,
        decoder_dim: int = 128,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels

        # Project encoder tokens to decoder space
        self.proj = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.norm = nn.LayerNorm(decoder_dim)

        # Upsample: 3 stages of 2x trilinear upsampling + Conv
        self.up_blocks = nn.ModuleList()
        ch = decoder_dim
        num_ups = int(torch.log2(torch.tensor(patch_size)).item())  # e.g., patch_size=8 -> 3
        for _ in range(num_ups):
            self.up_blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                nn.Conv3d(ch, ch // 2, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(ch // 2),
                nn.GELU(),
            ))
            ch = ch // 2

        # Final reconstruction head
        self.head = nn.Conv3d(ch, in_channels, kernel_size=1)

    def forward(self, tokens: torch.Tensor, grid: tuple):
        """
        Args:
            tokens: (B, N, embed_dim)  — encoder output tokens
            grid:   (D', H', W')       — spatial grid dimensions
        Returns:
            recon:  (B, in_channels, D, H, W)  — reconstructed volume
        """
        D_, H_, W_ = grid
        B = tokens.shape[0]

        x = self.proj(tokens)           # (B, N, decoder_dim)
        x = self.norm(x)

        # Reshape to 3D feature map
        x = rearrange(x, 'b (d h w) c -> b c d h w', d=D_, h=H_, w=W_)

        for up in self.up_blocks:
            x = up(x)

        recon = self.head(x)            # (B, in_channels, D, H, W)
        return recon


class MaskedReconstructionModel(nn.Module):
    """
    Full pre-training model: Encoder + mask + Decoder.
    A random subset of patch tokens is masked (zeroed) before the decoder
    to force the encoder to learn meaningful representations.
    """

    def __init__(self, encoder, decoder, mask_ratio: float = 0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio

    def _mask_tokens(self, tokens: torch.Tensor):
        """Randomly mask a fraction of tokens."""
        B, N, E = tokens.shape
        num_mask = int(N * self.mask_ratio)
        noise = torch.rand(B, N, device=tokens.device)
        ids_shuffle = noise.argsort(dim=1)
        mask = torch.zeros(B, N, device=tokens.device, dtype=torch.bool)
        mask.scatter_(1, ids_shuffle[:, :num_mask], True)
        tokens_masked = tokens.clone()
        tokens_masked[mask] = 0.0
        return tokens_masked, mask

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 1, D, H, W)
        Returns:
            recon:  (B, 1, D, H, W)
            mask:   (B, N) bool — True where token was masked
        """
        _, deep_feat, grid = self.encoder(x)
        deep_masked, mask = self._mask_tokens(deep_feat)
        recon = self.decoder(deep_masked, grid)
        return recon, mask


if __name__ == '__main__':
    from encoder import TALARIAEncoder
    enc = TALARIAEncoder(embed_dim=192, patch_size=8, depth=12)
    dec = ReconstructionDecoder(embed_dim=192, patch_size=8)
    model = MaskedReconstructionModel(enc, dec, mask_ratio=0.75)

    vol = torch.randn(2, 1, 96, 96, 96)
    recon, mask = model(vol)
    print(f"recon: {recon.shape}, mask: {mask.shape}")
