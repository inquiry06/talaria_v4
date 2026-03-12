"""
Phase 1: Self-Supervised Pre-training via Masked Volume Reconstruction.

Usage:
    python -m src.training.pretrain --config configs/pretrain.yaml
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from tqdm import tqdm

from src.models.encoder import TALARIAEncoder
from src.models.decoder import ReconstructionDecoder, MaskedReconstructionModel
from src.data.dataset import build_pretrain_dataset


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_recon_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int = 8,
) -> torch.Tensor:
    """
    MSE loss computed only on masked patch tokens.

    Args:
        recon:      (B, 1, D, H, W) reconstructed volume
        target:     (B, 1, D, H, W) original volume
        mask:       (B, N) bool — True for masked tokens
        patch_size: token patch size used in encoder
    """
    B, C, D, H, W = target.shape
    P = patch_size

    recon_patches  = recon.unfold(2, P, P).unfold(3, P, P).unfold(4, P, P)
    target_patches = target.unfold(2, P, P).unfold(3, P, P).unfold(4, P, P)

    recon_flat  = recon_patches.contiguous().view(B, -1, P ** 3)
    target_flat = target_patches.contiguous().view(B, -1, P ** 3)

    N = mask.shape[1]
    recon_flat  = recon_flat[:, :N]
    target_flat = target_flat[:, :N]

    mask_expanded = mask.unsqueeze(-1).expand_as(recon_flat)
    loss = ((recon_flat[mask_expanded] - target_flat[mask_expanded]) ** 2).mean()
    return loss


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, patch_size, epoch, scaler=None):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}", leave=False)
    for step, batch in enumerate(pbar):
        images = batch['image'].to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            recon, mask = model(images)
            loss = masked_recon_loss(recon, images, mask, patch_size)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, device, patch_size):
    model.eval()
    total_loss = 0.0

    pbar = tqdm(loader, desc="[Val]  ", leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        recon, mask = model(images)
        loss = masked_recon_loss(recon, images, mask, patch_size)
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir  = os.path.join('experiments', f"pretrain_{timestamp}")
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # --- Model ---
    token_patch_size = config.get('token_patch_size', config.get('patch_size', 8))
    encoder = TALARIAEncoder(
        in_channels=1,
        patch_size=token_patch_size,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        d_state=config.get('d_state', 16),
        expand=config.get('expand', 2),
        drop=config.get('drop', 0.1),
    )
    decoder = ReconstructionDecoder(
        embed_dim=config['embed_dim'],
        patch_size=token_patch_size,
        in_channels=1,
        decoder_dim=config.get('decoder_dim', 128),
    )
    model = MaskedReconstructionModel(
        encoder, decoder, mask_ratio=config.get('mask_ratio', 0.75)
    ).to(device)

    # --- Data (90/10 train-val split) ---
    full_dataset = build_pretrain_dataset(
        lits_root=config.get('lits_root'),
        tcia_root=config.get('tcia_root'),
        amos_root=config.get('amos_root'),
        patch_size=config.get('volume_patch_size', config.get('patch_size_infer', 96)),
        stride=config.get('stride', 48),
    )
    val_size   = max(1, int(len(full_dataset) * 0.1))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    loader_kwargs = dict(
        batch_size=config.get('batch_size', 4),
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **loader_kwargs)

    # --- Optimizer + Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('weight_decay', 0.05),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get('epochs', 100), eta_min=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs = config.get('epochs', 100)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, token_patch_size, epoch, scaler
        )
        val_loss = validate(model, val_loader, device, token_patch_size)
        scheduler.step()

        print(f"[Epoch {epoch:03d}/{epochs}] "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        # Save best by val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(ckpt_dir, 'best.ckpt'))
            print(f"  → best val_loss updated: {best_val_loss:.4f}")

        if epoch % config.get('save_every', 10) == 0:
            torch.save(ckpt, os.path.join(ckpt_dir, f'epoch_{epoch:04d}.ckpt'))

    print(f"\n[Phase 1] Pre-training complete.")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Checkpoint    : {ckpt_dir}/best.ckpt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/pretrain.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    main(config)
