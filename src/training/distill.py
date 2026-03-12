"""
Phase 2: Knowledge Distillation — DINOv2-style Self-Distillation.

Teacher: Phase 1 pretrained Mamba encoder, updated via EMA of Student each step.
Student: Lightweight Mamba encoder (half depth + half embed_dim).

Key differences from naive feature mimicking:
    1. EMA Teacher   — Teacher is NOT frozen; it tracks Student via exponential
                       moving average, so both improve together during training.
    2. Multi-view    — Same CT patch augmented two different ways; Student sees
                       view1, Teacher sees view2. Loss enforces same representation
                       regardless of augmentation (domain-invariant features).
    3. Cosine loss   — Direction alignment instead of MSE, more robust to scale.

Usage:
    python -m src.training.distill --config configs/distill.yaml \\
        --teacher_ckpt experiments/pretrain_<ts>/checkpoints/best.ckpt
"""

import os
import copy
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from tqdm import tqdm

from src.models.encoder import TALARIAEncoder
from src.data.dataset import build_pretrain_dataset


# ---------------------------------------------------------------------------
# CT-specific multi-view augmentation
# ---------------------------------------------------------------------------

class CTTwoViewAugment:
    """
    Generate two differently augmented views of the same CT patch.

    Augmentations applied independently to each view:
        - Random intensity shift / scale (simulate scanner variability)
        - Random Gaussian noise
        - Random flip (axial / sagittal / coronal)
        - Random 90-degree rotation along one axis
    """

    def __init__(
        self,
        intensity_shift: float = 0.1,
        intensity_scale: float = 0.1,
        noise_std:       float = 0.05,
        flip_prob:       float = 0.5,
        rot_prob:        float = 0.5,
    ):
        self.intensity_shift = intensity_shift
        self.intensity_scale = intensity_scale
        self.noise_std       = noise_std
        self.flip_prob       = flip_prob
        self.rot_prob        = rot_prob

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation to a single (B, 1, D, H, W) tensor."""
        # Intensity jitter — simulates different CT scanners / protocols
        shift = torch.empty(1).uniform_(-self.intensity_shift, self.intensity_shift).item()
        scale = 1.0 + torch.empty(1).uniform_(-self.intensity_scale, self.intensity_scale).item()
        x = x * scale + shift

        # Gaussian noise
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        # Random flip along D / H / W
        for dim in [2, 3, 4]:
            if torch.rand(1).item() < self.flip_prob:
                x = torch.flip(x, dims=[dim])

        # Random 90-degree rotation in one plane
        if torch.rand(1).item() < self.rot_prob:
            k    = torch.randint(1, 4, (1,)).item()
            dims = [(2, 3), (2, 4), (3, 4)][torch.randint(0, 3, (1,)).item()]
            x = torch.rot90(x, k=k, dims=list(dims))

        return x.clamp(0.0, 1.0)

    def __call__(self, x: torch.Tensor):
        """
        Args:
            x: (B, 1, D, H, W)
        Returns:
            view1, view2: two independently augmented tensors
        """
        return self._augment(x), self._augment(x)


# ---------------------------------------------------------------------------
# EMA Teacher
# ---------------------------------------------------------------------------

class EMATeacher:
    """
    Exponential Moving Average wrapper for the Teacher encoder.

    Teacher params are updated each step:
        theta_teacher = momentum * theta_teacher + (1 - momentum) * theta_student

    This means Teacher is a smoothed, lagged version of Student — more stable
    than Student itself, producing better training targets.
    """

    def __init__(self, student: nn.Module, momentum: float = 0.996):
        self.teacher  = copy.deepcopy(student)
        self.momentum = momentum
        # Teacher never needs gradients
        for p in self.teacher.parameters():
            p.requires_grad = False

    def update(self, student: nn.Module):
        """Call once per training step after optimizer.step()."""
        with torch.no_grad():
            for t_p, s_p in zip(self.teacher.parameters(), student.parameters()):
                t_p.data = self.momentum * t_p.data + (1.0 - self.momentum) * s_p.data

    def to(self, device):
        self.teacher = self.teacher.to(device)
        return self

    def eval(self):
        self.teacher.eval()
        return self


# ---------------------------------------------------------------------------
# DINOv2-style Loss
# ---------------------------------------------------------------------------

class DINODistillLoss(nn.Module):
    """
    Cosine similarity loss between student and teacher feature representations.

    Why cosine instead of MSE:
        - Invariant to feature magnitude (more robust across layers / scales)
        - Focuses on directional alignment — what matters for downstream tasks
        - Standard in DINO / DINOv2 / MoCo v3 style distillation
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        s_feat: torch.Tensor,
        t_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s_feat: (B, N, D) student features
            t_feat: (B, N, D) teacher features (no_grad from caller)
        Returns:
            scalar loss
        """
        # L2 normalize along feature dim
        s_norm = F.normalize(s_feat, dim=-1)
        t_norm = F.normalize(t_feat, dim=-1)

        # Cosine similarity → loss = 1 - similarity (want similarity → 1)
        cos_sim = (s_norm * t_norm).sum(dim=-1)   # (B, N)
        loss    = (1.0 - cos_sim).mean()
        return loss


# ---------------------------------------------------------------------------
# Student Encoder
# ---------------------------------------------------------------------------

class StudentEncoder(TALARIAEncoder):
    """
    Lightweight student: half depth + half embed_dim of teacher by default.
    Inherits TALARIAEncoder; architecture controlled entirely by config.
    """
    pass


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------

def train_one_epoch(
    student,
    ema_teacher,
    augment,
    loss_fn,
    loader,
    optimizer,
    device,
    epoch,
    scaler=None,
):
    student.train()
    ema_teacher.eval()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}", leave=False)
    for batch in pbar:
        images = batch['image'].to(device)   # (B, 1, D, H, W)

        # Two independently augmented views of the same CT patch
        view1, view2 = augment(images)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # Student processes view1
            _, s_deep, _ = student(view1)

            # Teacher processes view2 — no gradient through teacher
            with torch.no_grad():
                _, t_deep, _ = ema_teacher.teacher(view2)

            loss = loss_fn(s_deep, t_deep)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

        # EMA update: Teacher tracks Student
        ema_teacher.update(student)

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(student, ema_teacher, augment, loss_fn, loader, device):
    student.eval()
    ema_teacher.eval()
    total_loss = 0.0

    pbar = tqdm(loader, desc="[Val]  ", leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        view1, view2 = augment(images)
        _, s_deep, _ = student(view1)
        _, t_deep, _ = ema_teacher.teacher(view2)
        loss = loss_fn(s_deep, t_deep)
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: dict, teacher_ckpt: str):
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir   = os.path.join('experiments', f"distill_{timestamp}")
    ckpt_dir  = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    teacher_cfg = config['teacher']
    student_cfg = config['student']

    # --- Student (to be trained) ---
    student = StudentEncoder(
        embed_dim=student_cfg['embed_dim'],
        patch_size=student_cfg['patch_size'],
        depth=student_cfg['depth'],
        d_state=student_cfg.get('d_state', 16),
        expand=student_cfg.get('expand', 2),
        drop=student_cfg.get('drop', 0.1),
    ).to(device)

    # --- EMA Teacher: initialized from Phase 1 pretrained encoder ---
    # We first build a teacher-sized encoder, load Phase 1 weights,
    # then initialize EMA with student architecture.
    # EMA Teacher starts as a copy of student but immediately begins
    # receiving Phase 1 knowledge via the EMA momentum update.
    #
    # Alternative: initialize EMA teacher directly from Phase 1 weights
    # by building a teacher-sized encoder. Kept flexible via config flag.
    use_phase1_teacher = config.get('init_teacher_from_phase1', True)

    if use_phase1_teacher:
        # Build full-size teacher encoder, load Phase 1 weights
        pretrained_teacher = TALARIAEncoder(
            embed_dim=teacher_cfg['embed_dim'],
            patch_size=teacher_cfg['patch_size'],
            depth=teacher_cfg['depth'],
            d_state=teacher_cfg.get('d_state', 16),
            expand=teacher_cfg.get('expand', 2),
        )
        ckpt      = torch.load(teacher_ckpt, map_location='cpu')
        state     = ckpt.get('model_state_dict', ckpt)
        enc_state = {k.replace('encoder.', ''): v
                     for k, v in state.items() if k.startswith('encoder.')}
        pretrained_teacher.load_state_dict(enc_state, strict=False)
        pretrained_teacher = pretrained_teacher.to(device)

        # EMA Teacher is a copy of the pretrained teacher
        # (same arch as teacher_cfg — larger than student)
        ema_teacher = EMATeacher(
            pretrained_teacher,
            momentum=config.get('ema_momentum', 0.996),
        ).to(device)
        print(f"[Phase 2] EMA Teacher initialized from Phase 1 ckpt: {teacher_ckpt}")
    else:
        # EMA Teacher starts as a copy of student (simpler, no size mismatch)
        ema_teacher = EMATeacher(
            student,
            momentum=config.get('ema_momentum', 0.996),
        ).to(device)
        print("[Phase 2] EMA Teacher initialized from Student weights.")

    # --- Augmentation ---
    augment = CTTwoViewAugment(
        intensity_shift=config.get('aug_intensity_shift', 0.1),
        intensity_scale=config.get('aug_intensity_scale', 0.1),
        noise_std=config.get('aug_noise_std', 0.05),
        flip_prob=config.get('aug_flip_prob', 0.5),
        rot_prob=config.get('aug_rot_prob', 0.5),
    )

    # --- Loss ---
    loss_fn = DINODistillLoss(temperature=config.get('temperature', 0.1))

    # --- Data (90/10 split) ---
    full_ds = build_pretrain_dataset(
        lits_root=config.get('lits_root'),
        tcia_root=config.get('tcia_root'),
        amos_root=config.get('amos_root'),
        patch_size=teacher_cfg['patch_size'],
        stride=config.get('stride', 48),
    )
    val_size   = max(1, int(len(full_ds) * 0.1))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    loader_kwargs = dict(
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **loader_kwargs)

    # --- Optimizer (student only — teacher updated via EMA, not optimizer) ---
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.get('lr', 5e-4),
        weight_decay=config.get('weight_decay', 0.05),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get('epochs', 50), eta_min=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs = config.get('epochs', 50)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            student, ema_teacher, augment, loss_fn,
            train_loader, optimizer, device, epoch, scaler,
        )
        val_loss = validate(
            student, ema_teacher, augment, loss_fn, val_loader, device,
        )
        scheduler.step()

        print(f"[Epoch {epoch:03d}/{epochs}] "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Save student checkpoint (what gets used in Phase 3)
        ckpt_data = {
            'epoch':              epoch,
            'model_state_dict':   student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss':         train_loss,
            'val_loss':           val_loss,
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_data, os.path.join(ckpt_dir, 'best.ckpt'))
            print(f"  -> best val_loss updated: {best_val_loss:.4f}")

        if epoch % config.get('save_every', 10) == 0:
            torch.save(ckpt_data, os.path.join(ckpt_dir, f'epoch_{epoch:04d}.ckpt'))

    print(f"\n[Phase 2] DINOv2-style distillation complete.")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Student ckpt  : {ckpt_dir}/best.ckpt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',       type=str, default='configs/distill.yaml')
    parser.add_argument('--teacher_ckpt', type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    main(config, args.teacher_ckpt)
