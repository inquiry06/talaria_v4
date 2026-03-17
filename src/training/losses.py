"""
TALARIA Loss Functions.

    - DiceLoss:              Differentiable Dice loss for segmentation
    - BCEDiceLoss:           Combined BCE + Dice for class-imbalanced masks
    - FocalLoss:             Focal loss for hard example mining
    - KnowledgeDistillLoss:  Soft-label KD + feature-level MSE for Phase 2
    - TALARIALoss:           Combined loss for Phase 3 (seg + cls)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    Works with logit inputs (applies sigmoid internally).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, 1, D, H, W) or (B, C, D, H, W)
            targets: (B, D, H, W) or (B, 1, D, H, W) binary mask
        """
        probs = torch.sigmoid(logits)
        if targets.dim() == logits.dim() - 1:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        intersection = (probs * targets).sum(dim=(2, 3, 4))
        denom = probs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# BCE + Dice Loss
# ---------------------------------------------------------------------------

class BCEDiceLoss(nn.Module):
    """Weighted combination of Binary Cross-Entropy and Dice Loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_w  = bce_weight
        self.dice_w = dice_weight
        self.bce    = nn.BCEWithLogitsLoss()
        self.dice   = DiceLoss(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == logits.dim() - 1:
            targets_bce = targets.unsqueeze(1).float()
        else:
            targets_bce = targets.float()
        bce_loss  = self.bce(logits, targets_bce)
        dice_loss = self.dice(logits, targets)
        return self.bce_w * bce_loss + self.dice_w * dice_loss


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance (small lymph nodes).
    Reference: Lin et al., RetinaNet (ICCV 2017).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if targets.dim() == logits.dim() - 1:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()


# ---------------------------------------------------------------------------
# Focal Tversky Loss
# ---------------------------------------------------------------------------

class TverskyLoss(nn.Module):
    """
    Tversky Loss: asymmetric generalisation of Dice Loss.

    By setting alpha < beta we penalise False Negatives more heavily than
    False Positives, which is critical for small, sparse structures like
    lymph nodes where missing a positive voxel is more costly than a false
    alarm.

        TI = (TP + smooth) / (TP + alpha·FP + beta·FN + smooth)
        Loss = 1 - TI

    Typical values for lymph node segmentation: alpha=0.3, beta=0.7
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if targets.dim() == logits.dim() - 1:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        tp = (probs * targets).sum(dim=(2, 3, 4))
        fp = (probs * (1 - targets)).sum(dim=(2, 3, 4))
        fn = ((1 - probs) * targets).sum(dim=(2, 3, 4))

        tversky_idx = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky_idx.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss: applies a focal modulation on top of Tversky Loss
    to concentrate learning on hard, ambiguous voxels.

        FTL = TverskyLoss(alpha, beta) ** gamma

    - gamma < 1 : down-weights easy negatives (standard use case)
    - alpha=0.3, beta=0.7 : heavy FN penalty for sparse lymph nodes
    - gamma=0.75 : mild focal sharpening (Abraham & Khan, 2019)

    Reference: Abraham & Khan, "A Novel Focal Tversky Loss Function with
               Improved Attention U-Net for Lesion Segmentation", ISBI 2019.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta)
        self.gamma   = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.tversky(logits, targets) ** self.gamma


# ---------------------------------------------------------------------------
# Knowledge Distillation Loss (Phase 2) — kept for reference, not used
# ---------------------------------------------------------------------------

class KnowledgeDistillLoss(nn.Module):
    """
    Teacher → Student Knowledge Distillation Loss.

    Components:
        - Soft KL divergence on classification logits (with temperature scaling)
        - MSE on encoder feature tokens (feature mimicking)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        kl_weight: float = 1.0,
        feat_weight: float = 0.5,
    ):
        super().__init__()
        self.T = temperature
        self.kl_w   = kl_weight
        self.feat_w = feat_weight

    def soft_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence on soft label distributions."""
        s_log  = F.log_softmax(student_logits / self.T, dim=-1)
        t_soft = F.softmax(teacher_logits  / self.T, dim=-1)
        return F.kl_div(s_log, t_soft, reduction='batchmean') * (self.T ** 2)

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        student_logits: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            student_feat:   (B, N, E_s) student encoder deep features
            teacher_feat:   (B, N, E_t) teacher encoder deep features
            student_logits: (B, C) optional student classification logits
            teacher_logits: (B, C) optional teacher classification logits
        """
        # Feature-level MSE
        # Align dimensions if student/teacher have different embed_dim
        if student_feat.shape[-1] != teacher_feat.shape[-1]:
            # Use avg pool to align
            sf = student_feat.mean(-1)
            tf = teacher_feat.mean(-1)
        else:
            sf = student_feat
            tf = teacher_feat
        feat_loss = F.mse_loss(sf, tf.detach())

        total = self.feat_w * feat_loss

        if student_logits is not None and teacher_logits is not None:
            kl_loss = self.soft_kl(student_logits, teacher_logits)
            total = total + self.kl_w * kl_loss

        return total


# ---------------------------------------------------------------------------
# Multi-class Focal Loss (Classification용)
# ---------------------------------------------------------------------------

class MultiClassFocalLoss(nn.Module):
    """
    α-balanced Focal Loss for multi-class classification.
    (Lin et al., 2017 — softmax 기반, N-stage binary classification에 사용)

    FL(p_t) = -α_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha:      per-class weight tensor, shape (C,). None이면 uniform.
                    N-stage의 경우 N0:N1 imbalance 보정용으로 [1.0, 6.5] 권장.
        gamma:      focusing parameter. 0이면 standard cross-entropy.
                    N-stage binary의 경우 gamma=2.0 (Lin et al. 기본값)
        ignore_index: 해당 label은 loss 계산에서 제외
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = -1,
    ):
        super().__init__()
        if alpha is not None:
            self.register_buffer('alpha', alpha.float())
        else:
            self.alpha = None
        self.gamma        = gamma
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) raw logits
            targets: (B,)   integer class labels
        Returns:
            scalar focal loss
        """
        if self.ignore_index >= 0:
            valid = targets != self.ignore_index
            logits  = logits[valid]
            targets = targets[valid]
        if targets.numel() == 0:
            return logits.sum() * 0.0

        log_p = F.log_softmax(logits, dim=-1)          # (B, C)
        p     = log_p.exp()                             # (B, C)

        # gather p_t and log_p_t for the true class
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)   # (B,)
        p_t     = p.gather(1, targets.unsqueeze(1)).squeeze(1)        # (B,)

        focal_weight = (1.0 - p_t) ** self.gamma                      # (B,)

        if self.alpha is not None:
            alpha_t = self.alpha[targets]                              # (B,)
            focal_weight = alpha_t * focal_weight

        loss = -(focal_weight * log_p_t)
        return loss.mean()

    def forward_soft(
        self,
        logits: torch.Tensor,
        soft_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Manifold Mixup용 soft-target focal loss.
        soft_targets: (B, C) one-hot mixture

        focal weight는 argmax(soft_targets)의 p_t 기준으로 계산.
        """
        log_p = F.log_softmax(logits, dim=-1)
        p     = log_p.exp()

        pseudo_hard = soft_targets.argmax(dim=-1)              # (B,)
        p_t = p.gather(1, pseudo_hard.unsqueeze(1)).squeeze(1) # (B,)
        focal_weight = (1.0 - p_t) ** self.gamma               # (B,)

        if self.alpha is not None:
            alpha_t = self.alpha[pseudo_hard]
            focal_weight = alpha_t * focal_weight

        ce = -(soft_targets * log_p).sum(dim=-1)               # (B,)
        return (focal_weight * ce).mean()


# ---------------------------------------------------------------------------
# TALARIA Combined Loss (Phase 3)
# ---------------------------------------------------------------------------

class TALARIALoss(nn.Module):
    """
    Combined training loss for Phase 3 fine-tuning.

    Components:
        - T-Branch: BCEDice for tumor segmentation
        - N-Branch: FocalTverskyLoss (alpha=0.3, beta=0.7, gamma=0.75)
        - T-Stage:  CrossEntropyLoss (class-weighted, morphological feature 기반)
        - N-Stage:  α-balanced MultiClassFocalLoss (gamma=2.0, alpha=[1.0, 6.5])
                    N0/N1 class imbalance 대응 + hard negative mining
    """

    def __init__(
        self,
        t_seg_weight: float = 1.0,
        n_seg_weight: float = 2.0,
        t_cls_weight: float = 0.5,
        n_cls_weight: float = 0.5,
    ):
        super().__init__()
        self.t_seg_w = t_seg_weight
        self.n_seg_w = n_seg_weight
        self.t_cls_w = t_cls_weight
        self.n_cls_w = n_cls_weight

        self.t_seg_loss = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        self.n_seg_loss = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)

        # T-Stage: class-weighted CE (morphological feature MLP)
        self.register_buffer('t_cls_weight', torch.tensor([1.0, 1.3, 0.8, 0.7], dtype=torch.float32))
        self.t_cls_loss = nn.CrossEntropyLoss(weight=self.t_cls_weight, ignore_index=-1)

        # N-Stage: α-balanced focal loss (N1 minority class 강조)
        self.register_buffer('n_focal_alpha', torch.tensor([1.0, 6.5], dtype=torch.float32))
        self.n_cls_loss = MultiClassFocalLoss(
            alpha=torch.tensor([1.0, 6.5]),
            gamma=2.0,
            ignore_index=-1,
        )

    def _soft_cross_entropy(
        self,
        logits: torch.Tensor,
        soft_targets: torch.Tensor,
        class_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Soft-target cross entropy with optional class weighting."""
        log_probs = F.log_softmax(logits, dim=-1)

        if class_weight is None:
            return -(soft_targets * log_probs).sum(dim=-1).mean()

        weighted_targets = soft_targets * class_weight.unsqueeze(0)
        normalizer = weighted_targets.sum(dim=-1).clamp_min(1e-12)
        return (-(weighted_targets * log_probs).sum(dim=-1) / normalizer).mean()

    def _mixup_soft_targets(
        self,
        hard_targets: torch.Tensor,
        num_classes: int,
        lam: float,
        perm: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        one_hot = F.one_hot(hard_targets, num_classes=num_classes).to(dtype=dtype)
        return lam * one_hot + (1.0 - lam) * one_hot[perm]

    def forward(
        self,
        t_seg_logit: torch.Tensor,
        n_seg_logit: torch.Tensor,
        t_cls_logit: torch.Tensor,
        n_cls_logit: torch.Tensor,
        t_seg_gt: Optional[torch.Tensor] = None,
        n_seg_gt: Optional[torch.Tensor] = None,
        t_stage_gt: Optional[torch.Tensor] = None,
        n_stage_gt: Optional[torch.Tensor] = None,
        mixup_lam: Optional[float] = None,
        mixup_perm: Optional[torch.Tensor] = None,
        t_stage_soft: Optional[torch.Tensor] = None,
        n_stage_soft: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        losses = {}
        total  = torch.tensor(0.0, device=t_seg_logit.device)

        if t_seg_gt is not None:
            l = self.t_seg_loss(t_seg_logit, t_seg_gt)
            losses['t_seg'] = l.item()
            total = total + self.t_seg_w * l

        if n_seg_gt is not None:
            l = self.n_seg_loss(n_seg_logit, n_seg_gt)
            losses['n_seg'] = l.item()
            total = total + self.n_seg_w * l

        if t_stage_gt is not None:
            if t_stage_soft is not None:
                l = self._soft_cross_entropy(t_cls_logit, t_stage_soft.to(dtype=t_cls_logit.dtype), self.t_cls_weight)
            elif mixup_lam is not None and mixup_perm is not None:
                t_soft = self._mixup_soft_targets(
                    hard_targets=t_stage_gt,
                    num_classes=t_cls_logit.shape[-1],
                    lam=mixup_lam,
                    perm=mixup_perm,
                    dtype=t_cls_logit.dtype,
                )
                l = self._soft_cross_entropy(t_cls_logit, t_soft, self.t_cls_weight)
            else:
                l = self.t_cls_loss(t_cls_logit, t_stage_gt)
            losses['t_cls'] = l.item()
            total = total + self.t_cls_w * l

        if n_stage_gt is not None:
            if n_stage_soft is not None:
                l = self.n_cls_loss.forward_soft(
                    n_cls_logit, n_stage_soft.to(dtype=n_cls_logit.dtype))
            elif mixup_lam is not None and mixup_perm is not None:
                n_soft = self._mixup_soft_targets(
                    hard_targets=n_stage_gt,
                    num_classes=n_cls_logit.shape[-1],
                    lam=mixup_lam,
                    perm=mixup_perm,
                    dtype=n_cls_logit.dtype,
                )
                l = self.n_cls_loss.forward_soft(n_cls_logit, n_soft)
            else:
                l = self.n_cls_loss(n_cls_logit, n_stage_gt)
            losses['n_cls'] = l.item()
            total = total + self.n_cls_w * l

        losses['total'] = total.item()
        return total, losses


