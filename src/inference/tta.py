"""
TALARIA Test-Time Augmentation (TTA).

Augmentations applied at inference time:
    - Identity (no transform)
    - Flip along D, H, W axes
    - 90-degree rotations in axial plane

Each augmentation produces a set of predictions; they are averaged
via soft voting in soft_voting.py.
"""

import torch
import torch.nn.functional as F
from typing import List, Callable, Tuple, Dict


# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------

def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def flip_d(x: torch.Tensor) -> torch.Tensor:
    return x.flip(2)


def flip_h(x: torch.Tensor) -> torch.Tensor:
    return x.flip(3)


def flip_w(x: torch.Tensor) -> torch.Tensor:
    return x.flip(4)


def flip_dh(x: torch.Tensor) -> torch.Tensor:
    return x.flip(2).flip(3)


def flip_dhw(x: torch.Tensor) -> torch.Tensor:
    return x.flip(2).flip(3).flip(4)


def rot90_hw(x: torch.Tensor) -> torch.Tensor:
    """90-degree rotation in the H-W (axial) plane."""
    return x.rot90(1, dims=[3, 4])


def rot180_hw(x: torch.Tensor) -> torch.Tensor:
    return x.rot90(2, dims=[3, 4])


def rot270_hw(x: torch.Tensor) -> torch.Tensor:
    return x.rot90(3, dims=[3, 4])


# Inverse functions (to de-augment predictions)
INVERSE = {
    identity:  identity,
    flip_d:    flip_d,
    flip_h:    flip_h,
    flip_w:    flip_w,
    flip_dh:   flip_dh,
    flip_dhw:  flip_dhw,
    rot90_hw:  rot270_hw,
    rot180_hw: rot180_hw,
    rot270_hw: rot90_hw,
}

# Default TTA set (6 flips + 3 rotations + identity = 10 transforms)
DEFAULT_TTA_TRANSFORMS = [
    identity,
    flip_d,
    flip_h,
    flip_w,
    flip_dh,
    flip_dhw,
    rot90_hw,
    rot180_hw,
    rot270_hw,
]


# ---------------------------------------------------------------------------
# TTA Predictor
# ---------------------------------------------------------------------------

class TTAPredictor:
    """
    Applies a set of test-time augmentations to a single patch,
    collects model outputs, and inverts augmentations on segmentation masks.

    Args:
        model:      TALARIANet in finetune mode
        transforms: list of augmentation functions to apply
        device:     torch device
    """

    def __init__(
        self,
        model: torch.nn.Module,
        transforms: List[Callable] = None,
        device: torch.device = None,
    ):
        self.model = model
        self.transforms = transforms or DEFAULT_TTA_TRANSFORMS
        self.device = device or torch.device('cpu')
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_patch(self, patch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run TTA on a single 3D patch.

        Args:
            patch: (1, 1, D, H, W) — single patch tensor
        Returns:
            dict with averaged predictions:
                't_seg':  (1, 1, D, H, W) averaged tumor segmentation probability
                'n_seg':  (1, 1, D, H, W) averaged lymph node segmentation probability
                't_cls':  (1, 4) averaged T-stage logits
                'n_cls':  (1, 2) averaged N-stage logits
        """
        patch = patch.to(self.device)
        t_seg_acc = torch.zeros_like(patch)
        n_seg_acc = torch.zeros_like(patch)
        t_cls_acc = None
        n_cls_acc = None

        for aug in self.transforms:
            aug_patch = aug(patch)
            out = self.model(aug_patch)

            # Invert spatial transforms on segmentation predictions
            inv = INVERSE[aug]
            t_seg_acc += inv(out['t_seg'].sigmoid())
            n_seg_acc += inv(out['n_seg'].sigmoid())

            if t_cls_acc is None:
                t_cls_acc = out['t_cls']
                n_cls_acc = out['n_cls']
            else:
                t_cls_acc = t_cls_acc + out['t_cls']
                n_cls_acc = n_cls_acc + out['n_cls']

        n = len(self.transforms)
        return {
            't_seg': t_seg_acc / n,
            'n_seg': n_seg_acc / n,
            't_cls': t_cls_acc / n,
            'n_cls': n_cls_acc / n,
        }

    @torch.no_grad()
    def predict_volume(
        self,
        patches: List[torch.Tensor],
        coords: List[Tuple[int, int, int]],
        volume_shape: Tuple[int, int, int],
        patch_size: int = 96,
    ) -> Dict[str, torch.Tensor]:
        """
        Run TTA over all patches of a volume and stitch results.

        Args:
            patches:      list of (1, 1, P, P, P) tensors
            coords:       list of (d, h, w) top-left coords
            volume_shape: (D, H, W) full volume shape
            patch_size:   P
        Returns:
            dict with full-resolution outputs
        """
        D, H, W = volume_shape
        P = patch_size
        t_seg_vol = torch.zeros(1, 1, D, H, W)
        n_seg_vol = torch.zeros(1, 1, D, H, W)
        weight_vol = torch.zeros(1, 1, D, H, W)
        t_cls_list = []
        n_cls_list = []

        for patch, (d, h, w) in zip(patches, coords):
            preds = self.predict_patch(patch.to(self.device).cpu())
            d_end = min(d + P, D)
            h_end = min(h + P, H)
            w_end = min(w + P, W)
            t_seg_vol[..., d:d_end, h:h_end, w:w_end] += preds['t_seg'][..., :d_end-d, :h_end-h, :w_end-w]
            n_seg_vol[..., d:d_end, h:h_end, w:w_end] += preds['n_seg'][..., :d_end-d, :h_end-h, :w_end-w]
            weight_vol[..., d:d_end, h:h_end, w:w_end] += 1.0
            t_cls_list.append(preds['t_cls'])
            n_cls_list.append(preds['n_cls'])

        weight_vol = weight_vol.clamp(min=1e-6)
        t_seg_vol /= weight_vol
        n_seg_vol /= weight_vol

        # Average classification logits across patches
        t_cls_avg = torch.stack(t_cls_list).mean(0)
        n_cls_avg = torch.stack(n_cls_list).mean(0)

        return {
            't_seg': t_seg_vol,
            'n_seg': n_seg_vol,
            't_cls': t_cls_avg,
            'n_cls': n_cls_avg,
        }
