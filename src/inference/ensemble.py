"""
TALARIA Dual-Size Classification Ensemble for Inference.

Replaces TTA soft-voting with a two-model ensemble that uses different
input patch sizes to capture complementary spatial contexts:

    Model A (patch_size=96 ):  standard field-of-view — good for tumor extent
    Model B (patch_size=128):  wider field-of-view   — better lymph node context

Each model is trained independently on its own patch size (see finetune.py).
At inference, segmentation maps are resized to a common resolution and averaged;
classification logits are soft-voted (probability averaging).

Why patch-size ensemble instead of TTA?
    - TTA augments the *same* view differently → variance from geometry
    - Patch-size ensemble captures *different* anatomical context → richer diversity
    - Two independent models are less correlated → stronger ensemble signal
    - Avoids 9× inference overhead of full TTA

Usage (CLI):
    python -m src.inference.ensemble \\
        --config   configs/finetune.yaml \\
        --ckpt_a   experiments/finetune_96/checkpoints/best.ckpt \\
        --ckpt_b   experiments/finetune_128/checkpoints/best.ckpt \\
        --input    /path/to/ct_scan.nii.gz \\
        --output   /path/to/output/

Usage (API):
    from src.inference.ensemble import DualSizeEnsemble
    ensemble = DualSizeEnsemble(config, ckpt_a, ckpt_b)
    report   = ensemble.predict(nifti_path, output_dir)
"""

import os
import argparse
import json
import yaml
import torch
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from src.models.talaria import build_talaria, TALARIANet
from src.data.preprocessing import preprocess_ct


# ---------------------------------------------------------------------------
# Single-model sliding-window inference (no TTA)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_single_model(
    model: TALARIANet,
    nifti_path: str,
    patch_size: int,
    stride: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Sliding-window inference on one model without TTA.

    Returns dict with keys:
        't_seg_prob' : (D, H, W) float32 — tumor probability map
        'n_seg_prob' : (D, H, W) float32 — lymph node probability map
        't_probs'    : (4,) float32       — T-stage class probabilities
        'n_probs'    : (2,) float32       — N-stage class probabilities
        'vol_shape'  : (D, H, W) tuple
    """
    model.eval().to(device)

    patches, coords, vol_shape = preprocess_ct(nifti_path, patch_size, stride)
    D, H, W = vol_shape

    t_seg_acc  = np.zeros((D, H, W), dtype=np.float32)
    n_seg_acc  = np.zeros((D, H, W), dtype=np.float32)
    weight_acc = np.zeros((D, H, W), dtype=np.float32)
    t_cls_list: List[np.ndarray] = []
    n_cls_list: List[np.ndarray] = []

    for patch, (d, h, w) in zip(patches, coords):
        x = torch.from_numpy(patch).unsqueeze(0).to(device)  # (1, C, P, P, P)
        out = model(x)

        t_prob = out['t_seg'].sigmoid().squeeze().cpu().numpy()  # (P, P, P)
        n_prob = out['n_seg'].sigmoid().squeeze().cpu().numpy()
        t_cls  = torch.softmax(out['t_cls'].squeeze(), dim=-1).cpu().numpy()  # (4,)
        n_cls  = torch.softmax(out['n_cls'].squeeze(), dim=-1).cpu().numpy()  # (2,)

        P = patch_size
        d_end = min(d + P, D)
        h_end = min(h + P, H)
        w_end = min(w + P, W)
        sd, sh, sw = d_end - d, h_end - h, w_end - w

        t_seg_acc [d:d_end, h:h_end, w:w_end] += t_prob[:sd, :sh, :sw]
        n_seg_acc [d:d_end, h:h_end, w:w_end] += n_prob[:sd, :sh, :sw]
        weight_acc[d:d_end, h:h_end, w:w_end] += 1.0
        t_cls_list.append(t_cls)
        n_cls_list.append(n_cls)

    weight_acc = np.clip(weight_acc, a_min=1e-6, a_max=None)
    t_seg_prob = t_seg_acc / weight_acc
    n_seg_prob = n_seg_acc / weight_acc
    t_probs    = np.stack(t_cls_list).mean(0)  # average across patches
    n_probs    = np.stack(n_cls_list).mean(0)

    return {
        't_seg_prob': t_seg_prob,
        'n_seg_prob': n_seg_prob,
        't_probs':    t_probs,
        'n_probs':    n_probs,
        'vol_shape':  vol_shape,
    }


# ---------------------------------------------------------------------------
# Dual-Size Ensemble
# ---------------------------------------------------------------------------

class DualSizeEnsemble:
    """
    Ensemble of two TALARIANet models trained on different patch sizes.

    Model A uses patch_size_a (e.g. 96)  — standard receptive field.
    Model B uses patch_size_b (e.g. 128) — wider receptive field.

    Segmentation probability maps from both models are aligned to the same
    volume shape (model A's native resolution) via trilinear interpolation,
    then averaged with configurable weights.

    Classification probabilities are averaged directly (soft voting).

    Args:
        config:       base TALARIANet config dict (shared across both models)
        ckpt_a:       checkpoint path for model A
        ckpt_b:       checkpoint path for model B
        patch_size_a: patch size for model A (default 96)
        patch_size_b: patch size for model B (default 128)
        stride_a:     sliding-window stride for model A (default 48)
        stride_b:     sliding-window stride for model B (default 64)
        weight_a:     ensemble weight for model A (default 0.5)
        weight_b:     ensemble weight for model B (default 0.5)
        device:       torch device (auto-detected if None)
    """

    TSTAGE_LABELS = ['T1', 'T2', 'T3', 'T4']
    NSTAGE_LABELS = ['N0', 'N1']

    def __init__(
        self,
        config: dict,
        ckpt_a: str,
        ckpt_b: str,
        patch_size_a: int = 96,
        patch_size_b: int = 128,
        stride_a: int = 48,
        stride_b: int = 64,
        weight_a: float = 0.5,
        weight_b: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        self.patch_size_a = patch_size_a
        self.patch_size_b = patch_size_b
        self.stride_a     = stride_a
        self.stride_b     = stride_b
        self.weight_a     = weight_a
        self.weight_b     = weight_b
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[Ensemble] Loading Model A  (patch={patch_size_a}) from {ckpt_a}")
        cfg_a = {**config, 'patch_size': patch_size_a}
        self.model_a = self._load_model(cfg_a, ckpt_a)

        print(f"[Ensemble] Loading Model B  (patch={patch_size_b}) from {ckpt_b}")
        cfg_b = {**config, 'patch_size': patch_size_b}
        self.model_b = self._load_model(cfg_b, ckpt_b)

    def _load_model(self, config: dict, ckpt_path: str) -> TALARIANet:
        model = build_talaria({**config, 'mode': 'finetune'})
        ckpt  = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
        model.eval()
        return model

    def _align_seg(
        self,
        prob_map: np.ndarray,
        target_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Resize a segmentation probability map to target_shape via trilinear interpolation.
        Used to align model B's output to model A's native volume resolution.
        """
        if prob_map.shape == target_shape:
            return prob_map
        t = torch.from_numpy(prob_map).unsqueeze(0).unsqueeze(0).float()  # (1,1,D,H,W)
        t = F.interpolate(t, size=target_shape, mode='trilinear', align_corners=False)
        return t.squeeze().numpy()

    @torch.no_grad()
    def predict(
        self,
        nifti_path: str,
        output_dir: str,
        seg_threshold: float = 0.5,
    ) -> dict:
        """
        Run dual-size ensemble inference on a single CT NIfTI file.

        Args:
            nifti_path:    path to input CT (.nii.gz)
            output_dir:    directory to save segmentation masks + JSON report
            seg_threshold: probability threshold for binary masks

        Returns:
            report dict with T/N-stage predictions and per-class probabilities
        """
        os.makedirs(output_dir, exist_ok=True)

        # --- Run each model independently ---
        print("[Ensemble] Model A inference ...")
        res_a = _run_single_model(
            self.model_a, nifti_path, self.patch_size_a, self.stride_a, self.device
        )
        print("[Ensemble] Model B inference ...")
        res_b = _run_single_model(
            self.model_b, nifti_path, self.patch_size_b, self.stride_b, self.device
        )

        # --- Align model B seg maps to model A volume shape ---
        target_shape = res_a['vol_shape']
        t_seg_b_aligned = self._align_seg(res_b['t_seg_prob'], target_shape)
        n_seg_b_aligned = self._align_seg(res_b['n_seg_prob'], target_shape)

        # --- Weighted average ---
        wa, wb = self.weight_a, self.weight_b
        t_seg_avg = wa * res_a['t_seg_prob'] + wb * t_seg_b_aligned
        n_seg_avg = wa * res_a['n_seg_prob'] + wb * n_seg_b_aligned
        t_probs   = wa * res_a['t_probs']    + wb * res_b['t_probs']
        n_probs   = wa * res_a['n_probs']    + wb * res_b['n_probs']

        # Renormalise classification probs (they sum to 1 each; weighted avg may drift)
        t_probs = t_probs / t_probs.sum()
        n_probs = n_probs / n_probs.sum()

        # --- Binary masks ---
        t_mask = (t_seg_avg >= seg_threshold).astype(np.uint8)
        n_mask = (n_seg_avg >= seg_threshold).astype(np.uint8)
        t_stage = self.TSTAGE_LABELS[int(t_probs.argmax())]
        n_stage = self.NSTAGE_LABELS[int(n_probs.argmax())]

        # --- Save NIfTI outputs ---
        ref_img = sitk.ReadImage(nifti_path)

        def _save(arr: np.ndarray, fname: str, dtype=np.uint8):
            img = sitk.GetImageFromArray(arr.astype(dtype))
            img.CopyInformation(ref_img)
            sitk.WriteImage(img, os.path.join(output_dir, fname))

        _save(t_mask,   'tumor_mask.nii.gz',  np.uint8)
        _save(n_mask,   'lymph_mask.nii.gz',  np.uint8)
        _save(t_seg_avg,'tumor_prob.nii.gz',  np.float32)
        _save(n_seg_avg,'lymph_prob.nii.gz',  np.float32)

        # --- JSON report ---
        report = {
            'T_stage':  t_stage,
            'N_stage':  n_stage,
            'T_probs':  {f'T{i+1}': float(p) for i, p in enumerate(t_probs)},
            'N_probs':  {f'N{i}':   float(p) for i, p in enumerate(n_probs)},
            'ensemble': {
                'model_a_patch': self.patch_size_a,
                'model_b_patch': self.patch_size_b,
                'weight_a':      self.weight_a,
                'weight_b':      self.weight_b,
            },
        }
        with open(os.path.join(output_dir, 'tnm_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n=== TALARIA Dual-Size Ensemble Report ===")
        print(f"  T-Stage : {t_stage}  {report['T_probs']}")
        print(f"  N-Stage : {n_stage}  {report['N_probs']}")
        print(f"  Outputs : {output_dir}")
        return report


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TALARIA Dual-Size Ensemble Inference')
    parser.add_argument('--config',       type=str, required=True,
                        help='Model config YAML (shared base config)')
    parser.add_argument('--ckpt_a',       type=str, required=True,
                        help='Checkpoint for model A (small patch size)')
    parser.add_argument('--ckpt_b',       type=str, required=True,
                        help='Checkpoint for model B (large patch size)')
    parser.add_argument('--input',        type=str, required=True,
                        help='Input CT NIfTI file (.nii.gz)')
    parser.add_argument('--output',       type=str, required=True,
                        help='Output directory')
    parser.add_argument('--patch_size_a', type=int, default=96)
    parser.add_argument('--patch_size_b', type=int, default=128)
    parser.add_argument('--stride_a',     type=int, default=48)
    parser.add_argument('--stride_b',     type=int, default=64)
    parser.add_argument('--weight_a',     type=float, default=0.5)
    parser.add_argument('--weight_b',     type=float, default=0.5)
    parser.add_argument('--threshold',    type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ensemble = DualSizeEnsemble(
        config=config,
        ckpt_a=args.ckpt_a,
        ckpt_b=args.ckpt_b,
        patch_size_a=args.patch_size_a,
        patch_size_b=args.patch_size_b,
        stride_a=args.stride_a,
        stride_b=args.stride_b,
        weight_a=args.weight_a,
        weight_b=args.weight_b,
    )
    ensemble.predict(args.input, args.output, seg_threshold=args.threshold)
