"""
TALARIA segmentation 시각화 스크립트
best.ckpt 로드 후 val set 환자에 대해 T-seg 예측 결과를 CT slice에 overlay하여 PNG로 저장
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.talaria import TALARIAModel
from src.data.dataset import HCCTACEDataset

def sliding_window_inference(model, ct, patch_size=96, overlap=0.5, device='cuda'):
    B, C, D, H, W = ct.shape
    stride = int(patch_size * (1 - overlap))
    pred_sum = torch.zeros(B, 1, D, H, W, device=device)
    count    = torch.zeros(B, 1, D, H, W, device=device)
    for d in range(0, max(1, D - patch_size + 1), stride):
        for h in range(0, max(1, H - patch_size + 1), stride):
            for w in range(0, max(1, W - patch_size + 1), stride):
                d2 = min(d + patch_size, D); h2 = min(h + patch_size, H); w2 = min(w + patch_size, W)
                d1 = d2 - patch_size; h1 = h2 - patch_size; w1 = w2 - patch_size
                patch = ct[:, :, d1:d2, h1:h2, w1:w2]
                with torch.no_grad():
                    out = model(patch)
                    seg = torch.sigmoid(out['t_seg'])
                pred_sum[:, :, d1:d2, h1:h2, w1:w2] += seg
                count[:, :, d1:d2, h1:h2, w1:w2]    += 1
    return pred_sum / count.clamp(min=1)


def visualize(
    ckpt_path: str = "experiments/finetune_p96/checkpoints/best.ckpt",
    out_dir: str = "experiments/finetune_p96/vis",
    num_patients: int = 3,
    num_slices: int = 5,
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[vis] device: {device}")

    # ckpt 로드
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    # 모델 초기화
    model = TALARIAModel(
        in_channels=cfg.get("in_channels", 1),
        t_classes=cfg.get("t_classes", 4),
        n_classes=cfg.get("n_classes", 2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[vis] model loaded from epoch {ckpt.get('epoch')}")

    # val dataset
    dataset = HCCTACEDataset(
        data_root=cfg.get("data_root", "data/HCC_TACE"),
        metadata_path=cfg.get("metadata_path", "data/hcc_tnm_labels.json"),
        split="val",
        patch_size=cfg.get("patch_size", 96),
    )
    print(f"[vis] val patients: {len(dataset)}")

    t_stage_names = {0: "T1", 1: "T2", 2: "T3", 3: "T4"}
    n_stage_names = {0: "N0", 1: "N1"}

    for idx in range(min(num_patients, len(dataset))):
        sample = dataset[idx]
        ct    = sample["image"].unsqueeze(0).to(device)   # (1,1,D,H,W)
        seg_gt = sample["seg_mask"]                        # (1,D,H,W)
        t_gt  = sample["tstage"].item()
        n_gt  = sample["nstage"].item()
        pid   = sample.get("patient_id", f"patient_{idx}")

        with torch.no_grad():
            outputs = model(ct)

        t_seg_pred = sliding_window_inference(model, ct, patch_size=96, device=device).squeeze().cpu().numpy()
        seg_gt_np = seg_gt.squeeze().cpu().numpy()
        ct_np     = ct.squeeze().cpu().numpy()
        t_cls_pred = model(ct[:, :, :96, :96, :96])['t_cls'].softmax(-1).squeeze().detach().cpu().numpy()
        n_cls_pred = model(ct[:, :, :96, :96, :96])['n_cls'].softmax(-1).squeeze().detach().cpu().numpy()
        t_pred_label = t_cls_pred.argmax()
        n_pred_label = n_cls_pred.argmax()

        # tumor가 있는 slice 찾기
        tumor_slices = np.where(seg_gt_np.sum(axis=(1, 2)) > 0)[0]
        if len(tumor_slices) == 0:
            tumor_slices = np.linspace(0, ct_np.shape[0]-1, num_slices, dtype=int)
        else:
            step = max(1, len(tumor_slices) // num_slices)
            tumor_slices = tumor_slices[::step][:num_slices]

        fig, axes = plt.subplots(len(tumor_slices), 3, figsize=(12, 4 * len(tumor_slices)))
        if len(tumor_slices) == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(
            f"Patient: {pid}\n"
            f"T-stage  GT={t_stage_names[t_gt]}  Pred={t_stage_names[t_pred_label]}  "
            f"({'✓' if t_gt == t_pred_label else '✗'})\n"
            f"N-stage  GT={n_stage_names[n_gt]}  Pred={n_stage_names[n_pred_label]}  "
            f"({'✓' if n_gt == n_pred_label else '✗'})",
            fontsize=13, fontweight="bold"
        )

        for row, s in enumerate(tumor_slices):
            ct_slice  = ct_np[s]
            gt_slice  = seg_gt_np[s]
            pr_slice  = t_seg_pred[s]

            # CT 정규화 (시각화용)
            ct_vis = np.clip(ct_slice, -1, 1)
            ct_vis = (ct_vis - ct_vis.min()) / (ct_vis.max() - ct_vis.min() + 1e-8)

            # col 0: CT + GT seg
            axes[row, 0].imshow(ct_vis, cmap="gray")
            axes[row, 0].imshow(np.ma.masked_where(gt_slice < 0.5, gt_slice),
                                cmap="Reds", alpha=0.5, vmin=0, vmax=1)
            axes[row, 0].set_title(f"Slice {s} | GT seg", fontsize=10)
            axes[row, 0].axis("off")

            # col 1: CT + Pred seg (threshold 0.5)
            pr_binary = (pr_slice > 0.3).astype(float)
            axes[row, 1].imshow(ct_vis, cmap="gray")
            axes[row, 1].imshow(np.ma.masked_where(pr_binary < 0.5, pr_binary),
                                cmap="Blues", alpha=0.5, vmin=0, vmax=1)
            axes[row, 1].set_title(f"Slice {s} | Pred seg (thr=0.5)", fontsize=10)
            axes[row, 1].axis("off")

            # col 2: probability heatmap
            im = axes[row, 2].imshow(ct_vis, cmap="gray")
            axes[row, 2].imshow(pr_slice, cmap="jet", alpha=0.4, vmin=0, vmax=1)
            axes[row, 2].set_title(f"Slice {s} | Prob heatmap", fontsize=10)
            axes[row, 2].axis("off")

        red_patch  = mpatches.Patch(color="red",  alpha=0.5, label="GT tumor")
        blue_patch = mpatches.Patch(color="blue", alpha=0.5, label="Pred tumor")
        fig.legend(handles=[red_patch, blue_patch], loc="lower center",
                   ncol=2, fontsize=11, bbox_to_anchor=(0.5, 0.01))

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        save_path = os.path.join(out_dir, f"{pid}_seg_vis.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[vis] saved → {save_path}")

    print(f"\n[vis] Done! {out_dir} 폴더 확인하세요.")


if __name__ == "__main__":
    visualize()
