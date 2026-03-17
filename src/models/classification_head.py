"""
TALARIA Classification Head.

T-Stage: tumor segmentation mask에서 morphological feature를 추출한 뒤
         3-layer MLP로 T1/T2/T3/T4 분류 (논문 방식 구현)

N-Stage: deep_feat GAP -> MLP로 N0/N1 분류
         Manifold Mixup (latent space) 지원
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Morphological Feature Extractor (T-Stage)
# ---------------------------------------------------------------------------

class MorphologicalFeatureExtractor(nn.Module):
    """
    Tumor segmentation mask에서 T-stage 분류에 필요한 형태적 특징 추출.

    AJCC 8th edition T-stage criteria 기반:
        - maximum lesion diameter (cm)
        - lesion count
        - vascular invasion proxy  (간문맥/간정맥 근접도)
        - hepatic lobe involvement fraction

    Args:
        voxel_spacing_mm: isotropic voxel spacing (default 1.0mm, normalized 입력 기준)

    Input:
        seg_prob: (B, 1, D, H, W) sigmoid probability map (from T-Branch)
        threshold: binary mask 변환 threshold (default 0.5)

    Output:
        feat: (B, 4) float32 — [max_diam_norm, lesion_count_norm,
                                vasc_invasion_proxy, lobe_frac]

    Note:
        - 모든 feature는 [0,1]로 min-max 정규화
        - connected component는 미분 불가능 -> MLP 입력 전 detach
        - 학습 중에는 soft prob map으로 미분 가능한 대체값 사용
    """

    def __init__(self, voxel_spacing_mm: float = 1.0):
        super().__init__()
        self.voxel_spacing_mm = voxel_spacing_mm

        # diameter 정규화 기준: 10cm (100mm) = AJCC T4 기준 상한
        self.max_diam_norm = 100.0
        # lesion count 정규화 기준: 5개 이상이면 T3 이상
        self.max_count_norm = 5.0

    @torch.no_grad()
    def _connected_components_3d(self, binary: torch.Tensor) -> Tuple[int, float]:
        """
        간단한 connected component labeling (BFS, CPU).
        binary: (D, H, W) bool tensor

        Returns:
            n_components: int
            max_diam_vox: float (Euclidean bounding box 대각선 길이, voxel 단위)
        """
        mask_np = binary.cpu().numpy().astype(bool)
        D, H, W = mask_np.shape

        visited = [[[ False]*W for _ in range(H)] for _ in range(D)]
        components = []

        def bfs(sd, sh, sw):
            queue = [(sd, sh, sw)]
            visited[sd][sh][sw] = True
            voxels = []
            while queue:
                d, h, w = queue.pop()
                voxels.append((d, h, w))
                for dd, dh, dw in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                    nd,nh,nw = d+dd, h+dh, w+dw
                    if 0<=nd<D and 0<=nh<H and 0<=nw<W:
                        if mask_np[nd,nh,nw] and not visited[nd][nh][nw]:
                            visited[nd][nh][nw] = True
                            queue.append((nd,nh,nw))
            return voxels

        for d in range(D):
            for h in range(H):
                for w in range(W):
                    if mask_np[d,h,w] and not visited[d][h][w]:
                        components.append(bfs(d, h, w))

        if not components:
            return 0, 0.0

        max_diam = 0.0
        for comp in components:
            ds = [v[0] for v in comp]
            hs = [v[1] for v in comp]
            ws = [v[2] for v in comp]
            # bounding box 대각선 (voxel)
            diam = ((max(ds)-min(ds))**2 + (max(hs)-min(hs))**2 + (max(ws)-min(ws))**2) ** 0.5
            max_diam = max(max_diam, diam)

        return len(components), max_diam

    def _soft_features(self, prob: torch.Tensor) -> torch.Tensor:
        """
        학습 중 미분 가능한 soft feature 추출.
        binary mask 없이 probability map에서 근사값 계산.

        prob: (D, H, W) float [0,1]
        Returns: (4,) tensor
        """
        D, H, W = prob.shape

        # 1. max diameter proxy: soft mask의 spatial extent (weighted std)
        total = prob.sum().clamp(min=1e-6)
        d_idx = torch.arange(D, device=prob.device, dtype=prob.dtype)
        h_idx = torch.arange(H, device=prob.device, dtype=prob.dtype)
        w_idx = torch.arange(W, device=prob.device, dtype=prob.dtype)

        mean_d = (prob.sum(dim=(1,2)) * d_idx).sum() / total
        mean_h = (prob.sum(dim=(0,2)) * h_idx).sum() / total
        mean_w = (prob.sum(dim=(0,1)) * w_idx).sum() / total

        var_d = ((d_idx - mean_d)**2 * prob.sum(dim=(1,2))).sum() / total
        var_h = ((h_idx - mean_h)**2 * prob.sum(dim=(0,2))).sum() / total
        var_w = ((w_idx - mean_w)**2 * prob.sum(dim=(0,1))).sum() / total

        # diameter proxy = 2 * sqrt(max variance) * spacing
        diam_vox = 2.0 * (torch.stack([var_d, var_h, var_w]).max().clamp(min=0.0).sqrt())
        diam_mm  = diam_vox * self.voxel_spacing_mm
        diam_norm = (diam_mm / self.max_diam_norm).clamp(0.0, 1.0)

        # 2. lesion count proxy: volume / typical lesion size
        volume = prob.sum()
        count_proxy = (volume / (4.0/3.0 * 3.14159 * (10.0**3))).clamp(0.0, self.max_count_norm)
        count_norm  = count_proxy / self.max_count_norm

        # 3. vascular invasion proxy: high-prob mass in central 1/3 region
        d1, d2 = D//3, 2*D//3
        h1, h2 = H//3, 2*H//3
        w1, w2 = W//3, 2*W//3
        central_mass = prob[d1:d2, h1:h2, w1:w2].sum()
        vasc_proxy   = (central_mass / total).clamp(0.0, 1.0)

        # 4. hepatic lobe fraction: coronal half 비율
        left_mass  = prob[:, :, :W//2].sum()
        right_mass = prob[:, :, W//2:].sum()
        lobe_frac  = (torch.min(left_mass, right_mass) /
                      (torch.max(left_mass, right_mass).clamp(min=1e-6))).clamp(0.0, 1.0)

        return torch.stack([diam_norm, count_norm, vasc_proxy, lobe_frac])

    def forward(self, seg_prob: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Args:
            seg_prob: (B, 1, D, H, W) T-Branch sigmoid output
        Returns:
            morph_feat: (B, 4) float32
        """
        B = seg_prob.shape[0]
        feats = []

        for b in range(B):
            prob = seg_prob[b, 0]   # (D, H, W)

            if self.training:
                # 학습 중: soft, 미분 가능
                feat = self._soft_features(prob)
            else:
                # 추론 중: binary mask -> exact morphological features
                binary = (prob > threshold)
                n_comp, max_diam_vox = self._connected_components_3d(binary)

                diam_mm   = max_diam_vox * self.voxel_spacing_mm
                diam_norm = min(diam_mm / self.max_diam_norm, 1.0)
                count_norm = min(n_comp / self.max_count_norm, 1.0)

                # vascular invasion proxy (soft, always)
                total = prob.sum().clamp(min=1e-6)
                D, H, W = prob.shape
                d1,d2 = D//3, 2*D//3
                h1,h2 = H//3, 2*H//3
                w1,w2 = W//3, 2*W//3
                vasc_proxy = (prob[d1:d2,h1:h2,w1:w2].sum() / total).clamp(0.0, 1.0)

                left  = prob[:,:,:W//2].sum()
                right = prob[:,:,W//2:].sum()
                lobe_frac = (torch.min(left, right) /
                             torch.max(left, right).clamp(min=1e-6)).clamp(0.0, 1.0)

                feat = torch.tensor(
                    [diam_norm, count_norm, float(vasc_proxy), float(lobe_frac)],
                    dtype=seg_prob.dtype, device=seg_prob.device
                )

            feats.append(feat)

        return torch.stack(feats, dim=0)   # (B, 4)


# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    Dual classification head.

    T-Stage: MorphologicalFeatureExtractor(seg_prob) -> 3-layer MLP -> T1~T4
    N-Stage: GAP(deep_feat) -> MLP -> N0/N1  (Manifold Mixup 지원)

    Args:
        in_ch:     encoder deep_feat channels (320)
        t_classes: T-stage classes (4)
        n_classes: N-stage classes (2)
        dropout:   dropout rate
    """

    def __init__(
        self,
        in_ch: int = 320,
        t_classes: int = 4,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.morph_extractor = MorphologicalFeatureExtractor()
        self.gap = nn.AdaptiveAvgPool3d(1)

        # T-Stage: morphological features(4) -> 3-layer MLP
        self.t_head = nn.Sequential(
            nn.Linear(4, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, t_classes),
        )

        # N-Stage: deep_feat GAP(320) -> MLP  (Manifold Mixup here)
        self.n_head = nn.Sequential(
            nn.Linear(in_ch, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(
        self,
        deep_feat: torch.Tensor,
        t_seg_prob: torch.Tensor,
        apply_manifold_mixup: bool = True,
        mixup_alpha: float = 2.0,
        mixup_prob: float = 1.0,
        perm_idx: Optional[torch.Tensor] = None,
        lam: Optional[float] = None,
    ):
        """
        Args:
            deep_feat:  (B, 320, D/16, H/16, W/16)
            t_seg_prob: (B, 1, D, H, W)  T-Branch sigmoid output
        Returns:
            t_logit:    (B, t_classes)
            n_logit:    (B, n_classes)
            mixup_meta: dict {mixup_lam, mixup_perm, mixup_applied}
        """
        # ── T-Stage: morphological features -> MLP ──────────────────────────
        morph_feat = self.morph_extractor(t_seg_prob)   # (B, 4)
        t_logit    = self.t_head(morph_feat)             # (B, t_classes)

        # ── N-Stage: GAP + optional Manifold Mixup ──────────────────────────
        x = self.gap(deep_feat).flatten(1)   # (B, 320)

        mixup_applied = False
        mixup_lam: Optional[float] = None
        mixup_perm: Optional[torch.Tensor] = None

        bsz = x.size(0)
        can_mixup = bsz > 1 and mixup_prob > 0.0 and mixup_alpha > 0.0

        if (self.training and apply_manifold_mixup and can_mixup
                and torch.rand((), device=x.device) < min(float(mixup_prob), 1.0)):
            mixup_perm = perm_idx if perm_idx is not None \
                else torch.randperm(bsz, device=x.device)
            if lam is None:
                mixup_lam = float(torch.distributions.Beta(
                    mixup_alpha, mixup_alpha).sample().item())
            else:
                mixup_lam = float(lam)
            x = mixup_lam * x + (1.0 - mixup_lam) * x[mixup_perm]
            mixup_applied = True

        n_logit = self.n_head(x)   # (B, n_classes)

        mixup_meta: Dict[str, Any] = {
            'mixup_lam':     mixup_lam,
            'mixup_perm':    mixup_perm,
            'mixup_applied': mixup_applied,
        }
        return t_logit, n_logit, mixup_meta


if __name__ == '__main__':
    head      = ClassificationHead()
    deep_feat = torch.randn(2, 320, 6, 6, 6)
    t_seg_prob = torch.sigmoid(torch.randn(2, 1, 96, 96, 96))

    head.eval()
    t_logit, n_logit, meta = head(deep_feat, t_seg_prob)
    print(f"t_logit: {t_logit.shape}")   # (2, 4)
    print(f"n_logit: {n_logit.shape}")   # (2, 2)
    print(f"mixup:   {meta}")
