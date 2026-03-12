**한국어** | [English](README.md)

# TALARIA: 간암 병기 결정을 위한 종양 인식 림프절 분석 및 통합 평가 프레임워크

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 개요

TALARIA는 3D CT 영상을 이용한 간암 자동 병기 결정을 위한 3단계 딥러닝 프레임워크입니다. Mamba 기반 상태 공간 모델(SSM) 인코더, 교사-학생 지식 증류, 이중 분기 분할 헤드, TNM 분류 헤드를 통합하여 종양 인식 기반의 포괄적인 평가 파이프라인을 제공합니다.

![TALARIA 아키텍처](figures/fig1.png)

*그림 1: TALARIA-Net Core 파이프라인 — 환자 3D CT 입력부터 T/N 영역 분할 및 TNM 병기 보고서 출력까지.*

---

## 아키텍처

TALARIA는 강건하고 효율적인 간암 병기 결정을 위해 3단계 학습 전략을 사용합니다:

![학습 파이프라인](figures/fig2.png)

*그림 2: 3단계 학습 파이프라인.*

### Phase 1: 자기지도 사전학습 (Self-Supervised Pre-training)
- **Mamba-SSM 인코더**: 3D 체적 특징 추출을 위한 선형 복잡도 O(N) 구조
- **재구성 디코더**: 마스킹된 볼륨 재구성을 통한 자기지도 사전학습 (LiTS, AMOS, TCIA 데이터셋 활용)

### Phase 2: 지식 증류 (Knowledge Distillation)
- **교사 → 학생** 증류를 통해 인코더를 경량 학생 모델로 압축
- 낮은 연산 비용으로 분할 및 분류 성능 유지

### Phase 3: 태스크별 헤드 학습 (인코더 동결)
- **(a) 이중 분기 분할 헤드 (Dual-Branch Segmentation Head)**
  - **T-Branch**: 심층 특징 기반 대형 종양 분할
  - **N-Branch**: 얕은 특징 + 어텐션 게이트를 이용한 미세 림프절 검출 (10mm 미만)
- **(b) T-Stage 분류 헤드**: T1 ~ T4 분류
- **(c) N-Stage 분류 헤드**: N0 ~ N1 분류

### 추론 (Inference)
- **소프트 보팅 TTA** (회전, 플립 증강) + **TTT 앙상블**

---

## 결과

TALARIA는 LiTS 벤치마크에서 모든 평가 지표에서 기존 기법을 능가합니다.

![결과 비교](figures/fig3.png)

*그림 3: 정성적 비교 — 정답(Ground Truth) vs. 기존 UNet vs. TALARIA(제안).*

| 방법 | DSC (종양) | 정밀도 | 재현율 | T-Stage AUC | N-Stage AUC |
|---|---|---|---|---|---|
| UNet | 0.681 | 0.703 | 0.672 | 0.821 | 0.794 |
| nnUNet | 0.714 | 0.731 | 0.698 | 0.848 | 0.823 |
| **TALARIA (제안)** | **0.739+** | **0.761** | **0.728** | **0.891** | **0.867** |

---

## 데이터셋

| 데이터셋 | 촬영 방식 | 스캔 수 | 어노테이션 | 활용 단계 |
|---|---|---|---|---|
| [LiTS](https://competitions.codalab.org/competitions/17094) | CT | 131 | 간 & 종양 (HCC, ICC) | Phase 1 사전학습 + Phase 3 파인튜닝 |
| [TCIA (TCGA-LIHC)](https://www.cancerimagingarchive.net/) | CT | ~250 | 간 병변 | Phase 1 사전학습 |
| [TCIA (CPTAC-LIHC)](https://www.cancerimagingarchive.net/) | CT | ~100 | 간 병변 | Phase 1 사전학습 |
| [TCIA (HCC-TACE-Seg)](https://www.cancerimagingarchive.net/) | CT | ~224 | HCC 분할 | Phase 1 사전학습 |
| [AMOS](https://amos22.grand-challenge.org/) | CT + MRI | 500 CT + 100 MRI | 15개 장기 | Phase 1 사전학습 (비레이블 스트리밍) |

---

## 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/talaria.git
cd talaria

# 가상환경 생성
python -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

> **참고**: `mamba-ssm`은 CUDA 11.6 이상과 호환 GPU가 필요합니다. 자세한 설치 방법은 [mamba-ssm](https://github.com/state-spaces/mamba)을 참조하세요.

---

## 사용법

### Phase 1: 자기지도 사전학습

```bash
bash scripts/run_pretrain.sh
# 또는
python -m src.training.pretrain --config configs/pretrain.yaml
```

### Phase 2: 지식 증류

```bash
bash scripts/run_distill.sh
# 또는
python -m src.training.distill --config configs/distill.yaml \
    --teacher_ckpt experiments/pretrain_<timestamp>/checkpoints/best.ckpt
```

### Phase 3: 태스크별 파인튜닝

```bash
bash scripts/run_finetune.sh
# 또는
python -m src.training.finetune --config configs/finetune.yaml \
    --student_ckpt experiments/distill_<timestamp>/checkpoints/best.ckpt
```

### 추론

```bash
python -m src.inference.soft_voting \
    --config configs/finetune.yaml \
    --checkpoint experiments/finetune_<timestamp>/checkpoints/best.ckpt \
    --input /path/to/ct_scan.nii.gz \
    --output /path/to/output/
```

---

## 실험 관리

모든 실험 결과는 `experiments/` 폴더에 다음 구조로 저장됩니다:

```
experiments/
└── {phase}_{YYYYMMDD_HHMMSS}/
    ├── config.yaml          # 사용된 설정 파일 스냅샷
    ├── checkpoints/         # 모델 체크포인트 (*.ckpt) — gitignore 적용
    ├── logs/                # 학습 로그 (*.log) — gitignore 적용
    └── results/             # 평가 지표, 예측 결과
```

자세한 내용은 [experiments/README.md](experiments/README.md)를 참조하세요.

---

## 인용

TALARIA를 연구에 활용하신 경우 아래를 인용해 주세요:

```bibtex
@article{talaria2025,
  title     = {TALARIA: Tumor-Aware Lymph-node Analysis and Robust Integrated Assessment for Liver Cancer Staging},
  author    = {Your Name and Collaborators},
  journal   = {arXiv preprint},
  year      = {2025},
  url       = {https://arxiv.org/abs/xxxx.xxxxx}
}
```

---

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.
