# Datasets

This directory is a placeholder for dataset root symlinks or metadata.
Raw data files (*.nii.gz, *.dcm) are listed in `.gitignore` and must be downloaded separately.

---

## 1. LiTS (Liver Tumor Segmentation)

- **URL**: https://competitions.codalab.org/competitions/17094
- **Scans**: 131 abdominal CT scans
- **Annotations**: Liver (label=1) and Tumor (label=2) — HCC, ICC histology
- **Used for**: Phase 1 pre-training + Phase 3 fine-tuning and evaluation

### Expected Structure

```
lits/
├── imagesTr/
│   ├── volume-0.nii.gz
│   ├── volume-1.nii.gz
│   └── ...
└── labelsTr/
    ├── segmentation-0.nii.gz
    ├── segmentation-1.nii.gz
    └── ...
```

### Download

1. Register at https://competitions.codalab.org/competitions/17094
2. Download `Training Batch 1` and `Training Batch 2`
3. Place files under `datasets/lits/`

---

## 2. TCIA (The Cancer Imaging Archive)

Three liver collections are used:

| Collection | Cases | Annotations |
|---|---|---|
| TCGA-LIHC | ~250 | Liver lesions |
| CPTAC-LIHC | ~100 | Liver lesions |
| HCC-TACE-Seg | ~224 | HCC segmentation masks |

- **URL**: https://www.cancerimagingarchive.net/
- **Used for**: Phase 1 pre-training

### Download

Use the TCIA Data Retriever or NBIA Data Retriever:
```bash
pip install tcia-utils
# Then use the TCIA REST API or Data Retriever GUI
```

Place downloaded collections under:
```
datasets/tcia/
├── TCGA-LIHC/
├── CPTAC-LIHC/
└── HCC-TACE-Seg/
```

Optional: Create `datasets/tcia/metadata.json` with TNM stage labels per case:
```json
{
  "TCGA-LIHC-XXXXXX": {"tstage": 1, "nstage": 0},
  ...
}
```

---

## 3. AMOS (Abdominal Multi-Organ Segmentation)

- **URL**: https://amos22.grand-challenge.org/
- **Scans**: 500 CT + 100 MRI
- **Annotations**: 15 organ segmentations (including liver, spleen, kidney, etc.)
- **Used for**: Phase 1 pre-training (unlabeled streaming — organ labels not required)

### Download

Follow instructions at https://amos22.grand-challenge.org/Dataset/

Expected structure (nnUNet format):
```
datasets/amos/
├── imagesTr/
│   ├── amos_0001_0000.nii.gz
│   └── ...
└── labelsTr/
    ├── amos_0001.nii.gz
    └── ...
```

---

## Preprocessing

All datasets are preprocessed automatically by `src/data/preprocessing.py`:

1. **HU Windowing**: Clip to [-100, 400] HU (liver window), normalize to [0, 1]
2. **Isotropic Resampling**: Resample to 1x1x1 mm using SimpleITK
3. **Patch Extraction**: 96x96x96 voxel patches with stride 48
4. **Z-score Normalization**: Within foreground voxels

---

## Dataset Summary

| Dataset | Modality | Scans | Labels | Phase |
|---|---|---|---|---|
| LiTS | CT | 131 | Liver + Tumor | 1, 3 |
| TCIA-TCGA-LIHC | CT | ~250 | Liver lesion | 1 |
| TCIA-CPTAC-LIHC | CT | ~100 | Liver lesion | 1 |
| TCIA-HCC-TACE-Seg | CT | ~224 | HCC mask | 1 |
| AMOS | CT+MRI | 600 | 15 organs | 1 |
