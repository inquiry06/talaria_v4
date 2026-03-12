# Experiments Directory

All experiment outputs are stored under this directory with the following naming convention:

```
experiments/
└── {phase}_{YYYYMMDD_HHMMSS}/
    ├── config.yaml          # Snapshot of the configuration used for this run
    ├── checkpoints/         # Model checkpoints (*.ckpt) — gitignored
    │   ├── best.ckpt        # Best checkpoint (lowest loss / highest metric)
    │   ├── epoch_0010.ckpt
    │   └── final.ckpt       # Final checkpoint at the end of training
    ├── logs/                # Training logs — gitignored
    │   └── train.log
    └── results/             # Evaluation metrics and prediction outputs
        ├── metrics.json     # JSON with DSC, AUC, etc.
        ├── tnm_report.json  # TNM staging report (inference only)
        └── visualizations/  # Optional segmentation overlays
```

## Phase Naming Convention

| Phase | Prefix | Example |
|---|---|---|
| Phase 1 (Pre-training) | `pretrain_` | `pretrain_20250301_143022/` |
| Phase 2 (Distillation) | `distill_` | `distill_20250302_091500/` |
| Phase 3 (Fine-tuning)  | `finetune_` | `finetune_20250303_120000/` |

## Notes

- `checkpoints/` and `logs/*.log` are listed in `.gitignore` to avoid committing large files.
- Always refer to `config.yaml` inside each experiment folder to reproduce results.
- Use `results/metrics.json` for comparing runs.
