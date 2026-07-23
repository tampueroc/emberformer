# Quick Start: Running Analysis

## Overview

Two analysis systems available:

1. **Basic Analysis** - Existing interpretability suite (spatial, temporal, features, wind, extreme events)
2. **Advanced Analysis** - New thesis-ready analyses with timestamped outputs

---

## 1. Basic Analysis (Already Implemented)

### Run All Analyses
```bash
cd /Users/tampueroc/Code/Personal/Thesis/emberformer
python analysis/phase_1/run_analysis.py
```

### Run Specific Analyses
```bash
# Skip slow feature ablation
python analysis/phase_1/run_analysis.py --skip features

# Run only spatial and temporal
python analysis/phase_1/run_analysis.py --skip features wind extreme

# Use fewer samples for speed
python analysis/phase_1/run_analysis.py --num-samples 50
```

### Outputs (saved to `analysis/phase_1/`)
- `spatial/*.png` - Gradient-based saliency maps
- `temporal/temporal_importance.png` - Temporal attention patterns
- `features/feature_importance.png` - Feature ablation results
- `wind/wind_alignment.png` - Wind-fire alignment
- `extreme_events/*.png` - Extreme event visualizations

---

## 2. Advanced Analysis (Thesis-Ready, New)

### Run with Auto-Timestamp
```bash
cd /Users/tampueroc/Code/Personal/Thesis/emberformer
python analysis/phase_1/run_advanced_analysis.py
```

Outputs saved to: `results/phase_1/{YYYYMMDD_HHMMSS}/`

### Run with Custom Name
```bash
python analysis/phase_1/run_advanced_analysis.py --run-name frozen_dino_baseline
```

Outputs saved to: `results/phase_1/frozen_dino_baseline/`

### Run Specific Analyses
```bash
# Currently implemented: entropy
python analysis/phase_1/run_advanced_analysis.py --analyses entropy

# Coming soon: trajectory, heads, uncertainty, history, embeddings, gradients
# python analysis/phase_1/run_advanced_analysis.py --analyses entropy trajectory heads
```

### Outputs (thesis-ready format)
```
results/phase_1/{timestamp}/
├── figures/
│   ├── fig_attention_entropy_comparison.png (300 DPI)
│   └── fig_attention_entropy_comparison_caption.txt
├── tables/
│   └── tab_attention_entropy.tex (LaTeX ready)
├── metrics/
│   └── attention_entropy_metrics.json (with stats)
└── reports/
    └── attention_entropy_report.md
```

---

## Quick Commands

### Check if checkpoint exists
```bash
ls -lh ~/data/emberformer/checkpoints/dino_phase1_best.pt
```

### Run fast basic analysis (5-10 min)
```bash
python analysis/phase_1/run_analysis.py --skip features wind extreme --num-samples 50
```

### Run thesis-ready entropy analysis (10-15 min)
```bash
python analysis/phase_1/run_advanced_analysis.py --num-samples 100
```

### Use CPU instead of GPU
```bash
python analysis/phase_1/run_analysis.py --device cpu
python analysis/phase_1/run_advanced_analysis.py --device cpu
```

---

## Troubleshooting

### Missing checkpoint
```bash
# Check available checkpoints
ls ~/data/emberformer/checkpoints/

# If missing, you need to train Phase 1 first
uv run python scripts/train_dino.py --phase 1 --config configs/emberformer_dino.yaml --gpu 0
```

### Out of memory
```bash
# Use fewer samples
python analysis/phase_1/run_analysis.py --num-samples 50 --skip features

# Or run on CPU (slower but no memory limit)
python analysis/phase_1/run_analysis.py --device cpu --num-samples 50
```

### Module import errors
```bash
# Make sure you're in the project root
cd /Users/tampueroc/Code/Personal/Thesis/emberformer

# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"
```

---

## Expected Run Times

### Basic Analysis (full suite)
- Spatial importance: 5-10 min
- Temporal importance: 10-15 min
- Feature ablation: 30-60 min (slow!)
- Wind analysis: 15-20 min
- Extreme events: 20-30 min
- **Total: ~1.5-2 hours**

### Advanced Analysis (entropy only)
- Attention entropy: 10-15 min
- **Total: ~15 min per analysis**

---

## Phase 2 Analysis (After Fine-Tuning)

Once Phase 2 training completes, run the same analyses:

```bash
# Basic analysis on Phase 2
python analysis/phase_1/run_analysis.py \
    --checkpoint ~/data/emberformer/checkpoints/dino_phase2_best.pt \
    --output-dir analysis/phase_2

# Advanced analysis on Phase 2
python analysis/phase_1/run_advanced_analysis.py \
    --checkpoint ~/data/emberformer/checkpoints/dino_phase2_best.pt \
    --phase 2 \
    --run-name finetuned_dino
```

---

## Next Steps

1. **Run basic analysis** to verify model behavior
2. **Run advanced analysis** to generate thesis-ready outputs
3. **Compare Phase 1 vs Phase 2** after fine-tuning
4. **Implement remaining TIER 1-3 analyses** (see `PHASE1_IMPLEMENTATION_PLAN.md`)

---

## Additional Resources

- **Basic analysis details**: `analysis/phase_1/README.md`
- **Advanced analysis plan**: `analysis/phase_1/PHASE1_IMPLEMENTATION_PLAN.md`
- **Novel analysis ideas**: `ADVANCED_ANALYSIS_IDEAS.md`
- **Thesis structure**: `analysis/THESIS_WRITING_GUIDANCE.md`
- **Results structure**: `results/README.md`
