# Phase 1: Low-Hanging Fruit Analysis Implementation Plan

This document outlines the easiest, highest-impact analyses to implement immediately for Phase 1, with outputs formatted for thesis writing.

---

## Selected Analyses (Prioritized by Implementation Ease)

### ✅ **TIER 1: Single Forward Pass (Easiest)**
These require only one model inference and simple computations.

1. **Attention Entropy Analysis** (ADVANCED_ANALYSIS_IDEAS.md §1.2)
   - Compute entropy of attention distributions across DINO layers and Temporal Transformer
   - Compare normal fire vs. EWE moments
   - **Output**: Entropy curves, statistical tables, thesis-ready figure with caption

2. **Temporal Attention Trajectory** (§2.2)
   - Extract and visualize temporal transformer attention weights over time
   - Track attention shift as RoE increases
   - **Output**: Temporal heatmaps, attention flow plots, quantitative metrics

3. **Attention Head Specialization** (§8.2)
   - Analyze if different transformer heads focus on different aspects
   - Measure head diversity and compute head-specific statistics
   - **Output**: Per-head attention visualizations, diversity metrics table

### ✅ **TIER 2: Multiple Forward Passes (Medium Effort)**

4. **Attention Consistency as Uncertainty** (§5.2)
   - Run inference with dropout enabled (Monte Carlo Dropout)
   - Measure variance in attention weights as confidence proxy
   - **Output**: Uncertainty maps, correlation with prediction accuracy

5. **Critical History Window Detection** (§2.1)
   - Ablate temporal windows (use t-1 only, t-1:t-2, t-1:t-3, t-1:t-4)
   - Measure IoU degradation for each configuration
   - **Output**: Performance curves, optimal window size table

6. **DINO Embedding Space Visualization** (§8.1)
   - Extract DINO features before fusion layer
   - Apply UMAP/t-SNE dimensionality reduction
   - Color by: fuel type, terrain slope, fire state, RoE level
   - **Output**: 2D/3D scatter plots, cluster analysis

### ✅ **TIER 3: Gradient-Based (Slightly More Complex)**

7. **Gradient Flow Through DINO** (§6.1)
   - Compute gradients w.r.t. DINO layers during backprop
   - Compare frozen vs. fine-tuned models
   - **Output**: Gradient magnitude plots, layer-wise flow visualization

---

## Implementation Structure

```
analysis/phase_1/
├── advanced/                          # New analyses
│   ├── __init__.py
│   ├── attention_entropy.py           # TIER 1.1
│   ├── temporal_trajectory.py         # TIER 1.2
│   ├── head_specialization.py         # TIER 1.3
│   ├── attention_uncertainty.py       # TIER 2.1
│   ├── history_window_ablation.py     # TIER 2.2
│   ├── embedding_space.py             # TIER 2.3
│   └── gradient_flow.py               # TIER 3.1
├── outputs/                           # Thesis-ready outputs
│   ├── figures/                       # High-res figures with captions
│   ├── tables/                        # LaTeX/Markdown tables
│   ├── metrics/                       # JSON structured metrics
│   └── reports/                       # Markdown analysis reports
└── run_advanced_analysis.py           # Main runner
```

---

## Output Format Requirements (Thesis-Ready)

### 1. **Structured Metrics (JSON)**
```json
{
  "analysis_name": "attention_entropy",
  "timestamp": "2025-10-19T14:30:00",
  "model_checkpoint": "checkpoints/phase1_frozen.pt",
  "dataset": "test_set_100_samples",
  "metrics": {
    "mean_entropy_normal_fire": 2.34,
    "mean_entropy_ewe": 1.87,
    "std_entropy_normal_fire": 0.45,
    "std_entropy_ewe": 0.32,
    "statistical_test": {
      "test": "t-test",
      "p_value": 0.0001,
      "effect_size_cohens_d": 1.23
    }
  },
  "interpretation": "EWE predictions show significantly lower attention entropy (p<0.001, d=1.23), indicating focused attention on critical fire boundaries during extreme events."
}
```

### 2. **Figure Files with Metadata**
- **Filename**: `fig_attention_entropy_comparison.png` (high DPI: 300+)
- **Caption file**: `fig_attention_entropy_comparison_caption.txt`
  ```
  Figure X: Attention entropy distribution across DINO transformer layers for normal fire (blue) vs. EWE (red) predictions. Lower entropy during EWE indicates concentrated attention on fire boundaries (p<0.001, Cohen's d=1.23, N=100). Error bars represent 95% confidence intervals.
  ```

### 3. **LaTeX Tables**
```latex
\begin{table}[h]
\centering
\caption{Attention Entropy by Fire Event Type}
\begin{tabular}{lcccc}
\toprule
Event Type & Mean Entropy & Std Dev & N Samples & p-value \\
\midrule
Normal Fire & 2.34 & 0.45 & 150 & - \\
EWE & 1.87 & 0.32 & 50 & $<0.001^{***}$ \\
\bottomrule
\end{tabular}
\label{tab:attention_entropy}
\end{table}
```

### 4. **Markdown Reports** (One per analysis)
```markdown
# Analysis Report: Attention Entropy Across Fire Event Types

## Objective
Quantify attention distribution patterns in DINO encoder to understand model focus during normal vs. extreme wildfire events.

## Method
- Extracted attention weights from all 12 DINO transformer blocks
- Computed Shannon entropy for each attention distribution
- Compared entropy distributions using independent t-test
- Analyzed N=200 samples (150 normal, 50 EWE)

## Key Findings
1. **EWE predictions show 20% lower attention entropy** (1.87 vs 2.34, p<0.001)
2. **Entropy decreases in deeper layers** for both event types
3. **Greatest divergence at layers 8-10**, suggesting mid-level feature specialization

## Thesis Integration
- **Chapter 5.1**: Use as evidence for spatial importance analysis
- **Figure**: Include `fig_attention_entropy_comparison.png`
- **Table**: Reference `tab:attention_entropy`
- **Key Quote**: "The significant reduction in attention entropy during EWE prediction (Cohen's d=1.23) indicates that the model learns to concentrate on critical spatial features..."

## Files Generated
- `outputs/figures/fig_attention_entropy_comparison.png`
- `outputs/tables/tab_attention_entropy.tex`
- `outputs/metrics/attention_entropy_metrics.json`
```

---

## Shared Utilities Module

```python
# analysis/phase_1/advanced/thesis_utils.py

"""
Utilities for generating thesis-ready outputs
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

class ThesisOutputManager:
    """Manages structured output for thesis writing"""
    
    def __init__(self, output_dir: str = "analysis/phase_1/outputs"):
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.metrics_dir = self.output_dir / "metrics"
        self.reports_dir = self.output_dir / "reports"
        
        for d in [self.figures_dir, self.tables_dir, self.metrics_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def save_figure(
        self, 
        fig: plt.Figure, 
        name: str, 
        caption: str,
        chapter_section: str,
        dpi: int = 300
    ):
        """Save figure with caption and metadata"""
        fig_path = self.figures_dir / f"{name}.png"
        caption_path = self.figures_dir / f"{name}_caption.txt"
        
        fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        
        with open(caption_path, 'w') as f:
            f.write(f"[Chapter {chapter_section}]\n")
            f.write(caption)
        
        print(f"✓ Saved figure: {fig_path}")
        return fig_path
    
    def save_metrics(
        self, 
        name: str, 
        metrics: Dict[str, Any],
        interpretation: str,
        model_checkpoint: str,
        dataset: str
    ):
        """Save structured metrics as JSON"""
        metrics_path = self.metrics_dir / f"{name}_metrics.json"
        
        output = {
            "analysis_name": name,
            "timestamp": datetime.now().isoformat(),
            "model_checkpoint": model_checkpoint,
            "dataset": dataset,
            "metrics": metrics,
            "interpretation": interpretation
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Saved metrics: {metrics_path}")
        return metrics_path
    
    def save_latex_table(
        self, 
        name: str, 
        caption: str, 
        label: str,
        latex_content: str
    ):
        """Save LaTeX table"""
        table_path = self.tables_dir / f"{name}.tex"
        
        full_latex = f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
{latex_content}
\\label{{{label}}}
\\end{{table}}"""
        
        with open(table_path, 'w') as f:
            f.write(full_latex)
        
        print(f"✓ Saved table: {table_path}")
        return table_path
    
    def save_report(
        self, 
        name: str, 
        content: str
    ):
        """Save markdown analysis report"""
        report_path = self.reports_dir / f"{name}_report.md"
        
        with open(report_path, 'w') as f:
            f.write(content)
        
        print(f"✓ Saved report: {report_path}")
        return report_path

def format_p_value(p: float) -> str:
    """Format p-value for publication"""
    if p < 0.001:
        return "$<0.001^{***}$"
    elif p < 0.01:
        return f"${p:.3f}^{{**}}$"
    elif p < 0.05:
        return f"${p:.3f}^{{*}}$"
    else:
        return f"${p:.3f}$"

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std
```

---

## Execution Plan

### Week 1: TIER 1 (3 analyses)
- Implement `attention_entropy.py`
- Implement `temporal_trajectory.py`
- Implement `head_specialization.py`
- Run on test set
- Generate all outputs

### Week 2: TIER 2 (3 analyses)
- Implement `attention_uncertainty.py`
- Implement `history_window_ablation.py`
- Implement `embedding_space.py`
- Run on test set
- Generate all outputs

### Week 3: TIER 3 + Integration
- Implement `gradient_flow.py`
- Cross-reference all analyses
- Generate summary report
- Prepare thesis chapter draft

---

## Success Criteria

✅ Each analysis produces:
1. High-resolution figures (300 DPI minimum)
2. Structured metrics (JSON) with statistical tests
3. LaTeX tables ready for copy-paste
4. Markdown report with thesis integration notes
5. Clear file naming convention

✅ Outputs can be directly consumed by thesis-writing AI:
- All figures have captions
- All metrics have interpretations
- All tables have labels and references
- All reports map to thesis chapters

✅ Reproducibility:
- Analysis metadata includes model checkpoint and dataset
- Random seeds documented
- All dependencies in `pyproject.toml`

---

## Next Steps

1. **Create base utilities**: Implement `thesis_utils.py`
2. **Start with attention_entropy.py**: Easiest analysis, validates pipeline
3. **Iterate**: Each analysis follows same output pattern
4. **Aggregate**: Final summary report combines all findings

This plan balances quick wins with thesis-ready quality.
