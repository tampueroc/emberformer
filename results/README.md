# Results Directory

This directory contains timestamped analysis results for thesis writing.

## Structure

```
results/
├── phase_1/
│   ├── 20251019_143022/          # Timestamped run
│   │   ├── figures/              # High-res figures with captions
│   │   ├── tables/               # LaTeX tables
│   │   ├── metrics/              # JSON metrics with stats
│   │   └── reports/              # Markdown analysis reports
│   ├── 20251020_091545/
│   └── frozen_dino_baseline/     # Named run
├── phase_2/
└── phase_3/
```

## Usage

### Automatic Timestamping
```python
from analysis.phase_1.advanced import ThesisOutputManager

# Creates results/phase_1/{YYYYMMDD_HHMMSS}/
output_manager = ThesisOutputManager(phase=1, use_timestamp=True)
```

### Named Runs
```python
# Creates results/phase_1/frozen_dino_baseline/
output_manager = ThesisOutputManager(
    phase=1, 
    custom_run_name="frozen_dino_baseline"
)
```

### Latest Run (No Timestamp)
```python
# Creates results/phase_1/latest/ (overwrites on each run)
output_manager = ThesisOutputManager(phase=1, use_timestamp=False)
```

## Output Types

### 1. Figures (`figures/`)
- High-resolution PNG (300+ DPI)
- Separate caption file (`.txt`)
- Metadata JSON with chapter mapping

### 2. Tables (`tables/`)
- LaTeX format ready for copy-paste
- Includes `\begin{table}...\end{table}` wrapper
- Pre-formatted with booktabs style

### 3. Metrics (`metrics/`)
- Structured JSON with:
  - Analysis name and timestamp
  - Model checkpoint and dataset info
  - Computed metrics with statistical tests
  - Human-readable interpretation

### 4. Reports (`reports/`)
- Markdown analysis reports with:
  - Objective and method
  - Key findings (numbered list)
  - Thesis integration notes
  - Generated files list

## Thesis Integration

All outputs are designed for direct consumption by AI thesis-writing tools:
- Figures include chapter section mappings
- Tables include LaTeX labels for cross-referencing
- Metrics include statistical significance tests
- Reports map findings to specific thesis chapters

## Version Control

The `results/` directory is gitignored (too large), but:
- Directory structure is tracked
- README and documentation are committed
- Example outputs may be included for reference

## Archiving

For important runs, consider:
1. Copy timestamped directory to external storage
2. Create a summary in `EXPERIMENTS.md` referencing the run
3. Keep final thesis-ready outputs in a separate `thesis_outputs/` directory
