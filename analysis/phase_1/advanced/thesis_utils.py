"""
Utilities for generating thesis-ready outputs from analysis
All outputs formatted for direct consumption by thesis-writing AI
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
from scipy import stats


class ThesisOutputManager:
    """
    Manages structured output for thesis writing
    
    Generates:
    - High-resolution figures with captions
    - Structured metrics (JSON) with statistical tests
    - LaTeX tables ready for copy-paste
    - Markdown analysis reports
    """
    
    def __init__(
        self, 
        phase: int = 1, 
        use_timestamp: bool = True,
        custom_run_name: Optional[str] = None
    ):
        """
        Args:
            phase: Phase number (1, 2, 3, etc.)
            use_timestamp: Whether to create timestamped run directory
            custom_run_name: Optional custom name for this run (overrides timestamp)
        """
        base_dir = Path(f"results/phase_{phase}")
        
        if custom_run_name:
            run_dir = custom_run_name
        elif use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = timestamp
        else:
            run_dir = "latest"
        
        self.output_dir = base_dir / run_dir
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.metrics_dir = self.output_dir / "metrics"
        self.reports_dir = self.output_dir / "reports"
        
        for d in [self.figures_dir, self.tables_dir, self.metrics_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Output directory: {self.output_dir}")
    
    def save_figure(
        self, 
        fig: plt.Figure, 
        name: str, 
        caption: str,
        chapter_section: str,
        dpi: int = 300
    ) -> Path:
        """
        Save figure with caption and metadata
        
        Args:
            fig: matplotlib figure
            name: filename without extension
            caption: figure caption for thesis
            chapter_section: e.g., "5.1" for Chapter 5, Section 1
            dpi: resolution (300+ for publication quality)
        
        Returns:
            Path to saved figure
        """
        fig_path = self.figures_dir / f"{name}.png"
        caption_path = self.figures_dir / f"{name}_caption.txt"
        metadata_path = self.figures_dir / f"{name}_metadata.json"
        
        # Save figure
        fig.savefig(fig_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        
        # Save caption
        with open(caption_path, 'w') as f:
            f.write(f"[Chapter {chapter_section}]\n\n")
            f.write(caption)
        
        # Save metadata
        metadata = {
            "filename": fig_path.name,
            "chapter_section": chapter_section,
            "caption": caption,
            "dpi": dpi,
            "timestamp": datetime.now().isoformat()
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved figure: {fig_path}")
        return fig_path
    
    def save_metrics(
        self, 
        name: str, 
        metrics: Dict[str, Any],
        interpretation: str,
        model_checkpoint: str,
        dataset: str,
        method: Optional[str] = None
    ) -> Path:
        """
        Save structured metrics as JSON with full context
        
        Args:
            name: analysis name
            metrics: dictionary of computed metrics
            interpretation: human-readable interpretation for thesis
            model_checkpoint: path to model used
            dataset: dataset identifier
            method: optional method description
        
        Returns:
            Path to saved metrics file
        """
        metrics_path = self.metrics_dir / f"{name}_metrics.json"
        
        output = {
            "analysis_name": name,
            "timestamp": datetime.now().isoformat(),
            "model_checkpoint": model_checkpoint,
            "dataset": dataset,
            "method": method,
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
        rows: List[List[Any]],
        headers: List[str],
        alignment: Optional[str] = None
    ) -> Path:
        """
        Save LaTeX table ready for thesis
        
        Args:
            name: table filename
            caption: table caption
            label: LaTeX label (e.g., "tab:attention_entropy")
            rows: list of row data
            headers: column headers
            alignment: column alignment (e.g., "lcccc"), auto-generated if None
        
        Returns:
            Path to saved table file
        """
        table_path = self.tables_dir / f"{name}.tex"
        
        if alignment is None:
            alignment = 'l' + 'c' * (len(headers) - 1)
        
        # Build tabular content
        header_row = " & ".join(headers) + " \\\\"
        data_rows = []
        for row in rows:
            row_str = " & ".join(str(cell) for cell in row) + " \\\\"
            data_rows.append(row_str)
        
        tabular_content = f"""\\begin{{tabular}}{{{alignment}}}
\\toprule
{header_row}
\\midrule
{chr(10).join(data_rows)}
\\bottomrule
\\end{{tabular}}"""
        
        full_latex = f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
{tabular_content}
\\label{{{label}}}
\\end{{table}}"""
        
        with open(table_path, 'w') as f:
            f.write(full_latex)
        
        print(f"✓ Saved table: {table_path}")
        return table_path
    
    def save_report(
        self, 
        name: str, 
        objective: str,
        method: str,
        key_findings: List[str],
        thesis_integration: Dict[str, str],
        files_generated: List[Path]
    ) -> Path:
        """
        Save markdown analysis report with standardized structure
        
        Args:
            name: analysis name
            objective: what the analysis aims to discover
            method: how the analysis was performed
            key_findings: numbered list of main results
            thesis_integration: dict mapping chapters to usage notes
            files_generated: list of output file paths
        
        Returns:
            Path to saved report
        """
        report_path = self.reports_dir / f"{name}_report.md"
        
        # Build findings section
        findings_text = "\n".join([f"{i+1}. {finding}" for i, finding in enumerate(key_findings)])
        
        # Build integration section
        integration_text = "\n".join([
            f"- **{chapter}**: {usage}" 
            for chapter, usage in thesis_integration.items()
        ])
        
        # Build files section
        files_text = "\n".join([f"- `{f}`" for f in files_generated])
        
        content = f"""# Analysis Report: {name.replace('_', ' ').title()}

## Objective
{objective}

## Method
{method}

## Key Findings
{findings_text}

## Thesis Integration
{integration_text}

## Files Generated
{files_text}

---
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(report_path, 'w') as f:
            f.write(content)
        
        print(f"✓ Saved report: {report_path}")
        return report_path


def format_p_value(p: float) -> str:
    """
    Format p-value for publication with significance stars
    
    Args:
        p: p-value
    
    Returns:
        LaTeX-formatted string with significance indicators
    """
    if p < 0.001:
        return "$<0.001^{***}$"
    elif p < 0.01:
        return f"${p:.3f}^{{**}}$"
    elif p < 0.05:
        return f"${p:.3f}^{{*}}$"
    else:
        return f"${p:.3f}$"


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size
    
    Args:
        group1: first group samples
        group2: second group samples
    
    Returns:
        Cohen's d statistic
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def statistical_comparison(
    group1: np.ndarray, 
    group2: np.ndarray,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    test: str = "t-test"
) -> Dict[str, Any]:
    """
    Perform statistical comparison between two groups
    
    Args:
        group1: first group samples
        group2: second group samples
        group1_name: label for group 1
        group2_name: label for group 2
        test: "t-test" or "mann-whitney"
    
    Returns:
        Dictionary with test results and descriptive statistics
    """
    # Descriptive stats
    stats_dict = {
        group1_name: {
            "mean": float(np.mean(group1)),
            "std": float(np.std(group1)),
            "median": float(np.median(group1)),
            "n": len(group1)
        },
        group2_name: {
            "mean": float(np.mean(group2)),
            "std": float(np.std(group2)),
            "median": float(np.median(group2)),
            "n": len(group2)
        }
    }
    
    # Statistical test
    if test == "t-test":
        statistic, p_value = stats.ttest_ind(group1, group2)
        test_name = "Independent t-test"
    elif test == "mann-whitney":
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        test_name = "Mann-Whitney U test"
    else:
        raise ValueError(f"Unknown test: {test}")
    
    # Effect size
    cohens_d = compute_cohens_d(group1, group2)
    
    stats_dict["statistical_test"] = {
        "test": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "p_value_formatted": format_p_value(p_value),
        "cohens_d": float(cohens_d),
        "significant": p_value < 0.05
    }
    
    return stats_dict


def create_thesis_figure_style():
    """
    Apply consistent matplotlib style for thesis figures
    """
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
    })
