"""Results reporting and visualization for auto-tuning.

Generates JSON/CSV exports and performance heatmaps from benchmark results.
The heatmaps are the visual centerpiece — they show how each tuning parameter
affects throughput, making the "3-5x swing" claim tangible.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from pallas_forge.tune.runner import BenchmarkResult


class TuneReport:
    """Aggregates benchmark results and generates reports and visualizations.

    Args:
        results: List of BenchmarkResult from a tuning sweep.
        param_names: Names of the tuned parameters (for axis labels).
    """

    def __init__(self, results: list[BenchmarkResult], param_names: list[str] | None = None):
        self.results = results
        if param_names is None and results:
            param_names = list(results[0].config.keys())
        self.param_names = param_names or []

    def best(self, n: int = 1) -> list[BenchmarkResult]:
        """Return the top-n fastest configurations."""
        sorted_results = sorted(self.results, key=lambda r: r.median_ms)
        return sorted_results[:n]

    def worst(self, n: int = 1) -> list[BenchmarkResult]:
        """Return the n slowest configurations."""
        sorted_results = sorted(self.results, key=lambda r: r.median_ms, reverse=True)
        return sorted_results[:n]

    @property
    def speedup_range(self) -> float:
        """Ratio of slowest to fastest median time."""
        if not self.results:
            return 1.0
        times = [r.median_ms for r in self.results]
        return max(times) / max(min(times), 1e-9)

    def to_json(self, path: str | Path) -> None:
        """Export results to JSON."""
        data = [r.to_dict() for r in self.results]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def to_csv(self, path: str | Path) -> None:
        """Export results to CSV.

        Fieldnames are built from the union of keys across all rows, preserving
        first-seen order. This means heterogeneous configs (e.g. after a
        mid-analysis mutation, or concatenated results from multiple tune()
        runs) export cleanly instead of raising a ValueError.
        """
        if not self.results:
            return
        rows = [r.to_dict() for r in self.results]
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def heatmap(
        self,
        x_param: str,
        y_param: str,
        metric: str = "median_ms",
        *,
        title: str | None = None,
        save_path: str | Path | None = None,
        figsize: tuple[int, int] = (10, 8),
        cmap: str = "YlOrRd_r",
        annotate: bool = True,
    ):
        """Generate a 2D heatmap showing how two parameters affect performance.

        For configs with additional parameters beyond x_param and y_param,
        the best (lowest) metric value is used for each (x, y) cell.

        Args:
            x_param: Parameter name for x-axis.
            y_param: Parameter name for y-axis.
            metric: Metric to plot. One of: "median_ms", "tflops", "bandwidth_gb_s".
            title: Plot title. Auto-generated if None.
            save_path: Save figure to this path. Shows interactively if None.
            figsize: Figure size in inches.
            cmap: Matplotlib colormap. Default "YlOrRd_r" (red=slow, yellow=fast).
            annotate: Whether to annotate cells with values.

        Returns:
            matplotlib Figure object.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if x_param not in self.param_names or y_param not in self.param_names:
            raise ValueError(
                f"Parameters must be in {self.param_names}, got x={x_param}, y={y_param}"
            )

        # Collect unique values for each axis
        x_vals = sorted(set(r.config[x_param] for r in self.results))
        y_vals = sorted(set(r.config[y_param] for r in self.results))

        # Build 2D grid: for each (x, y) cell, take the best metric value
        grid = np.full((len(y_vals), len(x_vals)), np.nan)
        x_idx = {v: i for i, v in enumerate(x_vals)}
        y_idx = {v: i for i, v in enumerate(y_vals)}

        # Group results by (x, y) and take best
        cells: dict[tuple, list[float]] = defaultdict(list)
        for r in self.results:
            key = (r.config[x_param], r.config[y_param])
            val = getattr(r, metric, None)
            if val is None:
                val = r.to_dict().get(metric)
            if val is not None:
                cells[key].append(val)

        for (xv, yv), vals in cells.items():
            # For time metrics, lower is better; for throughput, higher is better
            if metric in ("tflops", "bandwidth_gb_s"):
                grid[y_idx[yv], x_idx[xv]] = max(vals)
            else:
                grid[y_idx[yv], x_idx[xv]] = min(vals)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(grid, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, label=metric)

        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([str(v) for v in x_vals])
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([str(v) for v in y_vals])
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)

        if title is None:
            title = f"{metric} by {x_param} vs {y_param}"
        ax.set_title(title)

        # Annotate cells
        if annotate:
            for yi in range(len(y_vals)):
                for xi in range(len(x_vals)):
                    val = grid[yi, xi]
                    if not np.isnan(val):
                        text = f"{val:.2f}"
                        ax.text(xi, yi, text, ha="center", va="center", fontsize=8)

        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
