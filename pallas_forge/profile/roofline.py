"""Roofline chart generation for Pallas kernel analysis.

The roofline model plots operational intensity (FLOPS/byte) vs attained performance
(GFLOPS/s), with hardware ceilings for peak compute and peak memory bandwidth.
This reveals whether each kernel configuration is compute-bound or memory-bound.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def roofline_chart(
    results: list,
    peak_tflops: float,
    peak_bandwidth_gb_s: float,
    *,
    title: str = "Roofline Analysis",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 7),
    labels: list[str] | None = None,
):
    """Generate a roofline chart from benchmark results.

    Each result must have `tflops` and `bandwidth_gb_s` attributes (computed
    by the BenchmarkRunner when flops_fn and bytes_fn are provided).

    Args:
        results: List of BenchmarkResult with tflops and bandwidth_gb_s.
        peak_tflops: Hardware peak TFLOPS (e.g., 275 for TPU v4).
        peak_bandwidth_gb_s: Hardware peak HBM bandwidth in GB/s (e.g., 1200 for TPU v4).
        title: Chart title.
        save_path: Save to file. Shows interactively if None.
        figsize: Figure size in inches.
        labels: Optional labels for each result point.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    # Ridge point: where compute ceiling meets memory ceiling
    # operational_intensity = peak_FLOPS / peak_bandwidth
    peak_flops = peak_tflops * 1e12
    peak_bw = peak_bandwidth_gb_s * 1e9
    ridge_point = peak_flops / peak_bw  # FLOPS/byte

    # Generate roofline ceiling
    oi_range = np.logspace(-1, 4, 500)  # operational intensity range
    ceiling = np.minimum(peak_flops / 1e9, oi_range * peak_bw / 1e9)  # GFLOPS

    fig, ax = plt.subplots(figsize=figsize)

    # Plot roofline ceiling
    ax.loglog(oi_range, ceiling, "k-", linewidth=2, label="Roofline ceiling")

    # Plot memory-bound and compute-bound regions
    mem_bound_oi = oi_range[oi_range < ridge_point]
    comp_bound_oi = oi_range[oi_range >= ridge_point]
    ax.fill_between(
        mem_bound_oi,
        0.1,
        np.minimum(peak_flops / 1e9, mem_bound_oi * peak_bw / 1e9),
        alpha=0.1,
        color="blue",
        label="Memory-bound region",
    )
    ax.fill_between(
        comp_bound_oi,
        0.1,
        np.full_like(comp_bound_oi, peak_flops / 1e9),
        alpha=0.1,
        color="red",
        label="Compute-bound region",
    )

    # Plot ridge point
    ax.axvline(x=ridge_point, color="gray", linestyle="--", alpha=0.5)
    ax.annotate(
        f"Ridge: {ridge_point:.1f} FLOP/B",
        xy=(ridge_point, peak_flops / 1e9),
        xytext=(ridge_point * 2, peak_flops / 1e9 * 0.5),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9,
        color="gray",
    )

    # Plot kernel data points
    for i, r in enumerate(results):
        if r.tflops is None or r.bandwidth_gb_s is None:
            continue

        attained_gflops = r.tflops * 1e3  # TFLOPS -> GFLOPS
        # Operational intensity = FLOPS / bytes
        total_flops = r.tflops * 1e12 * (r.median_ms / 1000.0)
        total_bytes = r.bandwidth_gb_s * 1e9 * (r.median_ms / 1000.0)
        oi = total_flops / max(total_bytes, 1)

        label = labels[i] if labels else str(r.config)
        ax.plot(oi, attained_gflops, "o", markersize=8, label=label[:30])

    ax.set_xlabel("Operational Intensity (FLOP/Byte)")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=0.1)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
