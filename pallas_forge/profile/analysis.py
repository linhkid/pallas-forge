"""Hardware utilization analysis helpers.

Provides functions to classify kernel boundedness and compute utilization
metrics from benchmark results.
"""

from __future__ import annotations


def compute_operational_intensity(total_flops: int, total_bytes: int) -> float:
    """Compute operational intensity: FLOPS per byte accessed.

    Higher values indicate more compute per byte of data movement.
    The ridge point of the roofline determines the boundary between
    memory-bound and compute-bound.
    """
    if total_bytes == 0:
        return float("inf")
    return total_flops / total_bytes


def compute_mxu_utilization(attained_tflops: float, peak_tflops: float) -> float:
    """Compute MXU utilization as a percentage of peak.

    Args:
        attained_tflops: Measured throughput in TFLOPS.
        peak_tflops: Hardware peak TFLOPS (e.g., 275 for TPU v4).

    Returns:
        Utilization percentage (0-100).
    """
    if peak_tflops <= 0:
        return 0.0
    return min(100.0, (attained_tflops / peak_tflops) * 100.0)


def compute_bandwidth_utilization(
    attained_gb_s: float,
    peak_gb_s: float,
) -> float:
    """Compute memory bandwidth utilization as a percentage of peak.

    Args:
        attained_gb_s: Measured bandwidth in GB/s.
        peak_gb_s: Hardware peak HBM bandwidth in GB/s.

    Returns:
        Utilization percentage (0-100).
    """
    if peak_gb_s <= 0:
        return 0.0
    return min(100.0, (attained_gb_s / peak_gb_s) * 100.0)


def classify_boundedness(
    operational_intensity: float,
    peak_tflops: float,
    peak_bandwidth_gb_s: float,
) -> str:
    """Classify a kernel as compute-bound or memory-bound.

    Uses the roofline model: if operational intensity is below the ridge point,
    the kernel is memory-bound; otherwise it's compute-bound.

    Args:
        operational_intensity: FLOPS per byte.
        peak_tflops: Hardware peak TFLOPS.
        peak_bandwidth_gb_s: Hardware peak HBM bandwidth in GB/s.

    Returns:
        "compute-bound" or "memory-bound".
    """
    # Ridge point = peak_FLOPS / peak_bandwidth
    peak_flops = peak_tflops * 1e12
    peak_bw = peak_bandwidth_gb_s * 1e9
    ridge_point = peak_flops / peak_bw

    if operational_intensity >= ridge_point:
        return "compute-bound"
    return "memory-bound"


# TPU hardware specs for common generations
TPU_SPECS = {
    "v4": {
        "peak_tflops_bf16": 275.0,
        "peak_bandwidth_gb_s": 1200.0,
        "vmem_mb": 32,
        "mxu_size": 128,
    },
    "v5e": {
        "peak_tflops_bf16": 197.0,
        "peak_bandwidth_gb_s": 819.0,
        "vmem_mb": 32,
        "mxu_size": 128,
    },
    "v5p": {
        "peak_tflops_bf16": 459.0,
        "peak_bandwidth_gb_s": 2765.0,
        "vmem_mb": 95,
        "mxu_size": 128,
    },
}
