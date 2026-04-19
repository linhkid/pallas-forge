"""Profiling utilities for Pallas kernels.

Provides roofline charts and hardware utilization analysis, with presets for
common TPU generations (v4, v5e, v5p).

Example::

    from pallas_forge.profile import roofline_chart, TPU_SPECS

    tpu = TPU_SPECS["v4"]
    roofline_chart(
        results,
        peak_tflops=tpu["peak_tflops_bf16"],
        peak_bandwidth_gb_s=tpu["peak_bandwidth_gb_s"],
        save_path="roofline.png",
    )
"""

from pallas_forge.profile.analysis import (
    TPU_SPECS,
    classify_boundedness,
    compute_bandwidth_utilization,
    compute_mxu_utilization,
    compute_operational_intensity,
)
from pallas_forge.profile.roofline import roofline_chart

__all__ = [
    "roofline_chart",
    "compute_operational_intensity",
    "compute_mxu_utilization",
    "compute_bandwidth_utilization",
    "classify_boundedness",
    "TPU_SPECS",
]
