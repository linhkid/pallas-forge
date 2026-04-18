"""pallas-forge: Auto-tuning framework for Pallas kernels on Google TPU."""

from pallas_forge._version import __version__
from pallas_forge.kernels.matmul import tiled_matmul
from pallas_forge.kernels.rmsnorm import fused_rmsnorm_residual
from pallas_forge.kernels.swiglu import fused_geglu, fused_swiglu
from pallas_forge.tune import tune
from pallas_forge.tune.config import TuneConfig
from pallas_forge.tune.runner import BenchmarkResult

__all__ = [
    "__version__",
    "tiled_matmul",
    "fused_rmsnorm_residual",
    "fused_swiglu",
    "fused_geglu",
    "tune",
    "TuneConfig",
    "BenchmarkResult",
]
