"""Benchmark runner for Pallas kernel auto-tuning.

The runner handles the critical details of reliable GPU/TPU benchmarking:
- Warmup passes to trigger JIT compilation and cache filling
- Statistical timing with multiple repetitions
- jax.block_until_ready() to measure actual execution time (not dispatch time)
- Optional FLOPS and bandwidth computation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import numpy as np


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single kernel configuration.

    All timing values are in milliseconds.
    """

    config: dict[str, Any]
    median_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    all_times_ms: list[float] = field(default_factory=list)
    tflops: float | None = None
    bandwidth_gb_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dictionary for JSON/CSV export."""
        d = {f"config_{k}": v for k, v in self.config.items()}
        d.update({
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        })
        if self.tflops is not None:
            d["tflops"] = self.tflops
        if self.bandwidth_gb_s is not None:
            d["bandwidth_gb_s"] = self.bandwidth_gb_s
        return d


class BenchmarkRunner:
    """Runs benchmarks for kernel configurations with proper timing methodology.

    Args:
        kernel_fn: The kernel to benchmark. Called as kernel_fn(*inputs, **config).
        input_fn: Creates inputs for a given config. Called as input_fn(config) -> tuple of arrays.
        n_warmup: Number of warmup iterations (not timed).
        n_repeat: Number of timed iterations for statistics.
        flops_fn: Optional. Given a config dict, returns total FLOPs for one kernel call.
        bytes_fn: Optional. Given a config dict, returns total bytes accessed for one kernel call.
    """

    def __init__(
        self,
        kernel_fn: Callable,
        input_fn: Callable[[dict[str, Any]], tuple],
        *,
        n_warmup: int = 5,
        n_repeat: int = 20,
        flops_fn: Callable[[dict[str, Any]], int] | None = None,
        bytes_fn: Callable[[dict[str, Any]], int] | None = None,
    ):
        self.kernel_fn = kernel_fn
        self.input_fn = input_fn
        self.n_warmup = n_warmup
        self.n_repeat = n_repeat
        self.flops_fn = flops_fn
        self.bytes_fn = bytes_fn

    def run_single(self, config: dict[str, Any]) -> BenchmarkResult:
        """Benchmark a single configuration.

        Returns a BenchmarkResult with timing statistics.
        """
        inputs = self.input_fn(config)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)

        # Warmup: trigger JIT compilation, fill caches
        for _ in range(self.n_warmup):
            out = self.kernel_fn(*inputs, **config)
            jax.block_until_ready(out)

        # Timed runs
        times_ms = []
        for _ in range(self.n_repeat):
            start = time.perf_counter()
            out = self.kernel_fn(*inputs, **config)
            jax.block_until_ready(out)
            elapsed = (time.perf_counter() - start) * 1000.0
            times_ms.append(elapsed)

        times_arr = np.array(times_ms)
        median_ms = float(np.median(times_arr))

        # Compute throughput metrics
        tflops = None
        bandwidth_gb_s = None

        if self.flops_fn is not None:
            total_flops = self.flops_fn(config)
            tflops = (total_flops / 1e12) / (median_ms / 1000.0)

        if self.bytes_fn is not None:
            total_bytes = self.bytes_fn(config)
            bandwidth_gb_s = (total_bytes / 1e9) / (median_ms / 1000.0)

        return BenchmarkResult(
            config=config,
            median_ms=median_ms,
            mean_ms=float(np.mean(times_arr)),
            std_ms=float(np.std(times_arr)),
            min_ms=float(np.min(times_arr)),
            max_ms=float(np.max(times_arr)),
            all_times_ms=times_ms,
            tflops=tflops,
            bandwidth_gb_s=bandwidth_gb_s,
        )

    def run_all(
        self,
        configs: list[dict[str, Any]],
        *,
        verbose: bool = True,
    ) -> list[BenchmarkResult]:
        """Benchmark all configurations.

        Results are returned sorted by median time (fastest first).
        """
        results = []
        total = len(configs)

        for i, config in enumerate(configs):
            if verbose:
                print(f"[{i + 1}/{total}] Benchmarking: {config}")

            try:
                result = self.run_single(config)
                results.append(result)
                if verbose:
                    print(f"  -> median: {result.median_ms:.3f} ms")
            except Exception as e:
                if verbose:
                    print(f"  -> FAILED: {e}")

        # Sort by median time (fastest first)
        results.sort(key=lambda r: r.median_ms)
        return results
