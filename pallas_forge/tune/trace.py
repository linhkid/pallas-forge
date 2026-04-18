"""XProf trace capture for auto-tuning.

Wraps JAX's profiler to capture hardware traces for the top-N kernel
configurations. These traces can be loaded in TensorBoard for detailed
MXU/VPU/memory utilization analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import jax


def capture_xprof_trace(
    fn: Callable,
    *args,
    output_dir: str = "/tmp/xprof_traces",
    trace_name: str | None = None,
    n_calls: int = 5,
) -> str:
    """Capture an XProf trace of a function execution.

    Runs the function multiple times under the JAX profiler and saves
    the trace to output_dir.

    Args:
        fn: Function to profile.
        *args: Arguments to pass to fn.
        output_dir: Directory to save trace files.
        trace_name: Subdirectory name for this trace. Auto-generated if None.
        n_calls: Number of times to call fn during tracing.

    Returns:
        Path to the trace directory.
    """
    trace_dir = Path(output_dir)
    if trace_name:
        trace_dir = trace_dir / trace_name
    trace_dir.mkdir(parents=True, exist_ok=True)

    trace_path = str(trace_dir)

    jax.profiler.start_trace(trace_path)
    try:
        for _ in range(n_calls):
            out = fn(*args)
            jax.block_until_ready(out)
    finally:
        jax.profiler.stop_trace()

    return trace_path


def capture_top_n_traces(
    results: list,
    kernel_fn: Callable,
    input_fn: Callable,
    *,
    n: int = 3,
    output_dir: str = "/tmp/xprof_traces",
) -> list[str]:
    """Capture XProf traces for the top-n fastest configurations.

    Args:
        results: List of BenchmarkResult, sorted by performance.
        kernel_fn: The kernel function to profile.
        input_fn: Creates inputs given a config dict.
        n: Number of top configs to trace.
        output_dir: Base directory for traces.

    Returns:
        List of trace directory paths.
    """
    trace_paths = []
    for i, result in enumerate(results[:n]):
        config = result.config
        inputs = input_fn(config)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)

        def run_kernel():
            return kernel_fn(*inputs, **config)

        config_str = "_".join(f"{k}{v}" for k, v in config.items())
        trace_name = f"rank{i}_{config_str}"

        path = capture_xprof_trace(
            run_kernel,
            output_dir=output_dir,
            trace_name=trace_name,
        )
        trace_paths.append(path)

    return trace_paths
