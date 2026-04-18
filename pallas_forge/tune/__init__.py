"""Auto-tuning framework for Pallas kernels.

The `tune()` function is the main entry point. It accepts any Pallas kernel,
a config space, and returns a TuneReport with timing results and visualizations.

Example::

    from pallas_forge.tune import tune, TuneConfig

    config = TuneConfig.from_dict({
        "block_m": [64, 128, 256],
        "block_n": [64, 128, 256],
    })

    report = tune(
        kernel_fn=my_kernel,
        input_fn=lambda cfg: (jax.random.normal(key, (1024, 1024)),),
        config=config,
    )
    report.heatmap("block_m", "block_n", save_path="heatmap.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pallas_forge.tune.config import TuneConfig
from pallas_forge.tune.report import TuneReport
from pallas_forge.tune.runner import BenchmarkResult, BenchmarkRunner
from pallas_forge.tune.search import GridSearch, RandomSearch, SearchStrategy


def tune(
    kernel_fn: Callable,
    input_fn: Callable[[dict[str, Any]], tuple],
    config: TuneConfig | dict | str | Path,
    *,
    strategy: str | SearchStrategy = "grid",
    n_warmup: int = 5,
    n_repeat: int = 20,
    flops_fn: Callable[[dict[str, Any]], int] | None = None,
    bytes_fn: Callable[[dict[str, Any]], int] | None = None,
    top_n_traces: int = 0,
    trace_output_dir: str = "/tmp/xprof_traces",
    verbose: bool = True,
) -> TuneReport:
    """Auto-tune a Pallas kernel over a configuration space.

    This is the single entry point for the auto-tuner. It:
    1. Normalizes the config into a TuneConfig
    2. Generates configurations via the search strategy
    3. Benchmarks each configuration with proper warmup and timing
    4. Optionally captures XProf traces for the top-N configs
    5. Returns a TuneReport with results and visualization methods

    Args:
        kernel_fn: The kernel to benchmark. Called as kernel_fn(*inputs, **config).
        input_fn: Creates inputs for a given config. Called as input_fn(config) -> tuple.
        config: Search space. Can be a TuneConfig, dict, or path to YAML file.
        strategy: Search strategy. "grid" for exhaustive, "random" for random sampling,
            or a SearchStrategy instance.
        n_warmup: Warmup iterations per config (not timed).
        n_repeat: Timed iterations per config.
        flops_fn: Given a config, returns total FLOPs per call (for TFLOPS calculation).
        bytes_fn: Given a config, returns total bytes accessed per call (for bandwidth).
        top_n_traces: Number of top configs to capture XProf traces for. 0 = no traces.
        trace_output_dir: Directory for XProf trace output.
        verbose: Print progress during benchmarking.

    Returns:
        TuneReport with results sorted by median time (fastest first).
    """
    # Normalize config
    if isinstance(config, dict):
        config = TuneConfig.from_dict(config)
    elif isinstance(config, (str, Path)):
        config = TuneConfig.from_yaml(config)

    # Normalize strategy
    if isinstance(strategy, str):
        if strategy == "grid":
            strategy = GridSearch()
        elif strategy == "random":
            strategy = RandomSearch()
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'grid' or 'random'.")

    # Generate configs
    configs = strategy.generate(config)
    if verbose:
        print(f"Auto-tuning {len(configs)} configurations...")

    # Run benchmarks
    runner = BenchmarkRunner(
        kernel_fn,
        input_fn,
        n_warmup=n_warmup,
        n_repeat=n_repeat,
        flops_fn=flops_fn,
        bytes_fn=bytes_fn,
    )
    results = runner.run_all(configs, verbose=verbose)

    # Create report
    report = TuneReport(results, param_names=config.param_names)

    if verbose and results:
        best = report.best(1)[0]
        print(f"\nBest config: {best.config} -> {best.median_ms:.3f} ms")
        print(f"Speedup range: {report.speedup_range:.1f}x")

    # Optional XProf traces
    if top_n_traces > 0 and results:
        from pallas_forge.tune.trace import capture_top_n_traces

        if verbose:
            print(f"\nCapturing XProf traces for top {top_n_traces} configs...")
        capture_top_n_traces(
            results,
            kernel_fn,
            input_fn,
            n=top_n_traces,
            output_dir=trace_output_dir,
        )

    return report


__all__ = [
    "tune",
    "TuneConfig",
    "TuneReport",
    "BenchmarkResult",
    "BenchmarkRunner",
    "GridSearch",
    "RandomSearch",
]
