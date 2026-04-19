# pallas-forge

A lightweight auto-tuning framework for [Pallas](https://jax.readthedocs.io/en/latest/pallas/index.html) kernels on Google TPU.

**pallas-forge** helps you systematically discover *why* some kernel configurations outperform others by 3-5x, producing performance heatmaps, XProf profiler traces, and roofline charts.

## Features

- **3 reference kernels** with progressive complexity:
  - Tiled MatMul (compute-bound, MXU-dominated)
  - Fused RMSNorm + Residual (memory-bound, VPU-dominated)
  - Fused SwiGLU / GeGLU (compute-bound, fused activation)
- **Auto-tuning framework** — grid search and random search over block sizes, grid dims, pipeline stages
- **Performance heatmaps** — visual proof of how parameter choices affect throughput
- **XProf integration** — capture and compare profiler traces for top configurations
- **Roofline analysis** — classify kernels as compute-bound or memory-bound
- **CPU testing** — all kernels run on CPU via Pallas interpret mode (no TPU required for development)

## Installation

```bash
# Basic install (CPU testing via interpret mode)
pip install -e .

# With development tools + visualization
pip install -e ".[dev,viz]"

# On a Linux TPU VM (libtpu is Linux-only)
pip install -e ".[all,tpu]"
```

> **Tip:** Create a dedicated conda env first to avoid polluting your base environment:
> ```bash
> conda create -n pallas-forge python=3.11 -y
> conda activate pallas-forge
> pip install -e ".[dev,viz]"
> ```

## Quick Start

### Auto-tune a kernel

```python
import jax
import jax.numpy as jnp
from pallas_forge import tiled_matmul
from pallas_forge.tune import tune, TuneConfig

# Define config space
config = TuneConfig.from_dict({
    "block_m": [64, 128, 256],
    "block_k": [64, 128],
    "block_n": [64, 128, 256],
})

# Define inputs
M, K, N = 2048, 2048, 2048
def input_fn(cfg):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (M, K), dtype=jnp.bfloat16)
    w = jax.random.normal(key, (K, N), dtype=jnp.bfloat16)
    return (x, w)

# Define kernel wrapper
def kernel_fn(x, w, *, block_m, block_k, block_n):
    return tiled_matmul(x, w, block_m=block_m, block_k=block_k, block_n=block_n)

# Run auto-tuning
report = tune(kernel_fn, input_fn, config)

# Export results
report.to_csv("results.csv")
report.heatmap("block_m", "block_n", save_path="heatmap.png")

# Best config
best = report.best(1)[0]
print(f"Best: {best.config} -> {best.median_ms:.3f} ms")
print(f"Speedup range: {report.speedup_range:.1f}x")
```

### Use a kernel directly

```python
from pallas_forge import tiled_matmul, fused_rmsnorm_residual, fused_swiglu

# MatMul
result = tiled_matmul(x, w, block_m=128, block_k=128, block_n=128)

# Fused RMSNorm + Residual
normed, new_residual = fused_rmsnorm_residual(x, residual, weight)

# Fused SwiGLU
output = fused_swiglu(x, w_gate, w_up, block_m=128, block_n=256)
```

### Roofline analysis

```python
from pallas_forge.profile import roofline_chart, TPU_SPECS

tpu = TPU_SPECS["v4"]  # or "v5e", "v5p"
roofline_chart(
    report.results,
    peak_tflops=tpu["peak_tflops_bf16"],
    peak_bandwidth_gb_s=tpu["peak_bandwidth_gb_s"],
    save_path="roofline.png",
)
```

Requires `flops_fn` and `bytes_fn` to be passed to `tune()` so TFLOPS and bandwidth get populated.

### XProf trace capture for top configurations

```python
report = tune(
    kernel_fn, input_fn, config,
    top_n_traces=3,
    trace_output_dir="./xprof_traces",
)
# Open traces in TensorBoard:  tensorboard --logdir ./xprof_traces
```

## Project Structure

```
pallas_forge/
  _compat.py          # CPU/TPU compatibility (interpret mode)
  kernels/
    matmul.py          # Tiled MatMul
    rmsnorm.py         # Fused RMSNorm + Residual
    swiglu.py          # Fused SwiGLU / GeGLU
  tune/
    __init__.py        # tune() entry point
    config.py          # TuneConfig search space
    search.py          # GridSearch, RandomSearch
    runner.py          # BenchmarkRunner with proper timing
    report.py          # JSON/CSV export + heatmaps
    trace.py           # XProf trace capture
  profile/
    roofline.py        # Roofline chart generation
    analysis.py        # Utilization analysis
benchmarks/            # Benchmark scripts per kernel
tests/                 # Correctness tests (run on CPU)
```

## Running Tests

```bash
# All tests (run on CPU via Pallas interpret mode — no TPU required)
pytest tests/ -v
```

All 54 tests should pass on CPU. Tests marked `@requires_tpu` are skipped
automatically when no TPU is detected.

## Running Benchmarks (requires TPU)

```bash
# Individual kernel
python benchmarks/bench_matmul.py

# All benchmarks
python benchmarks/run_all.py
```

Outputs (CSV/JSON results + PNG heatmaps) land in `results/`.

## Notebooks

Interactive walkthroughs in `notebooks/` — each includes a Colab setup cell so
you can run them on a free TPU runtime.

| Notebook | Focus |
|---|---|
| `01_tiled_matmul.ipynb` | BlockSpec, grid, block sizes, bfloat16 |
| `02_fused_rmsnorm.ipynb` | Kernel fusion, VPU vs MXU, HBM traffic |
| `03_swiglu_geglu.ipynb` | Compute-bound fusion, SwiGLU vs GeGLU |
| `04_auto_tuning.ipynb` | `tune()`, heatmaps, random search, YAML configs |

[//]: # ()
[//]: # (## Blog series)

[//]: # ()
[//]: # (Long-form articles accompanying the code, in `blog/`:)

[//]: # ()
[//]: # (1. [Why Pallas? Google's TPU needs its own kernel language]&#40;blog/01_why_pallas.md&#41; — the landscape)

[//]: # (2. [Your first Pallas kernel: Tiled MatMul on TPU]&#40;blog/02_first_kernel.md&#41; — hands-on walkthrough)

[//]: # (3. [The 3-5× problem: Auto-tuning Pallas kernels with pallas-forge]&#40;blog/03_autotuning_with_pallas_forge.md&#41; — this library)

[//]: # ()
[//]: # (See [`blog/README.md`]&#40;blog/README.md&#41; for the publication guide.)

## Supported TPU generations

`pallas_forge.profile.TPU_SPECS` provides hardware presets for roofline analysis:

| Generation | Peak bf16 TFLOPS | Peak HBM GB/s | VMEM (MB) |
|---|---|---|---|
| v4 | 275 | 1200 | 32 |
| v5e | 197 | 819 | 32 |
| v5p | 459 | 2765 | 95 |

## License

MIT
