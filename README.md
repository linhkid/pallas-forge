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
# Basic install
pip install -e .

# With development tools
pip install -e ".[dev,viz]"

# On TPU VM
pip install -e ".[all,tpu]"
```

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
# All tests (runs on CPU via interpret mode)
pytest tests/ -v

# Skip slow tests
pytest tests/ -v -k "not slow"
```

## Running Benchmarks (requires TPU)

```bash
# Individual kernel
python benchmarks/bench_matmul.py

# All benchmarks
python benchmarks/run_all.py
```

## License

MIT
