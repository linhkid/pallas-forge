# pallas-forge

A lightweight auto-tuning framework for [Pallas](https://jax.readthedocs.io/en/latest/pallas/index.html) kernels on Google TPU.

**pallas-forge** helps you systematically find the right block-size configuration for a Pallas kernel — and honestly answer the question *"is my custom kernel actually beating XLA?"* It ships three reference kernels, a kernel-agnostic auto-tuner with proper warmup + timing discipline, and performance heatmaps, roofline charts, and XProf trace capture as first-class outputs.

<p align="center">
  <img src="images/heatmap_matmul.png" alt="MatMul block-size heatmap on TPU v5e — 3.56× spread between best and worst config" width="520">
  &nbsp;
  <img src="images/roofline.png" alt="Roofline chart — pallas-forge reference kernels on TPU v5e" width="520">
</p>
<p align="center"><em>Measured on TPU v5e via Colab. Reproduce with <a href="notebooks/05_reproduce_figures.ipynb"><code>notebooks/05_reproduce_figures.ipynb</code></a>.</em></p>

### What the measurements say

Running the three reference kernels against unfused XLA baselines on a Colab TPU v5e (2048² bf16 matmul, 4×2048×4096 RMSNorm, LLaMA-sized SwiGLU):

| Kernel | XLA (ms) | Pallas (ms) | Speedup |
|---|---:|---:|---:|
| **Fused RMSNorm + Residual** | 3.05 | 0.88 | **3.44×** ✅ |
| Tiled MatMul | 0.25 | 0.32 | 0.77× |
| Fused SwiGLU | 2.75 | 4.22 | 0.65× |

The honest finding: **custom Pallas kernels earn their complexity in fusion opportunities** — eliminating HBM round-trips that XLA can't avoid (RMSNorm+residual wins by 3.44×). For standard matmul, our teaching-quality kernel lands within ~23% of XLA's decade-hand-tuned baseline, which is as much as you can reasonably hope for. The block-size sweep over matmul shows a **3.56× spread** from worst to best config (1.14 ms → 0.32 ms), which is the library's core use case: finding that best config systematically instead of by hand.

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

## Architecture

<p align="center">
  <img src="images/architecture.png" alt="pallas-forge software stack" width="720">
</p>

Your script calls pallas-forge; the kernels use `jax.experimental.pallas`, which lowers to Mosaic on TPU (or Triton on GPU). The tuner wraps everything in warm-up + statistical timing + optional XProf trace capture.

## Installation

### From PyPI (once published)

```bash
pip install pallas-forge          # CPU interpret mode
pip install "pallas-forge[viz]"   # with matplotlib/seaborn for heatmaps
pip install "pallas-forge[tpu]"   # on a Linux TPU VM
```

### From source

```bash
git clone https://github.com/nklinh91/pallas-forge
cd pallas-forge

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

### Speedup vs XLA baseline

<p align="center">
  <img src="images/speedup_vs_xla.png" alt="Pallas-forge vs XLA baseline (measured on TPU v5e)" width="820">
</p>

**Measured on TPU v5e (via Colab).** The three regimes:

- **Fused RMSNorm + Residual wins big: 3.44×** — the fusion eliminates an HBM
  round-trip on an intermediate that XLA can't easily avoid.
- **Tiled MatMul is nearly at parity (0.77×)** — XLA's matmul has had years of
  hand-tuning; our teaching-quality kernel lands within ~23%, which is as close
  as a from-scratch implementation reasonably gets.
- **Fused SwiGLU loses (0.65×)** — XLA already fuses the gate+activation+multiply
  pretty effectively; custom Pallas here isn't obviously worth the complexity.

Reproduce with [`notebooks/05_reproduce_figures.ipynb`](notebooks/05_reproduce_figures.ipynb)
on a Colab TPU runtime (takes ~5-10 minutes).

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
| `05_reproduce_figures.ipynb` | Regenerate all blog/README figures from real TPU measurements |

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

## Contributing & releases

- **Contributing**: open an issue or PR. Tests (`pytest tests/`) must pass on CPU.
- **Pre-commit hooks**: after `pip install -e ".[dev]"`, run `pre-commit install` once.
  Ruff and basic hygiene checks will then run on every commit, so you won't push lint-failing code.
- **Publishing**: see [`PUBLISHING.md`](PUBLISHING.md) for the PyPI release workflow (automated via GitHub Actions + Trusted Publishing).
- **Changelog**: see [`CHANGELOG.md`](CHANGELOG.md).

## License

MIT
