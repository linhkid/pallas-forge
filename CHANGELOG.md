# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] — 2026-04-20

Second TPU-validated release. 0.1.1 shipped a matmul kernel that ran on TPU
but hit two latent issues when integrated end-to-end: the RMSNorm kernel
failed the same sublane rule we'd fixed for matmul, and every kernel call
paid ~40 ms of Python/lowering overhead because the outer wrappers weren't
JIT-compiled. 0.1.2 fixes both.

### Fixed
- **`@jax.jit` on every kernel wrapper.** `tiled_matmul`, `fused_swiglu`,
  `fused_geglu`, and `fused_rmsnorm_residual` are now all decorated with
  `@jax.jit` (static args: block sizes + eps as appropriate). First call with
  a given config/shape combination traces and lowers once; subsequent calls
  hit JAX's cache. On TPU v5e this drops per-call latency from ~40 ms of
  Python overhead to the actual kernel time (single-digit ms for typical
  sizes). Benchmark results in 0.1.1 were dominated by this overhead.
- **RMSNorm now runs on real TPU.** The kernel used `BlockSpec((1, dim), ...)`
  which violates the TPU sublane alignment rule (second-to-last block dim
  must be divisible by 8). Rewrote to process 8 tokens per kernel instance
  (`BlockSpec((8, dim), ...)`) with the wrapper padding the token axis up to
  a multiple of 8 and slicing back.
- **`TuneReport.to_csv` handles heterogeneous rows.** When results have
  different config keys (after mutation, or concatenated from multiple
  `tune()` runs), the writer now takes the union of fieldnames across rows
  instead of raising `ValueError: dict contains fields not in fieldnames`.

### Changed
- Notebook block-size choices bumped so every documented config satisfies
  the TPU 128-lane rule.
- Notebooks pinned to `@v0.1.2` in the Colab install cell.
- Blog Part 3 rewritten with real TPU v5e measurements — the honest finding
  is that custom Pallas kernels beat XLA where *fusion* matters
  (RMSNorm+residual: 3.78× faster than the unfused XLA equivalent), and lose
  where XLA already has a hand-tuned kernel (matmul, SwiGLU). That nuance
  is the interesting story.
- README speedup caption switched from "synthetic illustrative numbers" to
  the measured v5e values.

### Known issues
- `dim < 128` only works in CPU interpret mode; on TPU the last-block-dim
  rule will reject it. Tests use `dim = 64` which passes on CPU but not on
  TPU. A future release will add dim-padding inside the RMSNorm wrapper.

## [0.1.1] — 2026-04-20

First TPU-hardware-validated release. 0.1.0 passed all CPU interpret-mode tests
but the matmul kernel failed on real TPU hardware; 0.1.1 fixes that.

### Fixed
- **`tiled_matmul` now runs on real TPU hardware.** The 0.1.0 kernel used
  `jax.lax.dynamic_slice` inside a `fori_loop` for K-reduction, which works
  under CPU interpret mode but is unsupported by the Pallas TPU tensor-core
  lowering. Rewrote to use the canonical 3D-grid pattern
  `(grid_m, grid_n, k_tiles)` with `pl.when(program_id(2) == 0)` resetting the
  accumulator on the first K tile.
- **Accumulator moved to fp32 across the K axis.** Previously the per-K-tile
  cast to the output dtype caused precision loss for bf16 inputs. The kernel
  now emits fp32 and the wrapper casts once at the end.
- **`num_stages` forward-compatible.** Newer JAX moved this knob from a direct
  `pallas_call` kwarg into `compiler_params`. `_compat.py` now inspects the
  installed JAX and forwards `num_stages` via `pltpu.CompilerParams` where
  supported, silently drops it otherwise.

### Added
- `notebooks/05_reproduce_figures.ipynb` — regenerate all blog/README figures
  from real TPU measurements on a Colab runtime.
- `scripts/generate_example_images.py` — synthesises placeholder images for
  the README and blog using the library's own `TuneReport.heatmap` and
  `roofline_chart` functions.
- `images/` — 7 placeholder PNGs (architecture diagram, matmul heatmaps,
  roofline, speedup, BlockSpec tiling, 8×128 array layout) with
  `images/README.md` documenting their synthetic provenance.
- `blog/` — three-part article series (~5,700 words) on TPU hardware, writing
  your first Pallas kernel, and auto-tuning with pallas-forge.
- `.github/workflows/ci.yml` and `publish.yml` — CI on every push, PyPI
  publishing via OIDC Trusted Publishing on `v*.*.*` tag push.
- `PUBLISHING.md` — PyPI release playbook.
- `.pre-commit-config.yaml` — ruff + hygiene hooks.

### Changed
- `pallas_forge.profile` now re-exports `roofline_chart`, `TPU_SPECS`, and
  utilisation helpers at the package level so users don't need deep imports.
- README restructured: added architecture diagram, split install into PyPI vs
  source, added roofline/XProf code examples, added TPU generations table.
- Notebook Part 2 expanded with a sublane/lane explainer backed by the new
  `array_layout.png` diagram.

### Known issues
- `block_k` and `block_n` below 128 are silently rejected by the TPU lowering.
  The notebook sweep already filters these out; the tuner surfaces the error
  per-config without aborting the sweep.

## [0.1.0] — 2026-04-19

Initial release.

### Added
- Three reference Pallas kernels:
  - `tiled_matmul` — blocked matrix multiplication with K-reduction via `fori_loop`
  - `fused_rmsnorm_residual` — fused RMSNorm + residual addition (memory-bound)
  - `fused_swiglu` / `fused_geglu` — fused gated linear unit activations
- Auto-tuning framework (`pallas_forge.tune`):
  - `TuneConfig` search space (dict + YAML)
  - `GridSearch`, `RandomSearch` strategies
  - `BenchmarkRunner` with proper warmup, `block_until_ready`, and statistical timing
  - `TuneReport` with JSON/CSV export and matplotlib heatmaps
  - Top-level `tune()` entry point
  - XProf trace capture for top-N configurations
- Profiling utilities (`pallas_forge.profile`):
  - `roofline_chart` — operational intensity vs attained throughput
  - `classify_boundedness`, `compute_operational_intensity`, etc.
  - `TPU_SPECS` — hardware presets for v4, v5e, v5p
- CPU/TPU compatibility shim (`_compat.py`) enabling all kernels to run on CPU via Pallas interpret mode
- 54 unit tests covering all kernels and the tuning framework (all pass on CPU)
- Four Jupyter notebooks with Colab TPU setup cells
- Three benchmark scripts (`bench_matmul.py`, `bench_rmsnorm.py`, `bench_swiglu.py`) with XLA baseline comparison
- Three-part blog series in `blog/`

[Unreleased]: https://github.com/nklinh91/pallas-forge/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/nklinh91/pallas-forge/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/nklinh91/pallas-forge/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nklinh91/pallas-forge/releases/tag/v0.1.0
