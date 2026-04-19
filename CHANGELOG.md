# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/nklinh91/pallas-forge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nklinh91/pallas-forge/releases/tag/v0.1.0
