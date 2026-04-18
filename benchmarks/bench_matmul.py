"""Benchmark script for Tiled MatMul kernel.

Compares the Pallas tiled matmul against JAX's default jnp.matmul (XLA baseline)
across a grid of block size configurations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pallas_forge.kernels.matmul import tiled_matmul
from pallas_forge.tune import TuneConfig, tune


# Problem size
M, K, N = 2048, 2048, 2048
DTYPE = jnp.bfloat16


def input_fn(config):
    """Create random inputs for benchmarking."""
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (M, K), dtype=DTYPE)
    w = jax.random.normal(k2, (K, N), dtype=DTYPE)
    return (x, w)


def kernel_fn(x, w, *, block_m, block_k, block_n, **_):
    """Wrapper matching the tune() calling convention."""
    return tiled_matmul(x, w, block_m=block_m, block_k=block_k, block_n=block_n)


def flops_fn(config):
    """Total FLOPs for a matmul: 2*M*K*N."""
    return 2 * M * K * N


def bytes_fn(config):
    """Total bytes accessed: read x + w, write output (bfloat16 = 2 bytes)."""
    elem_bytes = 2  # bfloat16
    return (M * K + K * N + M * N) * elem_bytes


def xla_baseline():
    """Benchmark JAX's default matmul (XLA compiler) for comparison."""
    from pallas_forge.tune.runner import BenchmarkRunner

    def xla_matmul(x, w):
        return jnp.matmul(x, w)

    runner = BenchmarkRunner(
        xla_matmul,
        lambda _: input_fn({}),
        n_warmup=5,
        n_repeat=20,
    )
    result = runner.run_single({})
    print(f"\nXLA baseline: {result.median_ms:.3f} ms")
    return result


def main():
    config = TuneConfig.from_dict({
        "block_m": [64, 128, 256],
        "block_k": [64, 128, 256],
        "block_n": [64, 128, 256],
    })

    # Add constraint: block sizes must be at least 64
    config.add_constraint(lambda p: all(v >= 64 for v in p.values()))

    print(f"Matrix size: {M}x{K} @ {K}x{N}, dtype={DTYPE}")
    print(f"Total configs: {len(config.grid())}")

    report = tune(
        kernel_fn=kernel_fn,
        input_fn=input_fn,
        config=config,
        strategy="grid",
        flops_fn=flops_fn,
        bytes_fn=bytes_fn,
    )

    # Export results
    report.to_csv("results/matmul_results.csv")
    report.to_json("results/matmul_results.json")

    # Generate heatmaps
    report.heatmap("block_m", "block_n", metric="median_ms",
                   title=f"MatMul Latency (ms) — {M}x{K}x{N}",
                   save_path="results/matmul_heatmap_time.png")
    report.heatmap("block_m", "block_k", metric="median_ms",
                   title=f"MatMul Latency by block_m vs block_k",
                   save_path="results/matmul_heatmap_mk.png")

    if report.results and report.results[0].tflops is not None:
        report.heatmap("block_m", "block_n", metric="tflops",
                       title=f"MatMul Throughput (TFLOPS)",
                       save_path="results/matmul_heatmap_tflops.png",
                       cmap="YlGn")

    # XLA baseline comparison
    xla_result = xla_baseline()
    best = report.best(1)[0]
    if xla_result.median_ms > 0:
        speedup = xla_result.median_ms / best.median_ms
        print(f"\nBest Pallas config: {best.config}")
        print(f"Pallas: {best.median_ms:.3f} ms, XLA: {xla_result.median_ms:.3f} ms")
        print(f"Speedup vs XLA: {speedup:.2f}x")

    print(f"\nResults saved to results/")


if __name__ == "__main__":
    main()
