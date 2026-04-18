"""Benchmark script for Fused RMSNorm + Residual kernel.

This is a memory-bound / VPU-bound kernel. The key tuning parameter is
block_size (hidden dimension elements per kernel instance).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pallas_forge.kernels.rmsnorm import fused_rmsnorm_residual
from pallas_forge.tune import TuneConfig, tune


# Problem size
BATCH, SEQ_LEN, DIM = 4, 2048, 4096
DTYPE = jnp.bfloat16


def input_fn(config):
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (BATCH, SEQ_LEN, DIM), dtype=DTYPE)
    residual = jax.random.normal(k2, (BATCH, SEQ_LEN, DIM), dtype=DTYPE)
    weight = jax.random.normal(k3, (DIM,), dtype=DTYPE)
    return (x, residual, weight)


def kernel_fn(x, residual, weight, *, block_size, **_):
    out, new_res = fused_rmsnorm_residual(x, residual, weight, block_size=block_size)
    return out


def bytes_fn(config):
    """Bytes accessed: read x + residual + weight, write output + new_residual."""
    elem_bytes = 2  # bfloat16
    tokens = BATCH * SEQ_LEN
    read_bytes = (tokens * DIM * 2 + DIM) * elem_bytes  # x + residual + weight
    write_bytes = tokens * DIM * 2 * elem_bytes  # output + new_residual
    return read_bytes + write_bytes


def xla_baseline():
    """Benchmark unfused JAX ops for comparison."""
    from pallas_forge.tune.runner import BenchmarkRunner

    def xla_rmsnorm_residual(x, residual, weight):
        new_res = x + residual
        variance = jnp.mean(new_res.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
        rms = jnp.sqrt(variance + 1e-6)
        normed = new_res.astype(jnp.float32) / rms
        return (normed * weight.astype(jnp.float32)).astype(x.dtype)

    runner = BenchmarkRunner(
        xla_rmsnorm_residual,
        lambda _: input_fn({}),
        n_warmup=5,
        n_repeat=20,
    )
    result = runner.run_single({})
    print(f"\nXLA baseline: {result.median_ms:.3f} ms")
    return result


def main():
    config = TuneConfig.from_dict({
        "block_size": [DIM],  # For RMSNorm, typically process full hidden dim
    })

    print(f"Shape: [{BATCH}, {SEQ_LEN}, {DIM}], dtype={DTYPE}")

    report = tune(
        kernel_fn=kernel_fn,
        input_fn=input_fn,
        config=config,
        strategy="grid",
        bytes_fn=bytes_fn,
    )

    report.to_csv("results/rmsnorm_results.csv")
    report.to_json("results/rmsnorm_results.json")

    xla_result = xla_baseline()
    best = report.best(1)[0]
    if xla_result.median_ms > 0:
        speedup = xla_result.median_ms / best.median_ms
        print(f"\nBest Pallas: {best.median_ms:.3f} ms, XLA: {xla_result.median_ms:.3f} ms")
        print(f"Speedup vs XLA: {speedup:.2f}x")


if __name__ == "__main__":
    main()
