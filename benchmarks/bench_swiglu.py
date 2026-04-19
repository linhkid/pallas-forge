"""Benchmark script for Fused SwiGLU kernel.

This is a compute-bound kernel where fusion saves HBM round-trips for
the gate and up projection intermediates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pallas_forge.kernels.swiglu import fused_swiglu
from pallas_forge.tune import TuneConfig, tune

# Problem size (typical LLaMA-style FFN dimensions)
BATCH_SEQ = 2048  # batch * seq_len
DIM = 4096
FFN_DIM = 11008
DTYPE = jnp.bfloat16


def input_fn(config):
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (BATCH_SEQ, DIM), dtype=DTYPE)
    w_gate = jax.random.normal(k2, (DIM, FFN_DIM), dtype=DTYPE)
    w_up = jax.random.normal(k3, (DIM, FFN_DIM), dtype=DTYPE)
    return (x, w_gate, w_up)


def kernel_fn(x, w_gate, w_up, *, block_m, block_n, **_):
    return fused_swiglu(x, w_gate, w_up, block_m=block_m, block_n=block_n)


def flops_fn(config):
    """FLOPs: two matmuls (2*M*K*N each) + activation + elementwise mul."""
    matmul_flops = 2 * BATCH_SEQ * DIM * FFN_DIM * 2  # two matmuls
    activation_flops = BATCH_SEQ * FFN_DIM * 5  # silu ≈ 5 ops per element
    mul_flops = BATCH_SEQ * FFN_DIM
    return matmul_flops + activation_flops + mul_flops


def bytes_fn(config):
    """Bytes: read x + w_gate + w_up, write output."""
    elem_bytes = 2  # bfloat16
    read_bytes = (BATCH_SEQ * DIM + DIM * FFN_DIM * 2) * elem_bytes
    write_bytes = BATCH_SEQ * FFN_DIM * elem_bytes
    return read_bytes + write_bytes


def xla_baseline():
    """Benchmark unfused JAX ops for comparison."""
    from pallas_forge.tune.runner import BenchmarkRunner

    def xla_swiglu(x, w_gate, w_up):
        gate = jax.nn.silu(jnp.matmul(x, w_gate))
        up = jnp.matmul(x, w_up)
        return gate * up

    runner = BenchmarkRunner(
        xla_swiglu,
        lambda _: input_fn({}),
        n_warmup=5,
        n_repeat=20,
    )
    result = runner.run_single({})
    print(f"\nXLA baseline: {result.median_ms:.3f} ms")
    return result


def main():
    config = TuneConfig.from_dict(
        {
            "block_m": [64, 128, 256],
            "block_n": [128, 256, 512],
        }
    )

    print(f"Shape: [{BATCH_SEQ}, {DIM}] x [{DIM}, {FFN_DIM}], dtype={DTYPE}")

    report = tune(
        kernel_fn=kernel_fn,
        input_fn=input_fn,
        config=config,
        strategy="grid",
        flops_fn=flops_fn,
        bytes_fn=bytes_fn,
    )

    report.to_csv("results/swiglu_results.csv")
    report.to_json("results/swiglu_results.json")

    report.heatmap(
        "block_m",
        "block_n",
        metric="median_ms",
        title=f"SwiGLU Latency (ms) — {BATCH_SEQ}x{DIM}x{FFN_DIM}",
        save_path="results/swiglu_heatmap_time.png",
    )

    xla_result = xla_baseline()
    best = report.best(1)[0]
    if xla_result.median_ms > 0:
        speedup = xla_result.median_ms / best.median_ms
        print(f"\nBest Pallas: {best.median_ms:.3f} ms, XLA: {xla_result.median_ms:.3f} ms")
        print(f"Speedup vs XLA: {speedup:.2f}x")


if __name__ == "__main__":
    main()
