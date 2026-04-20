"""Tiled Matrix Multiplication — the "hello world" of Pallas.

Implements blocked matrix multiplication with configurable block sizes,
demonstrating BlockSpec, grid, Ref types, and the 8x128 alignment constraint.

Uses the canonical Pallas TPU matmul pattern:
    - 3D grid (grid_m, grid_n, k_tiles) — K axis is iterated by the grid
    - Accumulator lives in o_ref across K iterations (Mosaic keeps it in VMEM)
    - pl.when(program_id(2) == 0) zeros the accumulator on the first K tile

TPU block-size constraints (enforced by the Pallas TPU lowering):
    - block_k must be divisible by 128 (last dim of x tile)
    - block_n must be divisible by 128 (last dim of w tile + output tile)
    - block_m must be divisible by 8

The block sizes are the primary tuning knobs. Larger blocks improve MXU
utilization but increase VMEM pressure.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from pallas_forge._compat import pallas_call_compat
from pallas_forge.kernels._utils import pad_to_multiple, unpad


def _matmul_kernel(x_ref, w_ref, o_ref):
    """Pallas kernel for one (i, j, k) cell of the 3D matmul grid.

    - On k=0, zeros the accumulator in o_ref.
    - Adds this K-slab's contribution to the accumulator.

    o_ref is always fp32 so the K-axis accumulation stays in fp32 across
    iterations — casting to the target output dtype happens once at the
    wrapper level, not per K tile. This matters for bf16 inputs where
    per-tile casts would accumulate rounding error.
    """

    @pl.when(pl.program_id(2) == 0)
    def _reset():
        o_ref[...] = jnp.zeros_like(o_ref)

    o_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)


def tiled_matmul(
    x: jax.Array,
    w: jax.Array,
    *,
    block_m: int = 128,
    block_k: int = 128,
    block_n: int = 128,
    num_stages: int = 2,
) -> jax.Array:
    """Tiled matrix multiplication using Pallas.

    Computes x @ w with configurable tile sizes. Handles non-aligned dimensions
    by padding inputs up to block-aligned sizes and slicing the result.

    Args:
        x: Left matrix, shape [M, K].
        w: Right matrix, shape [K, N].
        block_m: Tile size for M dimension. Must be a multiple of 8.
        block_k: Tile size for K dimension (reduction). Must be a multiple of 128 on TPU.
        block_n: Tile size for N dimension. Must be a multiple of 128 on TPU.
        num_stages: DMA pipeline stages. Forwarded to compiler_params where supported;
            silently dropped on JAX versions that don't expose the knob.

    Returns:
        Result matrix, shape [M, N].
    """
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError(f"Expected 2D inputs, got x.ndim={x.ndim}, w.ndim={w.ndim}")
    if x.shape[1] != w.shape[0]:
        raise ValueError(f"Inner dimensions must match: x.shape={x.shape}, w.shape={w.shape}")

    M, K = x.shape
    _, N = w.shape
    orig_M, orig_N = M, N
    out_dtype = x.dtype

    # Pad to block-aligned sizes so the 3D grid divides cleanly.
    x = pad_to_multiple(x, block_m, axis=0)
    x = pad_to_multiple(x, block_k, axis=1)
    w = pad_to_multiple(w, block_k, axis=0)
    w = pad_to_multiple(w, block_n, axis=1)

    M_pad, K_pad = x.shape
    _, N_pad = w.shape

    grid_m = M_pad // block_m
    grid_n = N_pad // block_n
    n_k_tiles = K_pad // block_k

    # 3D grid: the K axis is iterated by the grid itself, not via dynamic_slice
    # inside the kernel body. This is the canonical Pallas TPU matmul pattern
    # (dynamic_slice is not supported by the TPU tensor-core lowering).
    #
    # Output of the Pallas kernel is always fp32 for numerical stability across
    # K-axis accumulation. We cast to the caller's dtype in a single post-pass.
    result_f32 = pallas_call_compat(
        _matmul_kernel,
        grid=(grid_m, grid_n, n_k_tiles),
        in_specs=[
            pl.BlockSpec((block_m, block_k), lambda i, j, k: (i, k)),
            pl.BlockSpec((block_k, block_n), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j, k: (i, j)),
        out_shape=jax.ShapeDtypeStruct((M_pad, N_pad), jnp.float32),
        num_stages=num_stages,
    )(x, w)

    return unpad(result_f32, (orig_M, orig_N)).astype(out_dtype)
