"""Tiled Matrix Multiplication — the "hello world" of Pallas.

Implements blocked matrix multiplication with configurable block sizes,
demonstrating BlockSpec, grid, Ref types, and the 8x128 alignment constraint.

The block sizes (block_m, block_k, block_n) are the primary tuning knobs.
Larger blocks improve MXU utilization but increase VMEM pressure.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from pallas_forge._compat import pallas_call_compat
from pallas_forge.kernels._utils import pad_to_multiple, unpad


def _matmul_kernel(
    x_ref,  # [block_m, K]
    w_ref,  # [K, block_n]
    o_ref,  # [block_m, block_n]
    *,
    block_k: int,
):
    """Core Pallas kernel for tiled matrix multiplication.

    Each kernel instance computes one (block_m x block_n) output tile.
    It iterates over K in chunks of block_k, accumulating in float32.
    """
    K = x_ref.shape[1]
    k_tiles = K // block_k

    acc = jnp.zeros(o_ref.shape, dtype=jnp.float32)

    def body(k, acc):
        x_tile = jax.lax.dynamic_slice(x_ref[...], (0, k * block_k), (x_ref.shape[0], block_k))
        w_tile = jax.lax.dynamic_slice(w_ref[...], (k * block_k, 0), (block_k, w_ref.shape[1]))
        acc = acc + jnp.dot(x_tile, w_tile, preferred_element_type=jnp.float32)
        return acc

    acc = jax.lax.fori_loop(0, k_tiles, body, acc)
    o_ref[...] = acc.astype(o_ref.dtype)


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
    by padding and unpadding.

    Args:
        x: Left matrix, shape [M, K].
        w: Right matrix, shape [K, N].
        block_m: Tile size for M dimension.
        block_k: Tile size for K dimension (reduction loop).
        block_n: Tile size for N dimension.
        num_stages: DMA pipeline stages (>1 overlaps transfer with compute on TPU).

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

    # Pad to block-aligned dimensions
    x = pad_to_multiple(x, block_m, axis=0)
    x = pad_to_multiple(x, block_k, axis=1)
    w = pad_to_multiple(w, block_k, axis=0)
    w = pad_to_multiple(w, block_n, axis=1)

    M_pad, K_pad = x.shape
    _, N_pad = w.shape

    grid_m = M_pad // block_m
    grid_n = N_pad // block_n

    kernel = partial(_matmul_kernel, block_k=block_k)

    # 2D grid: each (i, j) computes one output tile, iterating over K internally
    result = pallas_call_compat(
        kernel,
        grid=(grid_m, grid_n),
        in_specs=[
            # Each grid cell (i, j) gets x[i*block_m:(i+1)*block_m, :] (full K row)
            pl.BlockSpec((block_m, K_pad), lambda i, j: (i, 0)),
            # Each grid cell (i, j) gets w[:, j*block_n:(j+1)*block_n] (full K col)
            pl.BlockSpec((K_pad, block_n), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((M_pad, N_pad), out_dtype),
        num_stages=num_stages,
    )(x, w)

    return unpad(result, (orig_M, orig_N))
