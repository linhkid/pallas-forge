"""Fused RMSNorm + Residual Addition kernel.

A memory-bound kernel that fuses two operations (RMS normalization and residual
addition) into a single pass over HBM. This demonstrates:

- VPU vs. MXU tradeoffs (RMSNorm is VPU-bound, not MXU-bound)
- When kernel fusion beats XLA's auto-compiler
- 1D grid over tokens, each program handles one token's hidden dimension

The key tuning parameter is `block_size` — how many elements of the hidden
dimension to process per kernel instance.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from pallas_forge._compat import pallas_call_compat


def _rmsnorm_residual_kernel(
    x_ref,  # [1, dim]
    residual_ref,  # [1, dim]
    weight_ref,  # [dim]
    out_ref,  # [1, dim]
    new_res_ref,  # [1, dim]
    *,
    eps: float,
):
    """Fused RMSNorm + residual addition kernel.

    For each token position:
    1. new_residual = x + residual
    2. rms = sqrt(mean(new_residual^2) + eps)
    3. output = (new_residual / rms) * weight
    """
    x = x_ref[0, :].astype(jnp.float32)
    residual = residual_ref[0, :].astype(jnp.float32)
    weight = weight_ref[...].astype(jnp.float32)

    # Fused residual add
    new_residual = x + residual

    # RMS normalization
    variance = jnp.mean(new_residual * new_residual)
    rms = jnp.sqrt(variance + eps)
    normed = new_residual / rms

    # Scale by weight
    output = normed * weight

    out_ref[0, :] = output.astype(out_ref.dtype)
    new_res_ref[0, :] = new_residual.astype(new_res_ref.dtype)


def fused_rmsnorm_residual(
    x: jax.Array,
    residual: jax.Array,
    weight: jax.Array,
    *,
    eps: float = 1e-6,
    block_size: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Fused RMSNorm + residual addition.

    Computes:
        new_residual = x + residual
        output = rmsnorm(new_residual) * weight

    Both operations happen in a single kernel pass, avoiding an extra
    HBM read/write for the intermediate residual.

    Args:
        x: Input tensor, shape [..., dim].
        residual: Residual tensor, same shape as x.
        weight: RMSNorm weight, shape [dim].
        eps: Epsilon for numerical stability.
        block_size: Unused (reserved for future sub-token tiling). Kept for API compat.

    Returns:
        Tuple of (normalized_output, new_residual), each same shape as x.
    """
    if x.shape != residual.shape:
        raise ValueError(f"x and residual must have same shape: {x.shape} vs {residual.shape}")
    if weight.shape != (x.shape[-1],):
        raise ValueError(f"weight must have shape [{x.shape[-1]}], got {weight.shape}")

    original_shape = x.shape
    dim = x.shape[-1]
    out_dtype = x.dtype

    # Flatten to 2D: [num_tokens, dim]
    x_2d = x.reshape(-1, dim)
    residual_2d = residual.reshape(-1, dim)
    num_tokens = x_2d.shape[0]

    kernel = partial(_rmsnorm_residual_kernel, eps=eps)

    # Grid over tokens; each kernel instance processes one token (1 row of dim)
    out, new_res = pallas_call_compat(
        kernel,
        grid=(num_tokens,),
        in_specs=[
            pl.BlockSpec((1, dim), lambda i: (i, 0)),  # x: one row
            pl.BlockSpec((1, dim), lambda i: (i, 0)),  # residual: one row
            pl.BlockSpec((dim,), lambda i: (0,)),  # weight: always full
        ],
        out_specs=[
            pl.BlockSpec((1, dim), lambda i: (i, 0)),  # out: one row
            pl.BlockSpec((1, dim), lambda i: (i, 0)),  # new_res: one row
        ],
        out_shape=[
            jax.ShapeDtypeStruct((num_tokens, dim), out_dtype),
            jax.ShapeDtypeStruct((num_tokens, dim), out_dtype),
        ],
    )(x_2d, residual_2d, weight)

    return out.reshape(original_shape), new_res.reshape(original_shape)


def rmsnorm_reference(
    x: jax.Array,
    weight: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
    """Reference JAX implementation of RMSNorm (for testing)."""
    variance = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
    rms = jnp.sqrt(variance + eps)
    normed = x.astype(jnp.float32) / rms
    return (normed * weight.astype(jnp.float32)).astype(x.dtype)
