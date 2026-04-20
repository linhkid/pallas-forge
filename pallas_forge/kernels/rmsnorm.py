"""Fused RMSNorm + Residual Addition kernel.

A memory-bound kernel that fuses two operations (RMS normalization and residual
addition) into a single pass over HBM. This demonstrates:

- VPU vs. MXU tradeoffs (RMSNorm is VPU-bound, not MXU-bound)
- When kernel fusion beats XLA's auto-compiler
- 1D grid over tile-groups of tokens

The kernel processes ``TOKENS_PER_TILE`` = 8 tokens per program instance so the
second-to-last block dimension is divisible by 8 — the TPU sublane alignment
rule that Pallas's TPU lowering enforces. The wrapper pads the token axis up
to a multiple of 8, runs the kernel, then slices back to the original length.

On CPU interpret mode these alignment rules are not enforced, so the kernel
works identically; on TPU it lowers cleanly to the MXU sublane/lane layout.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from pallas_forge._compat import pallas_call_compat

# Number of tokens processed per Pallas kernel instance. Must be a multiple
# of 8 to satisfy the TPU sublane alignment rule for the second-to-last dim.
TOKENS_PER_TILE = 8


def _rmsnorm_residual_kernel(
    x_ref,  # [TOKENS_PER_TILE, dim]
    residual_ref,  # [TOKENS_PER_TILE, dim]
    weight_ref,  # [dim]
    out_ref,  # [TOKENS_PER_TILE, dim]
    new_res_ref,  # [TOKENS_PER_TILE, dim]
    *,
    eps: float,
):
    """Fused RMSNorm + residual addition kernel (batched over 8 tokens).

    For each of the 8 tokens in this tile:
        new_residual = x + residual
        rms          = sqrt(mean(new_residual**2) + eps)
        output       = (new_residual / rms) * weight
    """
    x = x_ref[...].astype(jnp.float32)
    residual = residual_ref[...].astype(jnp.float32)
    weight = weight_ref[...].astype(jnp.float32)

    new_residual = x + residual  # (8, dim)

    # Per-token RMS — reduce across the hidden dim, keep the 8 rows.
    variance = jnp.mean(new_residual * new_residual, axis=-1, keepdims=True)
    rms = jnp.sqrt(variance + eps)
    normed = new_residual / rms

    # Broadcast weight across the 8 sublanes.
    output = normed * weight[None, :]

    out_ref[...] = output.astype(out_ref.dtype)
    new_res_ref[...] = new_residual.astype(new_res_ref.dtype)


@functools.partial(jax.jit, static_argnames=("eps", "block_size"))
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
        output       = rmsnorm(new_residual) * weight

    Both operations happen in a single kernel pass, avoiding an extra HBM
    read/write for the intermediate residual.

    The function is ``@jax.jit``-wrapped; the first call with a given
    (input-shape, eps) combination traces and lowers the kernel once, and
    subsequent calls hit JAX's cache.

    Args:
        x: Input tensor, shape ``[..., dim]``.
        residual: Residual tensor, same shape as ``x``.
        weight: RMSNorm weight, shape ``[dim]``.
        eps: Epsilon for numerical stability.
        block_size: Unused — reserved for future sub-token tiling. Kept for
            API compatibility.

    Returns:
        Tuple of (normalized_output, new_residual), each same shape as ``x``.
    """
    del block_size  # reserved; currently unused
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

    # Pad num_tokens up to a multiple of TOKENS_PER_TILE so the sublane
    # dimension of the block shape is aligned on TPU.
    pad = (-num_tokens) % TOKENS_PER_TILE
    if pad:
        x_2d = jnp.pad(x_2d, ((0, pad), (0, 0)))
        residual_2d = jnp.pad(residual_2d, ((0, pad), (0, 0)))
    padded_tokens = x_2d.shape[0]
    n_tiles = padded_tokens // TOKENS_PER_TILE

    kernel = functools.partial(_rmsnorm_residual_kernel, eps=eps)

    out_padded, new_res_padded = pallas_call_compat(
        kernel,
        grid=(n_tiles,),
        in_specs=[
            pl.BlockSpec((TOKENS_PER_TILE, dim), lambda i: (i, 0)),
            pl.BlockSpec((TOKENS_PER_TILE, dim), lambda i: (i, 0)),
            pl.BlockSpec((dim,), lambda i: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((TOKENS_PER_TILE, dim), lambda i: (i, 0)),
            pl.BlockSpec((TOKENS_PER_TILE, dim), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, dim), out_dtype),
            jax.ShapeDtypeStruct((padded_tokens, dim), out_dtype),
        ],
    )(x_2d, residual_2d, weight)

    # Slice back to the original (unpadded) token count.
    out = out_padded[:num_tokens]
    new_res = new_res_padded[:num_tokens]

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
