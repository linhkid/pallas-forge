"""Fused SwiGLU / GeGLU activation kernels.

Compute-bound fused activations used in modern transformers (LLaMA, Gemma, Mistral).
Combines the gating multiplication with the activation function in a single kernel,
reducing intermediate materialization to HBM.

SwiGLU: silu(x @ w_gate) * (x @ w_up)
GeGLU:  gelu(x @ w_gate) * (x @ w_up)

The fusion saves two full HBM round-trips for the gate and up intermediate results.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from pallas_forge._compat import pallas_call_compat
from pallas_forge.kernels._utils import pad_to_multiple, unpad


def _swiglu_kernel(
    x_ref,  # [block_m, dim]
    w_gate_ref,  # [dim, block_n]
    w_up_ref,  # [dim, block_n]
    out_ref,  # [block_m, block_n]
    *,
    use_gelu: bool,
):
    """Fused gated activation kernel.

    Computes activation(x @ w_gate) * (x @ w_up) in a single pass.
    The gate and up projections are computed without writing intermediates to HBM.
    """
    x = x_ref[...]
    w_gate = w_gate_ref[...]
    w_up = w_up_ref[...]

    # Gate projection + activation
    gate = jnp.dot(x, w_gate, preferred_element_type=jnp.float32)
    if use_gelu:
        gate = jax.nn.gelu(gate)
    else:
        gate = jax.nn.silu(gate)

    # Up projection
    up = jnp.dot(x, w_up, preferred_element_type=jnp.float32)

    # Gated output
    out_ref[...] = (gate * up).astype(out_ref.dtype)


def _fused_glu(
    x: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    *,
    block_m: int = 128,
    block_n: int = 128,
    use_gelu: bool = False,
    num_stages: int = 2,
) -> jax.Array:
    """Internal implementation for fused gated linear units.

    Args:
        x: Input, shape [M, K] (flattened batch*seq, hidden_dim).
        w_gate: Gate weight, shape [K, N].
        w_up: Up weight, shape [K, N].
        block_m: Tile size for M (token) dimension.
        block_n: Tile size for N (output) dimension.
        use_gelu: If True, use GELU; otherwise SiLU.
        num_stages: DMA pipeline stages.

    Returns:
        Output, shape [M, N].
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D input, got ndim={x.ndim}")
    if w_gate.shape != w_up.shape:
        raise ValueError(f"w_gate and w_up must have same shape: {w_gate.shape} vs {w_up.shape}")
    if x.shape[1] != w_gate.shape[0]:
        raise ValueError(
            f"Inner dims must match: x.shape[1]={x.shape[1]}, w_gate.shape[0]={w_gate.shape[0]}"
        )

    M, K = x.shape
    _, N = w_gate.shape
    orig_M, orig_N = M, N
    out_dtype = x.dtype

    # Pad to block-aligned dimensions
    x = pad_to_multiple(x, block_m, axis=0)
    w_gate = pad_to_multiple(w_gate, block_n, axis=1)
    w_up = pad_to_multiple(w_up, block_n, axis=1)

    M_pad = x.shape[0]
    N_pad = w_gate.shape[1]

    grid_m = M_pad // block_m
    grid_n = N_pad // block_n

    kernel = functools.partial(_swiglu_kernel, use_gelu=use_gelu)

    result = pallas_call_compat(
        kernel,
        grid=(grid_m, grid_n),
        in_specs=[
            pl.BlockSpec((block_m, K), lambda i, j: (i, 0)),
            pl.BlockSpec((K, block_n), lambda i, j: (0, j)),
            pl.BlockSpec((K, block_n), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((M_pad, N_pad), out_dtype),
        num_stages=num_stages,
    )(x, w_gate, w_up)

    return unpad(result, (orig_M, orig_N))


@functools.partial(jax.jit, static_argnames=("block_m", "block_n", "num_stages"))
def fused_swiglu(
    x: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    *,
    block_m: int = 128,
    block_n: int = 128,
    num_stages: int = 2,
) -> jax.Array:
    """Fused SwiGLU activation: ``silu(x @ w_gate) * (x @ w_up)``.

    ``@jax.jit``-wrapped: traces + lowers once per (block_m, block_n, num_stages,
    input-shape) combination; subsequent calls hit JAX's cache.

    Args:
        x: Input tensor, shape ``[..., dim]``. Batch dims are flattened internally.
        w_gate: Gate weight, shape ``[dim, ffn_dim]``.
        w_up: Up projection weight, shape ``[dim, ffn_dim]``.
        block_m: Tile size for token dimension (multiple of 8).
        block_n: Tile size for output dimension (multiple of 128 on TPU).
        num_stages: DMA pipeline stages.

    Returns:
        Output tensor, shape ``[..., ffn_dim]``.
    """
    original_batch_shape = x.shape[:-1]
    dim = x.shape[-1]
    x_2d = x.reshape(-1, dim)
    result = _fused_glu(x_2d, w_gate, w_up, block_m=block_m, block_n=block_n, num_stages=num_stages)
    return result.reshape(*original_batch_shape, -1)


@functools.partial(jax.jit, static_argnames=("block_m", "block_n", "num_stages"))
def fused_geglu(
    x: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    *,
    block_m: int = 128,
    block_n: int = 128,
    num_stages: int = 2,
) -> jax.Array:
    """Fused GeGLU activation: ``gelu(x @ w_gate) * (x @ w_up)``.

    ``@jax.jit``-wrapped: see ``fused_swiglu`` for caching semantics.

    Args:
        x: Input tensor, shape ``[..., dim]``. Batch dims are flattened internally.
        w_gate: Gate weight, shape ``[dim, ffn_dim]``.
        w_up: Up projection weight, shape ``[dim, ffn_dim]``.
        block_m: Tile size for token dimension (multiple of 8).
        block_n: Tile size for output dimension (multiple of 128 on TPU).
        num_stages: DMA pipeline stages.

    Returns:
        Output tensor, shape ``[..., ffn_dim]``.
    """
    original_batch_shape = x.shape[:-1]
    dim = x.shape[-1]
    x_2d = x.reshape(-1, dim)
    result = _fused_glu(
        x_2d,
        w_gate,
        w_up,
        block_m=block_m,
        block_n=block_n,
        use_gelu=True,
        num_stages=num_stages,
    )
    return result.reshape(*original_batch_shape, -1)
