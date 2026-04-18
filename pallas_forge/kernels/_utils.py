"""Shared utilities for Pallas kernels.

Provides alignment checking, padding, and validation functions needed by all kernels
to meet TPU hardware constraints (8x128 tile alignment).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def check_alignment(shape: tuple[int, ...], alignment: int = 128) -> bool:
    """Check if all dimensions of a shape meet the alignment requirement.

    TPU hardware requires tile dimensions to be multiples of 128 (or 8 for the
    minor dimension). This function checks the general case.
    """
    return all(dim % alignment == 0 for dim in shape)


def pad_to_multiple(x: jax.Array, multiple: int, axis: int) -> jax.Array:
    """Pad a JAX array along `axis` so its size is a multiple of `multiple`.

    Returns the original array unchanged if already aligned.
    """
    size = x.shape[axis]
    remainder = size % multiple
    if remainder == 0:
        return x
    pad_size = multiple - remainder
    pad_widths = [(0, 0)] * x.ndim
    pad_widths[axis] = (0, pad_size)
    return jnp.pad(x, pad_widths)


def unpad(x: jax.Array, target_shape: tuple[int, ...]) -> jax.Array:
    """Remove padding by slicing back to the target shape."""
    slices = tuple(slice(0, s) for s in target_shape)
    return x[slices]


def ceildiv(a: int, b: int) -> int:
    """Ceiling division: ceildiv(7, 3) = 3."""
    return (a + b - 1) // b


def next_multiple(n: int, multiple: int) -> int:
    """Round n up to the next multiple. E.g., next_multiple(100, 128) = 128."""
    return ceildiv(n, multiple) * multiple
