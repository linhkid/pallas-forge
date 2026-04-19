"""CPU/TPU compatibility shim for Pallas kernels.

This module is the architectural linchpin that enables all kernel correctness tests
to run on CPU without TPU hardware. It wraps `pallas_call` to automatically set
`interpret=True` when no TPU is available, which runs kernels as pure Python/NumPy.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import jax
from jax.experimental import pallas as pl


@lru_cache(maxsize=1)
def is_tpu_available() -> bool:
    """Check if TPU hardware is available."""
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except RuntimeError:
        return False


def get_default_interpret_mode() -> bool:
    """Returns True if no TPU is available (use interpret mode for CPU)."""
    return not is_tpu_available()


def pallas_call_compat(
    kernel_fn,
    *,
    grid: tuple[int, ...] | int,
    in_specs: Sequence[pl.BlockSpec],
    out_specs: pl.BlockSpec | Sequence[pl.BlockSpec],
    out_shape: Any,
    interpret: bool | None = None,
    **kwargs,
):
    """Wrapper around pallas_call that auto-sets interpret=True on CPU.

    This enables all Pallas kernels to run on CPU for correctness testing.
    On TPU, interpret mode is disabled for full hardware performance.

    Args:
        kernel_fn: The Pallas kernel function.
        grid: Grid dimensions for the kernel launch.
        in_specs: BlockSpec for each input.
        out_specs: BlockSpec for each output.
        out_shape: Shape specification for outputs.
        interpret: Force interpret mode. If None, auto-detect based on hardware.
        **kwargs: Additional arguments passed to pallas_call (e.g., num_stages).

    Returns:
        A JAX-compatible function that executes the kernel.
    """
    if interpret is None:
        interpret = get_default_interpret_mode()

    # num_stages is only valid on TPU, not in interpret mode
    if interpret and "num_stages" in kwargs:
        kwargs = {k: v for k, v in kwargs.items() if k != "num_stages"}

    return pl.pallas_call(
        kernel_fn,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        interpret=interpret,
        **kwargs,
    )
