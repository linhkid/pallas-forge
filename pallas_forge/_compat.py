"""CPU/TPU compatibility shim for Pallas kernels.

This module is the architectural linchpin that enables all kernel correctness tests
to run on CPU without TPU hardware. It wraps `pallas_call` to automatically set
`interpret=True` when no TPU is available, which runs kernels as pure Python/NumPy.

Also handles JAX API drift for kwargs like `num_stages` whose location has changed
between releases (top-level → `compiler_params` → dropped).
"""

from __future__ import annotations

import inspect
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


@lru_cache(maxsize=1)
def _pallas_call_accepts_num_stages() -> bool:
    """Detect whether the installed JAX accepts num_stages directly.

    Older JAX (≤~0.4.35) accepted `num_stages=N` as a direct kwarg to `pallas_call`.
    Newer JAX moved it into `compiler_params` on the TPU backend, and some versions
    dropped the knob entirely in favour of automatic pipelining.
    """
    try:
        sig = inspect.signature(pl.pallas_call)
        if "num_stages" in sig.parameters:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except (TypeError, ValueError):
        # If we can't introspect (e.g. C-level function), try at call time
        return True


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
        **kwargs: Additional arguments passed to pallas_call (e.g. num_stages).
            Unknown kwargs are silently dropped so the wrapper stays forward-
            compatible with newer JAX versions that moved knobs around.

    Returns:
        A JAX-compatible function that executes the kernel.
    """
    if interpret is None:
        interpret = get_default_interpret_mode()

    # `num_stages` has migrated between JAX versions. Pop it out here rather than
    # passing it blindly — on newer JAX it lives inside `compiler_params` and
    # passing it at the top level raises TypeError. Dropping it is safe: the
    # compiler picks a reasonable default for DMA pipelining.
    num_stages = kwargs.pop("num_stages", None)
    if num_stages is not None and not interpret:
        # Best-effort: try to forward it via the current JAX's compiler_params API.
        # If the API doesn't exist or doesn't accept num_stages, we silently skip.
        try:
            from jax.experimental.pallas import tpu as pltpu  # type: ignore

            cp_cls = getattr(pltpu, "CompilerParams", None)
            if cp_cls is not None:
                cp_sig = inspect.signature(cp_cls)
                if "num_stages" in cp_sig.parameters:
                    existing = kwargs.get("compiler_params")
                    if existing is None:
                        kwargs["compiler_params"] = cp_cls(num_stages=num_stages)
        except Exception:
            pass  # Not worth failing over — fall through without it.

    return pl.pallas_call(
        kernel_fn,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        interpret=interpret,
        **kwargs,
    )
