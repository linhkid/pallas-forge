"""Shared test fixtures and configuration."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "tpu_only: requires TPU hardware")


requires_tpu = pytest.mark.skipif(
    not any(d.platform == "tpu" for d in jax.devices()),
    reason="Requires TPU hardware",
)


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


@pytest.fixture(params=[jnp.float32, jnp.bfloat16], ids=["f32", "bf16"])
def dtype(request):
    return request.param


@pytest.fixture
def tolerance(dtype):
    """Return appropriate tolerance for the given dtype."""
    if dtype == jnp.bfloat16:
        return dict(atol=1e-1, rtol=1e-1)
    return dict(atol=1e-4, rtol=1e-4)
