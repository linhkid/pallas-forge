"""Tests for the Fused RMSNorm + Residual kernel."""

from __future__ import annotations

import pytest

import jax
import jax.numpy as jnp

from pallas_forge.kernels.rmsnorm import fused_rmsnorm_residual, rmsnorm_reference


class TestFusedRMSNormResidual:
    """Correctness tests for fused_rmsnorm_residual."""

    def test_basic_correctness(self, rng_key):
        """Basic correctness against reference implementation."""
        k1, k2, k3 = jax.random.split(rng_key, 3)
        batch, seq, dim = 2, 4, 64
        x = jax.random.normal(k1, (batch, seq, dim), dtype=jnp.float32)
        residual = jax.random.normal(k2, (batch, seq, dim), dtype=jnp.float32)
        weight = jax.random.normal(k3, (dim,), dtype=jnp.float32)

        out, new_res = fused_rmsnorm_residual(x, residual, weight)

        # Check residual add
        expected_residual = x + residual
        assert jnp.allclose(new_res, expected_residual, atol=1e-4, rtol=1e-4)

        # Check normalization
        expected_norm = rmsnorm_reference(expected_residual, weight)
        assert jnp.allclose(out, expected_norm, atol=1e-4, rtol=1e-4)

    def test_output_shapes(self, rng_key):
        """Output shapes should match input shapes."""
        k1, k2 = jax.random.split(rng_key)
        shape = (2, 8, 128)
        x = jax.random.normal(k1, shape, dtype=jnp.float32)
        residual = jax.random.normal(k2, shape, dtype=jnp.float32)
        weight = jnp.ones(shape[-1])

        out, new_res = fused_rmsnorm_residual(x, residual, weight)
        assert out.shape == shape
        assert new_res.shape == shape

    def test_zero_residual(self, rng_key):
        """With zero residual, new_residual should equal x."""
        dim = 64
        x = jax.random.normal(rng_key, (2, 4, dim), dtype=jnp.float32)
        residual = jnp.zeros_like(x)
        weight = jnp.ones(dim)

        out, new_res = fused_rmsnorm_residual(x, residual, weight)
        assert jnp.allclose(new_res, x, atol=1e-5)

    def test_unit_weight(self, rng_key):
        """With weight=1, output is just normalized x+residual."""
        k1, k2 = jax.random.split(rng_key)
        dim = 64
        x = jax.random.normal(k1, (2, 4, dim), dtype=jnp.float32)
        residual = jax.random.normal(k2, (2, 4, dim), dtype=jnp.float32)
        weight = jnp.ones(dim)

        out, new_res = fused_rmsnorm_residual(x, residual, weight)
        expected = rmsnorm_reference(new_res, weight)
        assert jnp.allclose(out, expected, atol=1e-4, rtol=1e-4)

    def test_eps_prevents_div_by_zero(self, rng_key):
        """Epsilon should prevent division by zero for all-zero input."""
        dim = 64
        x = jnp.zeros((1, 1, dim), dtype=jnp.float32)
        residual = jnp.zeros_like(x)
        weight = jnp.ones(dim)

        out, new_res = fused_rmsnorm_residual(x, residual, weight, eps=1e-6)
        assert jnp.all(jnp.isfinite(out))

    def test_bfloat16(self, rng_key):
        """Should work with bfloat16 inputs."""
        k1, k2 = jax.random.split(rng_key)
        dim = 64
        x = jax.random.normal(k1, (2, 4, dim), dtype=jnp.bfloat16)
        residual = jax.random.normal(k2, (2, 4, dim), dtype=jnp.bfloat16)
        weight = jnp.ones(dim, dtype=jnp.bfloat16)

        out, new_res = fused_rmsnorm_residual(x, residual, weight)
        assert out.dtype == jnp.bfloat16
        assert new_res.dtype == jnp.bfloat16

    def test_shape_mismatch_raises(self, rng_key):
        """Mismatched shapes should raise ValueError."""
        x = jnp.ones((2, 4, 64))
        residual = jnp.ones((2, 4, 32))
        weight = jnp.ones(64)

        with pytest.raises(ValueError, match="same shape"):
            fused_rmsnorm_residual(x, residual, weight)

    def test_weight_shape_mismatch_raises(self, rng_key):
        """Wrong weight shape should raise ValueError."""
        x = jnp.ones((2, 4, 64))
        residual = jnp.ones((2, 4, 64))
        weight = jnp.ones(32)

        with pytest.raises(ValueError, match="weight"):
            fused_rmsnorm_residual(x, residual, weight)
