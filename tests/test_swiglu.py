"""Tests for the Fused SwiGLU / GeGLU kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pallas_forge.kernels.swiglu import fused_geglu, fused_swiglu


def swiglu_reference(x, w_gate, w_up):
    """Reference JAX implementation of SwiGLU."""
    gate = jax.nn.silu(jnp.matmul(x, w_gate))
    up = jnp.matmul(x, w_up)
    return gate * up


def geglu_reference(x, w_gate, w_up):
    """Reference JAX implementation of GeGLU."""
    gate = jax.nn.gelu(jnp.matmul(x, w_gate))
    up = jnp.matmul(x, w_up)
    return gate * up


class TestFusedSwiGLU:
    """Correctness tests for fused_swiglu."""

    def test_basic_correctness(self, rng_key):
        """SwiGLU matches reference implementation."""
        k1, k2, k3 = jax.random.split(rng_key, 3)
        M, K, N = 64, 64, 64
        x = jax.random.normal(k1, (M, K), dtype=jnp.float32)
        w_gate = jax.random.normal(k2, (K, N), dtype=jnp.float32)
        w_up = jax.random.normal(k3, (K, N), dtype=jnp.float32)

        result = fused_swiglu(x, w_gate, w_up, block_m=32, block_n=32)
        expected = swiglu_reference(x, w_gate, w_up)

        assert result.shape == (M, N)
        assert jnp.allclose(result, expected, atol=1e-3, rtol=1e-3)

    def test_batched_input(self, rng_key):
        """Should handle 3D (batch, seq, dim) inputs."""
        k1, k2, k3 = jax.random.split(rng_key, 3)
        B, S, D, N = 2, 4, 64, 128
        x = jax.random.normal(k1, (B, S, D), dtype=jnp.float32)
        w_gate = jax.random.normal(k2, (D, N), dtype=jnp.float32)
        w_up = jax.random.normal(k3, (D, N), dtype=jnp.float32)

        result = fused_swiglu(x, w_gate, w_up, block_m=32, block_n=64)

        # Reference: flatten, compute, reshape
        x_flat = x.reshape(-1, D)
        expected = swiglu_reference(x_flat, w_gate, w_up).reshape(B, S, N)

        assert result.shape == (B, S, N)
        assert jnp.allclose(result, expected, atol=1e-3, rtol=1e-3)

    def test_bfloat16(self, rng_key):
        """Should work with bfloat16 and maintain output dtype."""
        k1, k2, k3 = jax.random.split(rng_key, 3)
        M, K, N = 64, 64, 64
        x = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
        w_gate = jax.random.normal(k2, (K, N), dtype=jnp.bfloat16)
        w_up = jax.random.normal(k3, (K, N), dtype=jnp.bfloat16)

        result = fused_swiglu(x, w_gate, w_up, block_m=32, block_n=32)
        assert result.dtype == jnp.bfloat16
        assert result.shape == (M, N)

    def test_unaligned_dimensions(self, rng_key):
        """Non-aligned dimensions should still work via padding."""
        k1, k2, k3 = jax.random.split(rng_key, 3)
        M, K, N = 50, 64, 70
        x = jax.random.normal(k1, (M, K), dtype=jnp.float32)
        w_gate = jax.random.normal(k2, (K, N), dtype=jnp.float32)
        w_up = jax.random.normal(k3, (K, N), dtype=jnp.float32)

        result = fused_swiglu(x, w_gate, w_up, block_m=32, block_n=32)
        expected = swiglu_reference(x, w_gate, w_up)

        assert result.shape == (M, N)
        assert jnp.allclose(result, expected, atol=1e-3, rtol=1e-3)


class TestFusedGeGLU:
    """Correctness tests for fused_geglu."""

    def test_basic_correctness(self, rng_key):
        """GeGLU matches reference implementation."""
        k1, k2, k3 = jax.random.split(rng_key, 3)
        M, K, N = 64, 64, 64
        x = jax.random.normal(k1, (M, K), dtype=jnp.float32)
        w_gate = jax.random.normal(k2, (K, N), dtype=jnp.float32)
        w_up = jax.random.normal(k3, (K, N), dtype=jnp.float32)

        result = fused_geglu(x, w_gate, w_up, block_m=32, block_n=32)
        expected = geglu_reference(x, w_gate, w_up)

        assert result.shape == (M, N)
        assert jnp.allclose(result, expected, atol=1e-3, rtol=1e-3)

    def test_geglu_vs_swiglu_different(self, rng_key):
        """GeGLU and SwiGLU should produce different results."""
        k1, k2, k3 = jax.random.split(rng_key, 3)
        M, K, N = 64, 64, 64
        x = jax.random.normal(k1, (M, K), dtype=jnp.float32)
        w_gate = jax.random.normal(k2, (K, N), dtype=jnp.float32)
        w_up = jax.random.normal(k3, (K, N), dtype=jnp.float32)

        swiglu_result = fused_swiglu(x, w_gate, w_up, block_m=32, block_n=32)
        geglu_result = fused_geglu(x, w_gate, w_up, block_m=32, block_n=32)

        assert not jnp.allclose(swiglu_result, geglu_result, atol=1e-3)
