"""Tests for the Tiled MatMul kernel."""

from __future__ import annotations

import pytest

import jax
import jax.numpy as jnp

from pallas_forge.kernels.matmul import tiled_matmul


class TestTiledMatmul:
    """Correctness tests for tiled_matmul (run on CPU via interpret mode)."""

    def test_small_aligned(self, rng_key, dtype, tolerance):
        """Small matrix that's perfectly aligned to block boundaries."""
        k1, k2 = jax.random.split(rng_key)
        M, K, N = 128, 128, 128
        x = jax.random.normal(k1, (M, K), dtype=dtype)
        w = jax.random.normal(k2, (K, N), dtype=dtype)

        result = tiled_matmul(x, w, block_m=64, block_k=64, block_n=64)
        expected = jnp.matmul(x, w)

        assert result.shape == (M, N)
        assert result.dtype == dtype
        assert jnp.allclose(result, expected, **tolerance)

    def test_medium_aligned(self, rng_key, dtype, tolerance):
        """Medium matrix with default block sizes."""
        k1, k2 = jax.random.split(rng_key)
        M, K, N = 256, 256, 256
        x = jax.random.normal(k1, (M, K), dtype=dtype)
        w = jax.random.normal(k2, (K, N), dtype=dtype)

        result = tiled_matmul(x, w, block_m=128, block_k=128, block_n=128)
        expected = jnp.matmul(x, w)

        assert jnp.allclose(result, expected, **tolerance)

    def test_unaligned_dimensions(self, rng_key):
        """Non-aligned dimensions trigger padding/unpadding."""
        k1, k2 = jax.random.split(rng_key)
        M, K, N = 100, 200, 150
        x = jax.random.normal(k1, (M, K), dtype=jnp.float32)
        w = jax.random.normal(k2, (K, N), dtype=jnp.float32)

        result = tiled_matmul(x, w, block_m=64, block_k=64, block_n=64)
        expected = jnp.matmul(x, w)

        assert result.shape == (M, N)
        assert jnp.allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_rectangular(self, rng_key):
        """Tall and wide matrices."""
        k1, k2 = jax.random.split(rng_key)
        M, K, N = 256, 64, 512
        x = jax.random.normal(k1, (M, K), dtype=jnp.float32)
        w = jax.random.normal(k2, (K, N), dtype=jnp.float32)

        result = tiled_matmul(x, w, block_m=64, block_k=64, block_n=128)
        expected = jnp.matmul(x, w)

        assert result.shape == (M, N)
        assert jnp.allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_identity_matrix(self, rng_key):
        """Multiply by identity should return the original."""
        k1 = rng_key
        M, K = 128, 128
        x = jax.random.normal(k1, (M, K), dtype=jnp.float32)
        w = jnp.eye(K, dtype=jnp.float32)

        result = tiled_matmul(x, w, block_m=64, block_k=64, block_n=64)
        assert jnp.allclose(result, x, atol=1e-5, rtol=1e-5)

    def test_zero_input(self, rng_key):
        """Zero input should produce zero output."""
        M, K, N = 128, 128, 128
        x = jnp.zeros((M, K), dtype=jnp.float32)
        w = jax.random.normal(rng_key, (K, N), dtype=jnp.float32)

        result = tiled_matmul(x, w, block_m=64, block_k=64, block_n=64)
        assert jnp.allclose(result, jnp.zeros((M, N)), atol=1e-6)

    def test_different_block_sizes(self, rng_key):
        """Various block size combinations should all produce correct results."""
        k1, k2 = jax.random.split(rng_key)
        M, K, N = 256, 256, 256
        x = jax.random.normal(k1, (M, K), dtype=jnp.float32)
        w = jax.random.normal(k2, (K, N), dtype=jnp.float32)
        expected = jnp.matmul(x, w)

        for bm, bk, bn in [(64, 64, 64), (128, 128, 128), (256, 128, 64)]:
            result = tiled_matmul(x, w, block_m=bm, block_k=bk, block_n=bn)
            assert jnp.allclose(result, expected, atol=1e-4, rtol=1e-4), \
                f"Failed for block_m={bm}, block_k={bk}, block_n={bn}"

    def test_invalid_inputs(self):
        """Invalid inputs should raise ValueError."""
        with pytest.raises(ValueError, match="2D"):
            tiled_matmul(jnp.ones((2, 3, 4)), jnp.ones((4, 5)))

        with pytest.raises(ValueError, match="Inner dimensions"):
            tiled_matmul(jnp.ones((2, 3)), jnp.ones((4, 5)))
