"""Tests for the BenchmarkRunner."""

from __future__ import annotations

import jax.numpy as jnp

from pallas_forge.tune.runner import BenchmarkResult, BenchmarkRunner


class TestBenchmarkResult:
    def test_to_dict(self):
        r = BenchmarkResult(
            config={"block_m": 128, "block_n": 256},
            median_ms=1.5,
            mean_ms=1.6,
            std_ms=0.1,
            min_ms=1.4,
            max_ms=1.8,
            tflops=10.5,
        )
        d = r.to_dict()
        assert d["config_block_m"] == 128
        assert d["config_block_n"] == 256
        assert d["median_ms"] == 1.5
        assert d["tflops"] == 10.5

    def test_to_dict_no_throughput(self):
        r = BenchmarkResult(
            config={"x": 1},
            median_ms=1.0,
            mean_ms=1.0,
            std_ms=0.0,
            min_ms=1.0,
            max_ms=1.0,
        )
        d = r.to_dict()
        assert "tflops" not in d
        assert "bandwidth_gb_s" not in d


class TestBenchmarkRunner:
    def test_basic_timing(self):
        """Runner should produce valid timing results with a trivial kernel."""
        def kernel(x, *, multiplier=2):
            return x * multiplier

        def input_fn(config):
            return (jnp.ones((64, 64)),)

        runner = BenchmarkRunner(kernel, input_fn, n_warmup=2, n_repeat=5)
        result = runner.run_single({"multiplier": 3})

        assert result.median_ms > 0
        assert result.mean_ms > 0
        assert result.std_ms >= 0
        assert result.config == {"multiplier": 3}

    def test_run_all_sorted(self):
        """Results should be sorted by median time (fastest first)."""
        call_count = {"slow": 0, "fast": 0}

        def kernel(x, *, delay_iters=1):
            for _ in range(delay_iters):
                x = x + 1
            return x

        def input_fn(config):
            return (jnp.ones((32, 32)),)

        runner = BenchmarkRunner(kernel, input_fn, n_warmup=1, n_repeat=3)
        results = runner.run_all(
            [{"delay_iters": 100}, {"delay_iters": 1}],
            verbose=False,
        )

        assert len(results) == 2
        # Faster config should come first
        assert results[0].median_ms <= results[1].median_ms

    def test_flops_computation(self):
        """TFLOPS should be computed when flops_fn is provided."""
        def kernel(x):
            return x * 2

        runner = BenchmarkRunner(
            kernel,
            lambda _: (jnp.ones((64, 64)),),
            n_warmup=1,
            n_repeat=3,
            flops_fn=lambda _: 1_000_000,
        )
        result = runner.run_single({})
        assert result.tflops is not None
        assert result.tflops > 0

    def test_bandwidth_computation(self):
        """Bandwidth should be computed when bytes_fn is provided."""
        def kernel(x):
            return x + 1

        runner = BenchmarkRunner(
            kernel,
            lambda _: (jnp.ones((64, 64)),),
            n_warmup=1,
            n_repeat=3,
            bytes_fn=lambda _: 64 * 64 * 4 * 2,  # read + write
        )
        result = runner.run_single({})
        assert result.bandwidth_gb_s is not None
        assert result.bandwidth_gb_s > 0
