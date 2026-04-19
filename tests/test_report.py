"""Tests for TuneReport output generation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pallas_forge.tune.report import TuneReport
from pallas_forge.tune.runner import BenchmarkResult


def _make_results():
    """Create sample results for testing."""
    return [
        BenchmarkResult(
            config={"block_m": 64, "block_n": 64},
            median_ms=2.0,
            mean_ms=2.1,
            std_ms=0.1,
            min_ms=1.9,
            max_ms=2.3,
        ),
        BenchmarkResult(
            config={"block_m": 64, "block_n": 128},
            median_ms=1.5,
            mean_ms=1.6,
            std_ms=0.1,
            min_ms=1.4,
            max_ms=1.8,
        ),
        BenchmarkResult(
            config={"block_m": 128, "block_n": 64},
            median_ms=3.0,
            mean_ms=3.1,
            std_ms=0.2,
            min_ms=2.8,
            max_ms=3.4,
        ),
        BenchmarkResult(
            config={"block_m": 128, "block_n": 128},
            median_ms=1.0,
            mean_ms=1.1,
            std_ms=0.05,
            min_ms=0.95,
            max_ms=1.2,
        ),
    ]


class TestTuneReport:
    def test_best(self):
        report = TuneReport(_make_results())
        best = report.best(1)
        assert len(best) == 1
        assert best[0].median_ms == 1.0

    def test_worst(self):
        report = TuneReport(_make_results())
        worst = report.worst(1)
        assert len(worst) == 1
        assert worst[0].median_ms == 3.0

    def test_speedup_range(self):
        report = TuneReport(_make_results())
        assert report.speedup_range == 3.0  # 3.0 / 1.0

    def test_to_json(self):
        report = TuneReport(_make_results())
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            report.to_json(path)
            with open(path) as f:
                data = json.load(f)
            assert len(data) == 4
            assert "config_block_m" in data[0]
            assert "median_ms" in data[0]
        finally:
            Path(path).unlink()

    def test_to_csv(self):
        report = TuneReport(_make_results())
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            report.to_csv(path)
            lines = Path(path).read_text().strip().split("\n")
            assert len(lines) == 5  # header + 4 results
            assert "config_block_m" in lines[0]
        finally:
            Path(path).unlink()

    def test_heatmap_creates_figure(self):
        """Heatmap should return a matplotlib Figure."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for testing

        report = TuneReport(_make_results(), param_names=["block_m", "block_n"])
        fig = report.heatmap("block_m", "block_n")
        assert fig is not None

    def test_heatmap_save(self):
        """Heatmap should save to file."""
        import matplotlib

        matplotlib.use("Agg")

        report = TuneReport(_make_results(), param_names=["block_m", "block_n"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_heatmap.png"
            report.heatmap("block_m", "block_n", save_path=str(path))
            assert path.exists()

    def test_heatmap_invalid_param(self):
        """Invalid parameter name should raise ValueError."""
        import matplotlib

        matplotlib.use("Agg")

        report = TuneReport(_make_results(), param_names=["block_m", "block_n"])
        import pytest

        with pytest.raises(ValueError, match="Parameters must be"):
            report.heatmap("nonexistent", "block_n")

    def test_empty_results(self):
        report = TuneReport([])
        assert report.best(1) == []
        assert report.speedup_range == 1.0
