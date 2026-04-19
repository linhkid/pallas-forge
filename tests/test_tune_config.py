"""Tests for the TuneConfig configuration space."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pallas_forge.tune.config import TuneConfig, TuneParam


class TestTuneParam:
    def test_basic(self):
        p = TuneParam(name="block_m", values=[64, 128, 256])
        assert p.name == "block_m"
        assert p.values == [64, 128, 256]

    def test_empty_values_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            TuneParam(name="empty", values=[])


class TestTuneConfig:
    def test_from_dict(self):
        config = TuneConfig.from_dict(
            {
                "block_m": [64, 128],
                "block_n": [128, 256],
            }
        )
        assert len(config.params) == 2
        assert config.param_names == ["block_m", "block_n"]

    def test_grid_exhaustive(self):
        config = TuneConfig.from_dict(
            {
                "a": [1, 2],
                "b": [3, 4],
            }
        )
        grid = config.grid()
        assert len(grid) == 4
        assert {"a": 1, "b": 3} in grid
        assert {"a": 2, "b": 4} in grid

    def test_total_combinations(self):
        config = TuneConfig.from_dict(
            {
                "a": [1, 2, 3],
                "b": [4, 5],
            }
        )
        assert config.total_combinations == 6

    def test_constraints_filter(self):
        config = TuneConfig.from_dict(
            {
                "a": [1, 2, 3],
                "b": [1, 2, 3],
            }
        )
        config.add_constraint(lambda p: p["a"] >= p["b"])
        grid = config.grid()
        assert all(c["a"] >= c["b"] for c in grid)
        assert len(grid) == 6  # (1,1), (2,1), (2,2), (3,1), (3,2), (3,3)

    def test_sample_count(self):
        config = TuneConfig.from_dict(
            {
                "a": list(range(10)),
                "b": list(range(10)),
            }
        )
        samples = config.sample(5, seed=42)
        assert len(samples) == 5

    def test_sample_reproducibility(self):
        config = TuneConfig.from_dict(
            {
                "a": list(range(10)),
                "b": list(range(10)),
            }
        )
        s1 = config.sample(5, seed=42)
        s2 = config.sample(5, seed=42)
        assert s1 == s2

    def test_from_yaml(self):
        yaml_content = "block_m: [64, 128]\nblock_n: [128, 256]\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            config = TuneConfig.from_yaml(path)
            assert config.param_names == ["block_m", "block_n"]
            assert config.total_combinations == 4
        finally:
            Path(path).unlink()

    def test_single_param(self):
        config = TuneConfig.from_dict({"block_size": [128, 256, 512]})
        grid = config.grid()
        assert len(grid) == 3
        assert all("block_size" in c for c in grid)
