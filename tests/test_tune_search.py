"""Tests for search strategies."""

from __future__ import annotations

from pallas_forge.tune.config import TuneConfig
from pallas_forge.tune.search import GridSearch, RandomSearch


class TestGridSearch:
    def test_returns_all_combos(self):
        config = TuneConfig.from_dict({"a": [1, 2], "b": [3, 4]})
        strategy = GridSearch()
        results = strategy.generate(config)
        assert len(results) == 4

    def test_respects_constraints(self):
        config = TuneConfig.from_dict({"a": [1, 2, 3], "b": [1, 2, 3]})
        config.add_constraint(lambda p: p["a"] > p["b"])
        strategy = GridSearch()
        results = strategy.generate(config)
        assert all(r["a"] > r["b"] for r in results)


class TestRandomSearch:
    def test_returns_requested_count(self):
        config = TuneConfig.from_dict({"a": list(range(20)), "b": list(range(20))})
        strategy = RandomSearch(n_trials=10, seed=42)
        results = strategy.generate(config)
        assert len(results) == 10

    def test_reproducible(self):
        config = TuneConfig.from_dict({"a": list(range(20)), "b": list(range(20))})
        s1 = RandomSearch(n_trials=10, seed=42).generate(config)
        s2 = RandomSearch(n_trials=10, seed=42).generate(config)
        assert s1 == s2

    def test_different_seeds_differ(self):
        config = TuneConfig.from_dict({"a": list(range(20)), "b": list(range(20))})
        s1 = RandomSearch(n_trials=10, seed=42).generate(config)
        s2 = RandomSearch(n_trials=10, seed=99).generate(config)
        assert s1 != s2
