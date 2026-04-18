"""Search strategies for auto-tuning.

Provides GridSearch and RandomSearch as implementations of the SearchStrategy protocol.
Custom strategies can be added by implementing the same interface.
"""

from __future__ import annotations

from typing import Any, Protocol

from pallas_forge.tune.config import TuneConfig


class SearchStrategy(Protocol):
    """Protocol for tuning search strategies.

    Any object with a `generate` method that takes a TuneConfig and returns
    a list of parameter dicts can be used as a search strategy.
    """

    def generate(self, config: TuneConfig) -> list[dict[str, Any]]: ...


class GridSearch:
    """Exhaustive search over all valid parameter combinations."""

    def generate(self, config: TuneConfig) -> list[dict[str, Any]]:
        return config.grid()


class RandomSearch:
    """Random sampling from the configuration space.

    Args:
        n_trials: Number of configurations to sample.
        seed: Random seed for reproducibility.
    """

    def __init__(self, n_trials: int = 50, seed: int = 42):
        self.n_trials = n_trials
        self.seed = seed

    def generate(self, config: TuneConfig) -> list[dict[str, Any]]:
        return config.sample(self.n_trials, self.seed)
