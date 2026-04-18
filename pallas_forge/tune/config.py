"""Configuration space definition for auto-tuning.

Defines the search space over which the auto-tuner explores kernel configurations.
Supports both Python dict and YAML-based definitions, with optional constraints
to filter out invalid or known-bad combinations.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml


@dataclass
class TuneParam:
    """A single tunable parameter with its possible values."""

    name: str
    values: list[int | float | str]

    def __post_init__(self):
        if not self.values:
            raise ValueError(f"TuneParam '{self.name}' must have at least one value")


@dataclass
class TuneConfig:
    """Defines the search space for auto-tuning.

    A TuneConfig holds a list of tunable parameters and optional constraints.
    Constraints are callables that take a parameter dict and return True if
    the combination is valid.

    Example::

        config = TuneConfig.from_dict({
            "block_m": [64, 128, 256],
            "block_n": [64, 128, 256],
            "block_k": [64, 128],
        })
        config.add_constraint(lambda p: p["block_m"] >= p["block_k"])
    """

    params: list[TuneParam]
    constraints: list[Callable[[dict[str, Any]], bool]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, list]) -> TuneConfig:
        """Create a TuneConfig from a dict mapping param names to value lists."""
        params = [TuneParam(name=k, values=v) for k, v in d.items()]
        return cls(params=params)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TuneConfig:
        """Load a TuneConfig from a YAML file.

        Expected format::

            block_m: [64, 128, 256]
            block_n: [64, 128, 256]
            block_k: [64, 128]
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def add_constraint(self, fn: Callable[[dict[str, Any]], bool]) -> None:
        """Add a constraint function. It receives a config dict and returns True if valid."""
        self.constraints.append(fn)

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self.params]

    @property
    def total_combinations(self) -> int:
        """Total combinations before constraint filtering."""
        n = 1
        for p in self.params:
            n *= len(p.values)
        return n

    def _is_valid(self, config: dict[str, Any]) -> bool:
        return all(c(config) for c in self.constraints)

    def grid(self) -> list[dict[str, Any]]:
        """Generate all valid parameter combinations (cartesian product filtered by constraints)."""
        names = [p.name for p in self.params]
        value_lists = [p.values for p in self.params]

        all_combos = [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]
        return [c for c in all_combos if self._is_valid(c)]

    def sample(self, n: int, seed: int = 42) -> list[dict[str, Any]]:
        """Randomly sample n valid parameter combinations.

        Uses rejection sampling: generate random combos until n valid ones found.
        Falls back to returning all valid combos if the space is smaller than n.
        """
        rng = random.Random(seed)
        names = [p.name for p in self.params]
        value_lists = [p.values for p in self.params]

        # If space is small, just filter the full grid
        if self.total_combinations <= n * 2:
            valid = self.grid()
            rng.shuffle(valid)
            return valid[:n]

        results: list[dict[str, Any]] = []
        seen: set[tuple] = set()
        max_attempts = n * 20

        for _ in range(max_attempts):
            if len(results) >= n:
                break
            combo = tuple(rng.choice(vals) for vals in value_lists)
            if combo in seen:
                continue
            seen.add(combo)
            config = dict(zip(names, combo))
            if self._is_valid(config):
                results.append(config)

        return results
