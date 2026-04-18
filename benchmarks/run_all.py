"""Run all benchmarks and generate a combined report."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BENCHMARKS = [
    "bench_matmul.py",
    "bench_rmsnorm.py",
    "bench_swiglu.py",
]


def main():
    bench_dir = Path(__file__).parent
    results_dir = bench_dir.parent / "results"
    results_dir.mkdir(exist_ok=True)

    for bench in BENCHMARKS:
        print(f"\n{'=' * 60}")
        print(f"Running {bench}")
        print(f"{'=' * 60}\n")

        result = subprocess.run(
            [sys.executable, str(bench_dir / bench)],
            cwd=str(bench_dir.parent),
        )
        if result.returncode != 0:
            print(f"WARNING: {bench} exited with code {result.returncode}")

    print(f"\n{'=' * 60}")
    print(f"All benchmarks complete. Results in {results_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
