from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcompute.benchmarking import print_benchmark_table, run_default_suite


def main() -> None:
    """Execute the default benchmark suite and print environment + table."""
    rows, env_note = run_default_suite()
    print(env_note)
    print()
    print_benchmark_table(rows)


if __name__ == "__main__":
    main()
