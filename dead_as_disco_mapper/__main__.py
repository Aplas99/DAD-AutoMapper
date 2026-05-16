import sys

from .app import main
from .benchmarking import run_benchmark_cli


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        raise SystemExit(run_benchmark_cli(sys.argv[2:]))
    raise SystemExit(main())
