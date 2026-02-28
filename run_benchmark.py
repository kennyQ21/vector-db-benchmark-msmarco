#!/usr/bin/env python3
"""
Vector Search Benchmark — CLI Entry Point

Usage:
    python run_benchmark.py --engines faiss qdrant elasticsearch --scales 10000 50000 100000
    python run_benchmark.py --engines faiss --scales 10000        # single engine test
    python run_benchmark.py                                        # all engines, all scales
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.benchmark import run_benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vector Search Benchmark: Elasticsearch vs FAISS vs Qdrant"
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=["faiss", "qdrant", "elasticsearch"],
        default=["faiss", "qdrant", "elasticsearch"],
        help="Engines to benchmark (default: all three)",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=int,
        default=[10_000],
        help="Dataset sizes to test (default: 10000)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=500,
        help="Number of evaluation queries (default: 500)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    results = run_benchmark(
        engine_names=args.engines,
        scales=args.scales,
        num_queries=args.num_queries,
    )

    print(f"\n✅ Benchmark complete! {len(results)} results generated.")


if __name__ == "__main__":
    main()
