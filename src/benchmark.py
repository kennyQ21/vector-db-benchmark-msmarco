"""
Benchmark Orchestrator — Run all engines at specified scales.

For each (engine, scale) combination:
  1. Load/subset data
  2. Load cached embeddings
  3. Index passages → measure indexing time & memory
  4. Run evaluation queries → measure per-query latency
  5. Compute Recall@k, MRR@10
  6. Save results as JSON

Also captures system info for reproducibility.
"""

import sys
import json
import time
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import load_data
from src.embedder import Embedder
from src.metrics import BenchmarkResult, evaluate_retrieval, compute_latency_stats
from src.engines.base import SearchEngine
from src.engines.faiss_engine import FaissEngine
from src.engines.qdrant_engine import QdrantEngine
from src.engines.elasticsearch_engine import ElasticsearchEngine


def get_system_info() -> dict:
    """Capture system info for benchmark reproducibility."""
    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "python_version": platform.python_version(),
        "embedding_model": config.EMBEDDING_MODEL,
        "random_seed": config.RANDOM_SEED,
    }


def create_engines(engine_names: list[str]) -> list[SearchEngine]:
    """Instantiate requested engines."""
    engines = []
    for name in engine_names:
        if name == "faiss":
            for idx_type in config.FAISS_INDEX_TYPES:
                engines.append(FaissEngine(index_type=idx_type))
        elif name == "qdrant":
            engines.append(QdrantEngine())
        elif name == "elasticsearch":
            engines.append(ElasticsearchEngine(hybrid=False))
            engines.append(ElasticsearchEngine(hybrid=True))
        else:
            print(f"  ⚠️ Unknown engine: {name}, skipping")
    return engines


def run_single_benchmark(
    engine: SearchEngine,
    passages_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    passage_embs: np.ndarray,
    query_embs: np.ndarray,
    qrels_df: pd.DataFrame,
    scale: int,
    top_k_values: list[int],
    mrr_k: int,
) -> BenchmarkResult:
    """
    Run a complete benchmark for one engine at one scale.

    Returns:
        BenchmarkResult with all metrics
    """
    print(f"\n{'─'*50}")
    print(f"  🏋️ {engine.name()} @ {scale:,} passages")
    print(f"{'─'*50}")

    # 1. Index
    stats = engine.index(
        doc_ids=passages_df["passage_id"].tolist(),
        embeddings=passage_embs,
        texts=passages_df["passage_text"].tolist(),
    )
    throughput = stats.num_documents / stats.index_time_sec if stats.index_time_sec > 0 else 0

    # 2. Query + measure latency
    print(f"  🔍 Querying {len(queries_df)} queries...")
    latencies = []
    all_retrieved = []

    is_hybrid = hasattr(engine, 'hybrid') and engine.hybrid

    for i, row in queries_df.iterrows():
        start = time.time()
        if is_hybrid:
            results = engine.search(
                query_embs[i], top_k=max(top_k_values),
                query_text=row["query_text"],
            )
        else:
            results = engine.search(query_embs[i], top_k=max(top_k_values))

        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)
        all_retrieved.append([r.doc_id for r in results])

    # 3. Build relevance sets per query
    all_relevant = []
    for i, row in queries_df.iterrows():
        qid = row["query_id"]
        relevant = set(qrels_df[qrels_df["query_id"] == qid]["passage_id"])
        all_relevant.append(relevant)

    # 4. Compute metrics
    retrieval_metrics = evaluate_retrieval(
        all_retrieved, all_relevant,
        k_values=top_k_values, mrr_k=mrr_k,
    )
    latency_stats = compute_latency_stats(latencies)

    # 5. Build result
    result = BenchmarkResult(
        engine_name=engine.name(),
        scale=scale,
        index_time_sec=stats.index_time_sec,
        memory_usage_mb=stats.memory_usage_mb,
        indexing_throughput=throughput,
        recall_at_10=retrieval_metrics.get("recall_at_10", 0),
        recall_at_50=retrieval_metrics.get("recall_at_50", 0),
        recall_at_100=retrieval_metrics.get("recall_at_100", 0),
        mrr_at_10=retrieval_metrics.get("mrr_at_10", 0),
        latency_mean_ms=latency_stats["mean"],
        latency_p50_ms=latency_stats["p50"],
        latency_p95_ms=latency_stats["p95"],
        latency_std_ms=latency_stats["std"],
    )

    # Print summary
    print(f"\n  📊 Results for {engine.name()} @ {scale:,}:")
    print(f"     Index:   {result.index_time_sec:.2f}s ({result.indexing_throughput:.0f} docs/s)")
    print(f"     Memory:  {result.memory_usage_mb:.1f} MB")
    print(f"     Recall@10: {result.recall_at_10:.4f}")
    print(f"     MRR@10:    {result.mrr_at_10:.4f}")
    print(f"     Latency:   P50={result.latency_p50_ms:.2f}ms  P95={result.latency_p95_ms:.2f}ms")

    # 6. Cleanup
    engine.cleanup()

    return result


def run_benchmark(
    engine_names: list[str],
    scales: list[int],
    num_queries: int = None,
    top_k: int = None,
):
    """
    Run the full benchmark across engines and scales.

    Args:
        engine_names: ["faiss", "qdrant", "elasticsearch"]
        scales:       [10000, 50000, 100000]
    """
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    top_k_values = config.TOP_K_VALUES
    mrr_k = config.MRR_K

    # System info
    sys_info = get_system_info()

    print("=" * 60)
    print("  🏆 Vector Search Benchmark")
    print("  Elasticsearch vs FAISS vs Qdrant on MS MARCO")
    print("=" * 60)
    print(f"\n  System:    {sys_info['platform']}")
    print(f"  CPU:       {sys_info['processor']} ({sys_info['cpu_count']} cores)")
    print(f"  RAM:       {sys_info['ram_total_gb']} GB")
    print(f"  Model:     {sys_info['embedding_model']}")
    print(f"  Scales:    {scales}")
    print(f"  Engines:   {engine_names}")
    print()

    # Initialize embedder (for generating embeddings at new scales)
    embedder = Embedder()

    all_results = []

    for scale in scales:
        print(f"\n{'='*60}")
        print(f"  📏 SCALE: {scale:,} passages")
        print(f"{'='*60}")

        # Load data
        passages_df, queries_df, qrels_df = load_data(scale)

        # Load/generate embeddings
        passage_embs = embedder.encode_passages(
            passages_df["passage_text"].tolist(), scale=scale,
        )
        query_embs = embedder.encode_queries(
            queries_df["query_text"].tolist(), scale=scale,
        )

        # Run each engine
        engines = create_engines(engine_names)
        for engine in engines:
            try:
                result = run_single_benchmark(
                    engine=engine,
                    passages_df=passages_df,
                    queries_df=queries_df,
                    passage_embs=passage_embs,
                    query_embs=query_embs,
                    qrels_df=qrels_df,
                    scale=scale,
                    top_k_values=top_k_values,
                    mrr_k=mrr_k,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  ❌ {engine.name()} failed: {e}")
                import traceback
                traceback.print_exc()

    # Save all results
    output = {
        "system_info": sys_info,
        "results": [r.to_dict() for r in all_results],
    }

    output_path = results_dir / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Results saved to {output_path}")

    # Print final summary table
    print(f"\n{'='*60}")
    print(f"  📊 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Engine':<15} {'Scale':>8} {'Recall@10':>10} {'MRR@10':>8} {'P50(ms)':>8} {'P95(ms)':>8} {'Idx(s)':>7}")
    print(f"  {'─'*13}   {'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")
    for r in all_results:
        print(f"  {r.engine_name:<15} {r.scale:>8,} {r.recall_at_10:>10.4f} {r.mrr_at_10:>8.4f} "
              f"{r.latency_p50_ms:>8.2f} {r.latency_p95_ms:>8.2f} {r.index_time_sec:>7.2f}")

    return all_results
