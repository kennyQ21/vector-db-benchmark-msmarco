"""
Evaluation Metrics — Recall@k, MRR@k, Latency Statistics.

Standard IR metrics for benchmarking vector search quality and performance.
All metrics are computed per-query then averaged.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Complete benchmark result for one engine at one scale."""
    engine_name: str
    scale: int

    # Indexing
    index_time_sec: float = 0.0
    memory_usage_mb: float = 0.0
    indexing_throughput: float = 0.0  # docs/sec

    # Retrieval quality
    recall_at_10: float = 0.0
    recall_at_50: float = 0.0
    recall_at_100: float = 0.0
    mrr_at_10: float = 0.0

    # Latency (ms)
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_std_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "engine": self.engine_name,
            "scale": self.scale,
            "index_time_sec": round(self.index_time_sec, 3),
            "memory_usage_mb": round(self.memory_usage_mb, 1),
            "indexing_throughput": round(self.indexing_throughput, 1),
            "recall_at_10": round(self.recall_at_10, 4),
            "recall_at_50": round(self.recall_at_50, 4),
            "recall_at_100": round(self.recall_at_100, 4),
            "mrr_at_10": round(self.mrr_at_10, 4),
            "latency_mean_ms": round(self.latency_mean_ms, 2),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "latency_std_ms": round(self.latency_std_ms, 2),
        }


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """
    Recall@k: fraction of relevant documents found in the top-k results.

    Args:
        retrieved_ids: ordered list of retrieved document IDs
        relevant_ids:  set of ground-truth relevant document IDs
        k:             cutoff position

    Returns:
        recall score between 0.0 and 1.0
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def mrr_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """
    Mean Reciprocal Rank@k: 1/rank of the first relevant result in top-k.

    Args:
        retrieved_ids: ordered list of retrieved document IDs
        relevant_ids:  set of ground-truth relevant document IDs
        k:             cutoff position

    Returns:
        reciprocal rank between 0.0 and 1.0
    """
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_latency_stats(latencies_ms: list[float]) -> dict:
    """
    Compute latency percentiles and statistics.

    Args:
        latencies_ms: list of query latencies in milliseconds

    Returns:
        dict with mean, p50, p95, std
    """
    arr = np.array(latencies_ms)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(arr.std()),
    }


def compute_recall_from_results(all_results, queries_df, qrels_df, top_k: int) -> float:
    """
    Compute Recall@k over a set of query SearchResults.

    Args:
        all_results: list of lists of SearchResult objects
        queries_df: dataframe containing queries
        qrels_df: dataframe containing relevance judgments
        top_k: integer cutoff value

    Returns:
        recall score between 0.0 and 1.0
    """
    hits = 0
    total = 0
    for i in range(min(len(all_results), len(queries_df))):
        qid = queries_df.iloc[i]["query_id"]
        relevant = set(qrels_df[qrels_df["query_id"] == qid]["passage_id"])
        if not relevant:
            continue
        retrieved = {r.doc_id for r in all_results[i][:top_k]}
        hits += len(retrieved & relevant)
        total += len(relevant)
    return hits / total if total > 0 else 0


def evaluate_retrieval(
    all_retrieved: list[list[int]],
    all_relevant: list[set[int]],
    k_values: list[int] = [10, 50, 100],
    mrr_k: int = 10,
) -> dict:
    """
    Compute all retrieval metrics averaged over queries.

    Args:
        all_retrieved: list of retrieved ID lists (one per query)
        all_relevant:  list of relevant ID sets (one per query)
        k_values:      Recall@k values to compute
        mrr_k:         MRR cutoff

    Returns:
        dict with recall_at_{k} and mrr_at_{mrr_k}
    """
    n = len(all_retrieved)
    results = {}

    for k in k_values:
        recalls = [recall_at_k(all_retrieved[i], all_relevant[i], k) for i in range(n)]
        results[f"recall_at_{k}"] = float(np.mean(recalls))

    mrrs = [mrr_at_k(all_retrieved[i], all_relevant[i], mrr_k) for i in range(n)]
    results[f"mrr_at_{mrr_k}"] = float(np.mean(mrrs))

    return results
