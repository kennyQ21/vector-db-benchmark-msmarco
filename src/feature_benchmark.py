"""
Feature Comparison Benchmark — Tests 5 capabilities across FAISS, Qdrant, ES.

Features tested:
  1. Pure vector speed (reused from main benchmark)
  2. Hybrid search (BM25 + vector)
  3. Full-text search (BM25 only)
  4. Metadata filtering (category filter + vector)
  5. Production ecosystem (qualitative + throughput)

Thin orchestrator — all search logic lives in engine classes.
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import load_data
from src.engines.faiss_engine import FaissEngine
from src.engines.qdrant_engine import QdrantEngine
from src.engines.elasticsearch_engine import ElasticsearchEngine
from src.metrics import compute_latency_stats

# ─── Synthetic Metadata ──────────────────────────────────

CATEGORIES = ["tech", "health", "finance", "general"]


def add_categories(passages_df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic category column using deterministic seed."""
    rng = np.random.RandomState(config.RANDOM_SEED)
    passages_df = passages_df.copy()
    passages_df["category"] = rng.choice(CATEGORIES, size=len(passages_df))
    return passages_df


# ─── Recall Helper ────────────────────────────────────────

def compute_recall(all_results, queries_df, qrels_df, n_queries):
    """Compute Recall@10 for a subset of queries."""
    hits = 0
    total = 0
    for i in range(min(n_queries, len(queries_df))):
        qid = queries_df.iloc[i]["query_id"]
        relevant = set(qrels_df[qrels_df["query_id"] == qid]["passage_id"])
        if not relevant:
            continue
        retrieved = {r.doc_id for r in all_results[i]}
        hits += len(retrieved & relevant)
        total += len(relevant)
    return hits / total if total > 0 else 0


# ─── Latency Test Runner ─────────────────────────────────

def run_latency_test(search_fn, inputs, label=""):
    """Run a search function on inputs, return latency stats + results."""
    latencies = []
    all_results = []
    for inp in inputs:
        start = time.time()
        results = search_fn(*inp) if isinstance(inp, tuple) else search_fn(inp)
        latencies.append((time.time() - start) * 1000)
        all_results.append(results)

    stats = compute_latency_stats(latencies)
    return {
        "latency_p50_ms": round(stats["p50"], 2),
        "avg_results": round(np.mean([len(r) for r in all_results]), 1),
    }, all_results


# ─── Feature Results from Main Benchmark ─────────────────

def load_main_results(engine_name, scale, results_path):
    """Pull results from existing benchmark_results.json."""
    try:
        with open(results_path) as f:
            data = json.load(f)
        for r in data["results"]:
            if r["engine"] == engine_name and r["scale"] == scale:
                return {
                    "supported": True,
                    "recall_at_10": r["recall_at_10"],
                    "latency_p50_ms": r["latency_p50_ms"],
                }
    except Exception:
        pass
    return {"supported": True, "recall_at_10": 0, "latency_p50_ms": 0}


# ─── Star Rating Matrix ──────────────────────────────────

FEATURE_MATRIX = {
    "Pure Vector Speed": {
        "FAISS": {"stars": 5, "note": "In-memory, sub-ms latency"},
        "Qdrant": {"stars": 4, "note": "HNSW, low-ms latency"},
        "Elasticsearch": {"stars": 2, "note": "Brute-force script_score (ES 7.x)"},
    },
    "Hybrid Search": {
        "FAISS": {"stars": 0, "note": "Not supported"},
        "Qdrant": {"stars": 2, "note": "Sparse+dense possible, not native BM25"},
        "Elasticsearch": {"stars": 5, "note": "Native BM25 + vector scoring"},
    },
    "Full-Text Search": {
        "FAISS": {"stars": 0, "note": "Not supported"},
        "Qdrant": {"stars": 0, "note": "Not supported"},
        "Elasticsearch": {"stars": 5, "note": "Full Lucene-based BM25"},
    },
    "Metadata Filtering": {
        "FAISS": {"stars": 2, "note": "Post-filter only (over-fetch required)"},
        "Qdrant": {"stars": 4, "note": "Native payload filter in HNSW"},
        "Elasticsearch": {"stars": 5, "note": "Native bool filter + vector"},
    },
    "Production Ecosystem": {
        "FAISS": {"stars": 1, "note": "Library only, no built-in serving"},
        "Qdrant": {"stars": 3, "note": "Managed cloud, REST/gRPC API"},
        "Elasticsearch": {"stars": 5, "note": "Full ecosystem: Kibana, security, monitoring"},
    },
}


# ─── Main ─────────────────────────────────────────────────

def run_feature_benchmark(scale: int = 10_000, num_queries: int = 100):
    """Run all feature tests and produce results."""
    print(f"\n{'='*60}")
    print(f"  🧪 Feature Comparison Benchmark — {scale:,} passages")
    print(f"{'='*60}\n")

    # ─── Load Data ────────────────────────────────────────
    passages_df, queries_df, qrels_df = load_data(scale)
    passages_df = add_categories(passages_df)
    passage_embs = np.load(f"data/embeddings/passages_{scale}.npy")
    query_embs = np.load(f"data/embeddings/queries_{scale}.npy")

    doc_ids = passages_df["passage_id"].tolist()
    texts = passages_df["passage_text"].tolist()
    categories = passages_df["category"].tolist()
    query_texts = queries_df["query_text"].tolist()
    metadata = {"category": categories}

    n_q = min(num_queries, len(query_embs))
    q_embs_sub = query_embs[:n_q]
    q_texts_sub = query_texts[:n_q]
    target_cat = "tech"

    results = {}
    results_path = Path(config.RESULTS_DIR) / "benchmark_results.json"

    # ─── 1. Pure Vector Speed ─────────────────────────────
    print("  1️⃣  Pure Vector Speed (from existing benchmark)...")
    results["pure_vector"] = {
        "FAISS": load_main_results("FAISS-FLAT", scale, results_path),
        "Qdrant": load_main_results("Qdrant", scale, results_path),
        "Elasticsearch": load_main_results("ES-Vector", scale, results_path),
    }
    for eng, r in results["pure_vector"].items():
        print(f"     {eng}: P50={r['latency_p50_ms']}ms, Recall@10={r['recall_at_10']}")

    # ─── 2. Hybrid Search (ES only) ──────────────────────
    print("\n  2️⃣  Hybrid Search (BM25 + Vector)...")
    es_hybrid = ElasticsearchEngine(hybrid=True)
    es_hybrid.index(doc_ids, passage_embs, texts=texts, metadata=metadata)

    stats, hybrid_results = run_latency_test(
        lambda qe, qt: es_hybrid.search_hybrid(qe, qt),
        list(zip(q_embs_sub, q_texts_sub)),
    )
    hybrid_recall = compute_recall(hybrid_results, queries_df, qrels_df, n_q)
    es_hybrid.cleanup()

    results["hybrid"] = {
        "FAISS": {"supported": False, "note": "No text search capability"},
        "Qdrant": {"supported": False, "note": "Limited sparse+dense, no native BM25"},
        "Elasticsearch": {
            "supported": True,
            "method": "BM25 + script_score (real queries)",
            "recall_at_10": round(hybrid_recall, 4),
            **stats,
        },
    }
    for eng, r in results["hybrid"].items():
        if r["supported"]:
            print(f"     {eng}: P50={r['latency_p50_ms']}ms, Recall@10={r['recall_at_10']}")
        else:
            print(f"     {eng}: ❌ Not supported")

    # ─── 3. Full-Text Search (ES only) ───────────────────
    print("\n  3️⃣  Full-Text Search (BM25 Only)...")
    es_ft = ElasticsearchEngine()
    es_ft.index(doc_ids, passage_embs, texts=texts)

    stats, ft_results = run_latency_test(
        es_ft.search_fulltext, q_texts_sub,
    )
    ft_recall = compute_recall(ft_results, queries_df, qrels_df, n_q)
    es_ft.cleanup()

    results["fulltext"] = {
        "FAISS": {"supported": False, "note": "Vector-only library"},
        "Qdrant": {"supported": False, "note": "Vector-only database"},
        "Elasticsearch": {
            "supported": True,
            "method": "BM25 match query",
            "recall_at_10": round(ft_recall, 4),
            **stats,
        },
    }
    for eng, r in results["fulltext"].items():
        if r["supported"]:
            print(f"     {eng}: P50={r['latency_p50_ms']}ms, Recall@10={r['recall_at_10']}")
        else:
            print(f"     {eng}: ❌ Not supported")

    # ─── 4. Metadata Filtering ────────────────────────────
    print(f"\n  4️⃣  Metadata Filtering (category='{target_cat}')...")

    # FAISS — post-filter
    faiss_eng = FaissEngine()
    faiss_eng.index(doc_ids, passage_embs, metadata=metadata)
    stats_faiss, _ = run_latency_test(
        lambda qe: faiss_eng.search_with_filter(qe, "category", target_cat),
        q_embs_sub,
    )
    faiss_eng.cleanup()

    # Qdrant — native payload filter
    qdrant_eng = QdrantEngine()
    qdrant_eng.index(doc_ids, passage_embs, metadata=metadata)
    stats_qdrant, _ = run_latency_test(
        lambda qe: qdrant_eng.search_with_filter(qe, "category", target_cat),
        q_embs_sub,
    )
    qdrant_eng.cleanup()

    # Elasticsearch — bool filter + vector
    es_filt = ElasticsearchEngine()
    es_filt.index(doc_ids, passage_embs, texts=texts, metadata=metadata)
    stats_es, _ = run_latency_test(
        lambda qe: es_filt.search_with_filter(qe, "category", target_cat),
        q_embs_sub,
    )
    es_filt.cleanup()

    results["filtering"] = {
        "FAISS": {
            "supported": True,
            "method": f"post-filter (over-fetch {FaissEngine.OVERFETCH_FACTOR}×)",
            "overfetch_factor": FaissEngine.OVERFETCH_FACTOR,
            **stats_faiss,
        },
        "Qdrant": {"supported": True, "method": "native payload filter", **stats_qdrant},
        "Elasticsearch": {"supported": True, "method": "native bool filter + script_score", **stats_es},
    }
    for eng, r in results["filtering"].items():
        print(f"     {eng}: P50={r['latency_p50_ms']}ms ({r['method']}), avg_results={r['avg_results']}")

    # ─── 5. Production Ecosystem ──────────────────────────
    print("\n  5️⃣  Production Ecosystem (qualitative + throughput)...")
    throughputs = {}
    try:
        with open(results_path) as f:
            bench_data = json.load(f)
        for r in bench_data["results"]:
            if r["scale"] == scale:
                key = {"FAISS-FLAT": "FAISS", "Qdrant": "Qdrant",
                       "ES-Vector": "Elasticsearch"}.get(r["engine"])
                if key:
                    throughputs[key] = round(r["indexing_throughput"], 0)
    except Exception:
        pass

    results["ecosystem"] = {
        "FAISS": {
            "score": 1, "note": "Library only — no serving, no auth, no monitoring",
            "indexing_throughput_docs_per_sec": throughputs.get("FAISS", 0),
        },
        "Qdrant": {
            "score": 3, "note": "Managed cloud, REST/gRPC, snapshots, basic monitoring",
            "indexing_throughput_docs_per_sec": throughputs.get("Qdrant", 0),
        },
        "Elasticsearch": {
            "score": 5, "note": "Full stack: Kibana, security, alerting, cross-cluster",
            "indexing_throughput_docs_per_sec": throughputs.get("Elasticsearch", 0),
        },
    }
    for eng, r in results["ecosystem"].items():
        tp = r.get("indexing_throughput_docs_per_sec", 0)
        tp_str = f", {tp:,.0f} docs/s" if tp else ""
        print(f"     {eng}: {'⭐'*r['score']} — {r['note']}{tp_str}")

    # ─── Save & Print ─────────────────────────────────────
    output = {
        "scale": scale,
        "num_queries": n_q,
        "feature_results": results,
        "star_matrix": FEATURE_MATRIX,
    }
    out_path = Path(config.RESULTS_DIR) / "feature_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Feature results saved to {out_path}")

    print(f"\n{'='*70}")
    print(f"  ⭐ FEATURE COMPARISON MATRIX")
    print(f"{'='*70}")
    print(f"\n  {'Feature':<25} {'FAISS':>10} {'Qdrant':>10} {'Elastic':>10}")
    print(f"  {'─'*23}   {'─'*10} {'─'*10} {'─'*10}")
    for feature, engines in FEATURE_MATRIX.items():
        f_stars = "⭐" * engines["FAISS"]["stars"] or "  ❌"
        q_stars = "⭐" * engines["Qdrant"]["stars"] or "  ❌"
        e_stars = "⭐" * engines["Elasticsearch"]["stars"] or "  ❌"
        print(f"  {feature:<25} {f_stars:>10} {q_stars:>10} {e_stars:>10}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Feature Comparison Benchmark")
    parser.add_argument("--scale", type=int, default=10_000)
    parser.add_argument("--num-queries", type=int, default=100)
    args = parser.parse_args()
    run_feature_benchmark(args.scale, args.num_queries)
