"""
Advanced Benchmark Experiments — Research-grade improvements.

Experiments:
  1. Qdrant ef_search sweep (recall–latency tradeoff)
  2. FAISS HNSW vs Flat (ANN-vs-exact comparison)
  3. Recall@k curves (k = 1, 5, 10, 50, 100)
  4. Memory / disk reporting (Qdrant disk size)
  5. Concurrency test (1 vs 4 vs 8 workers)
  6. Qdrant m=16 vs m=32 ef sweep at 100k
  7. Hybrid confusion matrix (both succeed / both fail counts)
  8. FAISS HNSW explicit param logging
"""

import sys
import json
import time
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import faiss
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import load_data
from src.engines.base import SearchResult
from src.engines.faiss_engine import FaissEngine
from src.engines.qdrant_engine import QdrantEngine
from src.engines.elasticsearch_engine import ElasticsearchEngine
from src.metrics import compute_latency_stats, compute_recall_from_results


# ═══════════════════════════════════════════════════════════
#  Experiment 1 — Qdrant ef_search Sweep
# ═══════════════════════════════════════════════════════════

def run_qdrant_ef_sweep(doc_ids, embeddings, query_embs, qrels_df, queries_df,
                        ef_values=(32, 64, 128, 256), top_k=10):
    """Sweep ef_search in Qdrant to show recall–latency tradeoff."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        HnswConfigDiff, OptimizersConfigDiff, SearchParams,
    )

    print(f"\n{'='*60}")
    print(f"  🔬 Experiment 1 — Qdrant ef_search Sweep")
    print(f"{'='*60}\n")

    storage = "data/qdrant_ef_sweep"
    if Path(storage).exists():
        shutil.rmtree(storage)

    client = QdrantClient(path=storage)
    n, dim = embeddings.shape

    client.create_collection(
        collection_name="ef_sweep",
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
    )

    batch_size = 500
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        points = [
            PointStruct(id=j, vector=embeddings[j].tolist(),
                        payload={"doc_id": int(doc_ids[j])})
            for j in range(i, end)
        ]
        client.upsert(collection_name="ef_sweep", points=points)

    print(f"  ✅ Indexed {n:,} points\n")

    results = []
    for ef in ef_values:
        latencies = []
        all_results = []
        for qe in query_embs:
            start = time.time()
            hits = client.query_points(
                collection_name="ef_sweep",
                query=qe.tolist(),
                limit=top_k,
                search_params=SearchParams(hnsw_ef=ef),
            )
            latencies.append((time.time() - start) * 1000)
            all_results.append([
                SearchResult(doc_id=int(h.payload["doc_id"]), score=float(h.score))
                for h in hits.points
            ])

        stats = compute_latency_stats(latencies)
        recall = compute_recall_from_results(all_results, queries_df, qrels_df, top_k)

        row = {"ef_search": ef, "recall_at_10": round(recall, 4),
               "latency_p50_ms": round(stats["p50"], 2),
               "latency_p95_ms": round(stats["p95"], 2)}
        results.append(row)
        print(f"  ef={ef:>4}  →  Recall@10={recall:.4f}  P50={stats['p50']:.2f}ms  P95={stats['p95']:.2f}ms")

    client.delete_collection("ef_sweep")
    client.close()
    shutil.rmtree(storage, ignore_errors=True)

    return results


# ═══════════════════════════════════════════════════════════
#  Experiment 2 — FAISS HNSW vs Flat
# ═══════════════════════════════════════════════════════════

def run_faiss_hnsw_comparison(doc_ids, embeddings, query_embs, qrels_df, queries_df, top_k=10):
    """Compare FAISS Flat (exact) vs HNSW (ANN)."""
    print(f"\n{'='*60}")
    print(f"  🔬 Experiment 2 — FAISS HNSW vs Flat")
    print(f"{'='*60}\n")

    n, dim = embeddings.shape
    vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
    doc_id_arr = np.array(doc_ids, dtype=np.int64)
    results = []

    for label, build_fn in [
        ("FAISS-Flat", lambda: _build_flat(vecs, dim)),
        ("FAISS-HNSW-16", lambda: _build_hnsw(vecs, dim, m=16)),
        ("FAISS-HNSW-32", lambda: _build_hnsw(vecs, dim, m=32)),
    ]:
        start = time.time()
        index = build_fn()
        idx_time = time.time() - start

        latencies = []
        all_results = []
        for qe in query_embs:
            s = time.time()
            q = qe.reshape(1, -1).astype(np.float32)
            scores, indices = index.search(q, top_k)
            latencies.append((time.time() - s) * 1000)
            all_results.append([
                SearchResult(doc_id=int(doc_id_arr[i]), score=float(sc))
                for sc, i in zip(scores[0], indices[0]) if i != -1
            ])

        stats = compute_latency_stats(latencies)
        recall = compute_recall_from_results(all_results, queries_df, qrels_df, top_k)

        row = {"engine": label, "index_time_sec": round(idx_time, 3),
               "recall_at_10": round(recall, 4),
               "latency_p50_ms": round(stats["p50"], 2),
               "latency_p95_ms": round(stats["p95"], 2)}
        results.append(row)
        print(f"  {label:<16} →  Recall@10={recall:.4f}  P50={stats['p50']:.2f}ms  "
              f"Index={idx_time:.3f}s")

    return results


def _build_flat(vecs, dim):
    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)
    return idx


def _build_hnsw(vecs, dim, m=16):
    idx = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = 200
    idx.hnsw.efSearch = 64
    idx.add(vecs)
    return idx


# ═══════════════════════════════════════════════════════════
#  Experiment 3 — Recall@k Curves
# ═══════════════════════════════════════════════════════════

def run_recall_at_k_curves(doc_ids, embeddings, query_embs, qrels_df, queries_df,
                           k_values=(1, 5, 10, 50, 100)):
    """Compute Recall@k for each engine at multiple k values."""
    print(f"\n{'='*60}")
    print(f"  🔬 Experiment 3 — Recall@k Curves")
    print(f"{'='*60}\n")

    n, dim = embeddings.shape
    vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
    doc_id_arr = np.array(doc_ids, dtype=np.int64)
    max_k = max(k_values)
    results = {}

    # FAISS Flat
    print("  Running FAISS Flat...")
    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)
    faiss_results = []
    for qe in query_embs:
        q = qe.reshape(1, -1).astype(np.float32)
        scores, indices = idx.search(q, max_k)
        faiss_results.append([
            SearchResult(doc_id=int(doc_id_arr[i]), score=float(sc))
            for sc, i in zip(scores[0], indices[0]) if i != -1
        ])

    results["FAISS-Flat"] = {
        k: round(compute_recall_from_results(
            [r[:k] for r in faiss_results], queries_df, qrels_df, k
        ), 4)
        for k in k_values
    }

    # Qdrant
    print("  Running Qdrant...")
    qdrant = QdrantEngine()
    qdrant.index(doc_ids, embeddings)
    qdrant_results = []
    for qe in query_embs:
        qdrant_results.append(qdrant.search(qe, top_k=max_k))
    qdrant.cleanup()

    results["Qdrant"] = {
        k: round(compute_recall_from_results(
            [r[:k] for r in qdrant_results], queries_df, qrels_df, k
        ), 4)
        for k in k_values
    }

    # Elasticsearch
    print("  Running Elasticsearch...")
    es = ElasticsearchEngine()
    es.index(doc_ids, embeddings, texts=None)
    es_results = []
    for qe in query_embs:
        es_results.append(es.search(qe, top_k=max_k))
    es.cleanup()

    results["ES-Vector"] = {
        k: round(compute_recall_from_results(
            [r[:k] for r in es_results], queries_df, qrels_df, k
        ), 4)
        for k in k_values
    }

    # Print
    header = f"  {'Engine':<16}" + "".join(f"  R@{k:<4}" for k in k_values)
    print(f"\n{header}")
    print(f"  {'─'*14}" + "  ─────" * len(k_values))
    for eng, recalls in results.items():
        vals = "".join(f"  {recalls[k]:.4f}" for k in k_values) if recalls else ""
        print(f"  {eng:<16}{vals}")

    return results


# ═══════════════════════════════════════════════════════════
#  Experiment 4 — Qdrant Disk Usage
# ═══════════════════════════════════════════════════════════

def measure_qdrant_disk(doc_ids, embeddings):
    """Measure Qdrant's actual disk footprint after indexing."""
    print(f"\n{'='*60}")
    print(f"  🔬 Experiment 4 — Qdrant Disk Usage")
    print(f"{'='*60}\n")

    qdrant = QdrantEngine()
    qdrant.index(doc_ids, embeddings)

    disk_mb = 0
    storage_path = Path(config.QDRANT_STORAGE_PATH)
    if storage_path.exists():
        total = sum(f.stat().st_size for f in storage_path.rglob("*") if f.is_file())
        disk_mb = total / (1024 * 1024)

    process_rss = psutil.Process().memory_info().rss / (1024 * 1024)
    n = len(doc_ids)
    raw_vec_mb = (n * embeddings.shape[1] * 4) / (1024 * 1024)

    result = {
        "num_vectors": n,
        "dim": embeddings.shape[1],
        "raw_vectors_mb": round(raw_vec_mb, 1),
        "disk_usage_mb": round(disk_mb, 1),
        "process_rss_mb": round(process_rss, 1),
        "overhead_ratio": round(disk_mb / raw_vec_mb, 2) if raw_vec_mb > 0 else 0,
    }
    print(f"  Vectors:     {n:,} × {embeddings.shape[1]}d")
    print(f"  Raw size:    {raw_vec_mb:.1f} MB")
    print(f"  Disk usage:  {disk_mb:.1f} MB (HNSW index + payloads)")
    print(f"  Overhead:    {result['overhead_ratio']:.2f}× raw size")
    print(f"  Process RSS: {process_rss:.1f} MB")

    qdrant.cleanup()
    return result


# ═══════════════════════════════════════════════════════════
#  Experiment 5 — Concurrency Test
# ═══════════════════════════════════════════════════════════

def run_concurrency_test(doc_ids, embeddings, query_embs, worker_counts=(1, 4, 8)):
    """Measure throughput / latency under concurrent query load."""
    print(f"\n{'='*60}")
    print(f"  🔬 Experiment 5 — Concurrency Test (FAISS)")
    print(f"{'='*60}\n")

    n, dim = embeddings.shape
    vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
    doc_id_arr = np.array(doc_ids, dtype=np.int64)

    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)

    q_batch = query_embs[:100]
    results = []

    for workers in worker_counts:
        def search_one(qe):
            s = time.time()
            q = qe.reshape(1, -1).astype(np.float32)
            idx.search(q, 10)
            return (time.time() - s) * 1000

        wall_start = time.time()
        latencies = []

        if workers == 1:
            for qe in q_batch:
                latencies.append(search_one(qe))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(search_one, qe) for qe in q_batch]
                for f in as_completed(futures):
                    latencies.append(f.result())

        wall_time = time.time() - wall_start
        throughput = len(q_batch) / wall_time
        stats = compute_latency_stats(latencies)

        row = {
            "workers": workers,
            "throughput_qps": round(throughput, 1),
            "latency_p50_ms": round(stats["p50"], 2),
            "latency_p95_ms": round(stats["p95"], 2),
            "wall_time_sec": round(wall_time, 2),
        }
        results.append(row)
        print(f"  {workers} worker(s)  →  {throughput:.0f} QPS  "
              f"P50={stats['p50']:.2f}ms  P95={stats['p95']:.2f}ms  "
              f"wall={wall_time:.2f}s")

    return results


# ═══════════════════════════════════════════════════════════
#  Experiment 6 — Qdrant m=16 vs m=32 ef sweep
# ═══════════════════════════════════════════════════════════

def qdrant_m_comparison_100k():
    """Run ef sweep at 100k with m=16 and m=32 to show recall ceiling shift."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        HnswConfigDiff, OptimizersConfigDiff, SearchParams,
    )

    scale = 100_000
    print(f"\n{'='*60}")
    print(f"  🔬 Qdrant m=16 vs m=32 at {scale:,}")
    print(f"{'='*60}\n")

    passages_df, queries_df, qrels_df = load_data(scale)
    passage_embs = np.load(f"data/embeddings/passages_{scale}.npy")
    query_embs = np.load(f"data/embeddings/queries_{scale}.npy")
    doc_ids = passages_df["passage_id"].tolist()
    n, dim = passage_embs.shape

    all_results = {}

    for m_val in [16, 32]:
        storage = f"data/qdrant_m{m_val}_sweep"
        if Path(storage).exists():
            shutil.rmtree(storage)

        client = QdrantClient(path=storage)
        client.create_collection(
            collection_name="m_sweep",
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=m_val, ef_construct=200),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
        )

        print(f"  Indexing {n:,} vectors (m={m_val})...")
        batch_size = 1000
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            points = [
                PointStruct(id=j, vector=passage_embs[j].tolist(),
                            payload={"doc_id": int(doc_ids[j])})
                for j in range(i, end)
            ]
            client.upsert(collection_name="m_sweep", points=points)
            if (i // batch_size) % 25 == 0:
                print(f"     ... {min(end, n):,}/{n:,}")
        print(f"  ✅ Indexed (m={m_val})\n")

        ef_values = [16, 32, 64, 128, 256]
        sweep_results = []

        for ef in ef_values:
            latencies = []
            result_lists = []
            for qe in query_embs:
                start = time.time()
                hits = client.query_points(
                    collection_name="m_sweep",
                    query=qe.tolist(),
                    limit=10,
                    search_params=SearchParams(hnsw_ef=ef),
                )
                latencies.append((time.time() - start) * 1000)
                result_lists.append([
                    SearchResult(doc_id=int(h.payload["doc_id"]), score=float(h.score))
                    for h in hits.points
                ])

            stats = compute_latency_stats(latencies)
            recall = compute_recall_from_results(result_lists, queries_df, qrels_df, 10)
            row = {
                "m": m_val, "ef_search": ef,
                "recall_at_10": round(recall, 4),
                "latency_p50_ms": round(stats["p50"], 2),
                "latency_p95_ms": round(stats["p95"], 2),
            }
            sweep_results.append(row)
            print(f"  m={m_val} ef={ef:>4}  →  Recall@10={recall:.4f}  "
                  f"P50={stats['p50']:.2f}ms")

        all_results[f"m{m_val}"] = sweep_results
        client.delete_collection("m_sweep")
        client.close()
        shutil.rmtree(storage, ignore_errors=True)
        print()

    return all_results


# ═══════════════════════════════════════════════════════════
#  Experiment 7 — Hybrid confusion matrix
# ═══════════════════════════════════════════════════════════

def hybrid_confusion_matrix():
    """Count all 4 outcome categories: both find, vector only, BM25 only, both miss."""
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import streaming_bulk

    scale = 10_000
    print(f"\n{'='*60}")
    print(f"  🔬 Hybrid Confusion Matrix — {scale:,}")
    print(f"{'='*60}\n")

    passages_df, queries_df, qrels_df = load_data(scale)
    passage_embs = np.load(f"data/embeddings/passages_{scale}.npy")
    query_embs = np.load(f"data/embeddings/queries_{scale}.npy")
    doc_ids = passages_df["passage_id"].tolist()
    texts = passages_df["passage_text"].tolist()
    query_texts = queries_df["query_text"].tolist()

    n, dim = passage_embs.shape
    vecs = np.ascontiguousarray(passage_embs, dtype=np.float32)
    doc_id_arr = np.array(doc_ids, dtype=np.int64)

    faiss_idx = faiss.IndexFlatIP(dim)
    faiss_idx.add(vecs)

    es = Elasticsearch(config.ES_HOST, request_timeout=60)
    idx_name = "confusion_test"
    if es.indices.exists(index=idx_name):
        es.indices.delete(index=idx_name)

    mapping = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": {
            "doc_id": {"type": "integer"},
            "text": {"type": "text", "analyzer": "standard"},
        }},
    }
    es.indices.create(index=idx_name, body=mapping)

    def actions():
        for i in range(n):
            yield {"_index": idx_name, "_id": i,
                   "_source": {"doc_id": int(doc_ids[i]), "text": texts[i]}}

    for ok, _ in streaming_bulk(es, actions(), chunk_size=500, raise_on_error=True):
        pass
    es.indices.refresh(index=idx_name)

    both_hit = 0
    vec_only = 0
    bm25_only = 0
    both_miss = 0
    total_with_rel = 0

    n_q = min(len(query_embs), len(queries_df))
    for i in range(n_q):
        qid = queries_df.iloc[i]["query_id"]
        relevant = set(qrels_df[qrels_df["query_id"] == qid]["passage_id"])
        if not relevant:
            continue
        total_with_rel += 1

        # Vector
        q = query_embs[i].reshape(1, -1).astype(np.float32)
        _, indices = faiss_idx.search(q, 10)
        vec_ids = {int(doc_id_arr[j]) for j in indices[0] if j != -1}
        v_hit = len(vec_ids & relevant) > 0

        # BM25
        body = {"size": 10, "query": {"match": {"text": query_texts[i]}}}
        resp = es.search(index=idx_name, body=body)
        bm25_ids = {int(h["_source"]["doc_id"]) for h in resp["hits"]["hits"]}
        b_hit = len(bm25_ids & relevant) > 0

        if v_hit and b_hit:
            both_hit += 1
        elif v_hit and not b_hit:
            vec_only += 1
        elif not v_hit and b_hit:
            bm25_only += 1
        else:
            both_miss += 1

    es.indices.delete(index=idx_name)
    es.close()

    result = {
        "total_queries_with_relevance": total_with_rel,
        "both_find": both_hit,
        "vector_only": vec_only,
        "bm25_only": bm25_only,
        "both_miss": both_miss,
    }

    print(f"  Queries with relevance judgments: {total_with_rel}\n")
    print(f"  ┌─────────────────┬────────────┬──────────┐")
    print(f"  │                 │ BM25 ✅    │ BM25 ❌  │")
    print(f"  ├─────────────────┼────────────┼──────────┤")
    print(f"  │ Vector ✅       │  {both_hit:>6}     │  {vec_only:>5}   │")
    print(f"  │ Vector ❌       │  {bm25_only:>6}     │  {both_miss:>5}   │")
    print(f"  └─────────────────┴────────────┴──────────┘")
    print(f"\n  Hybrid advantage: {vec_only + bm25_only} queries where one modality rescues the other")

    return result


# ═══════════════════════════════════════════════════════════
#  Experiment 8 — FAISS HNSW param logging
# ═══════════════════════════════════════════════════════════

def faiss_hnsw_params():
    """Log FAISS HNSW params explicitly for reviewer credibility."""
    print(f"\n{'='*60}")
    print(f"  🔬 FAISS HNSW Parameter Log")
    print(f"{'='*60}\n")

    params = [
        {"variant": "FAISS-HNSW-16", "m": 16, "efConstruction": 200, "efSearch": 64,
         "metric": "METRIC_INNER_PRODUCT"},
        {"variant": "FAISS-HNSW-32", "m": 32, "efConstruction": 200, "efSearch": 64,
         "metric": "METRIC_INNER_PRODUCT"},
    ]
    for p in params:
        print(f"  {p['variant']}:")
        print(f"    m={p['m']}, efConstruction={p['efConstruction']}, "
              f"efSearch={p['efSearch']}, metric={p['metric']}")
    return params


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def run_all_experiments(scale: int = 10_000):
    """Run all advanced experiments."""
    print(f"\n{'═'*60}")
    print(f"  🔬 Advanced Benchmark Experiments — {scale:,} passages")
    print(f"{'═'*60}")

    passages_df, queries_df, qrels_df = load_data(scale)
    passage_embs = np.load(f"data/embeddings/passages_{scale}.npy")
    query_embs = np.load(f"data/embeddings/queries_{scale}.npy")
    doc_ids = passages_df["passage_id"].tolist()

    all_results = {}

    # 1. Qdrant ef sweep (10k scale limit for quick run)
    all_results["qdrant_ef_sweep"] = run_qdrant_ef_sweep(
        doc_ids, passage_embs, query_embs, qrels_df, queries_df)

    # 2. FAISS HNSW
    all_results["faiss_hnsw"] = run_faiss_hnsw_comparison(
        doc_ids, passage_embs, query_embs, qrels_df, queries_df)

    # 3. Recall@k curves
    all_results["recall_at_k"] = run_recall_at_k_curves(
        doc_ids, passage_embs, query_embs, qrels_df, queries_df)

    # 4. Qdrant disk
    all_results["qdrant_disk"] = measure_qdrant_disk(doc_ids, passage_embs)

    # 5. Concurrency
    all_results["concurrency"] = run_concurrency_test(
        doc_ids, passage_embs, query_embs)

    # 6. Qdrant m=16 vs m=32 ef sweep
    all_results["qdrant_m_comparison_100k"] = qdrant_m_comparison_100k()

    # 7. Hybrid confusion matrix
    all_results["hybrid_confusion"] = hybrid_confusion_matrix()

    # 8. FAISS HNSW param logging
    all_results["faiss_hnsw_params"] = faiss_hnsw_params()

    # Save
    out_path = Path(config.RESULTS_DIR) / "advanced_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 All results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Benchmark Experiments")
    parser.add_argument("--scale", type=int, default=10_000)
    args = parser.parse_args()
    run_all_experiments(args.scale)
