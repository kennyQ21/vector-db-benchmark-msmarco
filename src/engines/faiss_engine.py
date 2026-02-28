"""
FAISS Search Engine — Flat (exact) and IVF (approximate) modes.

Implements the SearchEngine interface for benchmarking.
FAISS runs in-process (no external service needed).

Index types:
  - Flat (IndexFlatIP): Exact inner-product search — accuracy upper bound
  - IVF  (IndexIVFFlat): Approximate search — fair comparison with HNSW engines
"""

import sys
import time
from pathlib import Path

import numpy as np
import faiss
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.engines.base import SearchEngine, IndexStats, SearchResult


class FaissEngine(SearchEngine):
    """FAISS vector search engine with Flat and IVF index support."""

    def __init__(self, index_type: str = "flat"):
        """
        Args:
            index_type: "flat" (exact) or "ivf" (approximate)
        """
        if index_type not in ("flat", "ivf"):
            raise ValueError(f"Unknown index type: {index_type}. Use 'flat' or 'ivf'.")

        self.index_type = index_type
        self._faiss_index = None
        self.doc_ids = None  # maps FAISS internal idx → real doc_id
        self._metadata = {}  # field_name → np.array

    def name(self) -> str:
        return f"FAISS-{self.index_type.upper()}"

    def index(
        self,
        doc_ids: list[int],
        embeddings: np.ndarray,
        texts: list[str] | None = None,
        metadata: dict[str, list] | None = None,
    ) -> IndexStats:
        """Build FAISS index from embeddings."""
        n, dim = embeddings.shape
        if metadata:
            self._metadata = {k: np.array(v) for k, v in metadata.items()}
        self.doc_ids = np.array(doc_ids, dtype=np.int64)

        print(f"  📦 [{self.name()}] Indexing {n:,} vectors (dim={dim})...")
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        start = time.time()

        # Ensure contiguous float32 (FAISS requirement — single copy)
        vecs = np.ascontiguousarray(embeddings, dtype=np.float32)

        if self.index_type == "flat":
            # Exact inner-product search (cosine with normalized vectors)
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(vecs)

        elif self.index_type == "ivf":
            # Approximate search with inverted file index
            quantizer = faiss.IndexFlatIP(dim)
            nlist = min(config.FAISS_IVF_NLIST, n // 10)  # ensure enough data per cluster
            self._faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self._faiss_index.train(vecs)
            self._faiss_index.add(vecs)
            self._faiss_index.nprobe = config.FAISS_IVF_NPROBE

        elapsed = time.time() - start
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        mem_used = max(0, mem_after - mem_before)

        print(f"  ✅ [{self.name()}] Indexed in {elapsed:.2f}s, memory: ~{mem_used:.1f} MB")

        return IndexStats(
            num_documents=n,
            index_time_sec=elapsed,
            memory_usage_mb=mem_used,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search for nearest neighbors using inner product."""
        if self._faiss_index is None:
            raise RuntimeError("Index not built. Call index() first.")

        # FAISS expects (1, dim) for single query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._faiss_index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS returns -1 for missing results
            real_id = int(self.doc_ids[idx])
            results.append(SearchResult(doc_id=real_id, score=float(score)))

        return results

    def cleanup(self) -> None:
        """Free the FAISS index."""
        self._faiss_index = None
        self.doc_ids = None
        self._metadata = {}
        print(f"  🧹 [{self.name()}] Cleaned up")

    OVERFETCH_FACTOR = 5

    def search_with_filter(
        self,
        query_embedding: np.ndarray,
        filter_field: str,
        filter_value: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Post-filter: over-fetch then filter by metadata field."""
        if filter_field not in self._metadata:
            raise ValueError(f"Unknown metadata field: {filter_field}")

        overfetch = top_k * self.OVERFETCH_FACTOR
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._faiss_index.search(query, overfetch)

        field_arr = self._metadata[filter_field]
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if field_arr[idx] == filter_value:
                results.append(SearchResult(
                    doc_id=int(self.doc_ids[idx]), score=float(score),
                ))
            if len(results) >= top_k:
                break
        return results


# ─── CLI Test ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="FAISS Engine Test")
    parser.add_argument("--scale", type=int, default=10_000)
    parser.add_argument("--index-type", choices=["flat", "ivf"], default="flat")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    # Load data
    passages_df = pd.read_parquet(f"data/processed/passages_{args.scale}.parquet")
    queries_df = pd.read_parquet(f"data/processed/queries_{args.scale}.parquet")
    passage_embs = np.load(f"data/embeddings/passages_{args.scale}.npy")
    query_embs = np.load(f"data/embeddings/queries_{args.scale}.npy")

    print(f"\n{'='*60}")
    print(f"  FAISS Engine Test — {args.index_type.upper()} — {args.scale:,} passages")
    print(f"{'='*60}\n")

    # Build index
    engine = FaissEngine(index_type=args.index_type)
    stats = engine.index(
        doc_ids=passages_df["passage_id"].tolist(),
        embeddings=passage_embs,
    )

    # Run sample queries with latency measurement
    print(f"\n  🔍 Running {len(queries_df)} queries (top-{args.top_k})...\n")
    latencies = []
    all_results = []

    for i, row in queries_df.iterrows():
        start = time.time()
        results = engine.search(query_embs[i], top_k=args.top_k)
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)
        all_results.append(results)

    latencies = np.array(latencies)

    # Print stats
    print(f"  📊 Latency Stats ({len(latencies)} queries):")
    print(f"     P50:  {np.percentile(latencies, 50):.2f} ms")
    print(f"     P95:  {np.percentile(latencies, 95):.2f} ms")
    print(f"     Mean: {latencies.mean():.2f} ms")
    print(f"     Std:  {latencies.std():.2f} ms")

    # Show sample result
    sample_query = queries_df.iloc[0]["query_text"]
    sample_results = all_results[0]
    print(f"\n  🔍 Sample query: \"{sample_query}\"")
    print(f"  📋 Top-{args.top_k} results:")
    for j, r in enumerate(sample_results[:5]):
        passage = passages_df[passages_df["passage_id"] == r.doc_id]["passage_text"].values
        text_preview = passage[0][:100] + "..." if len(passage) > 0 else "?"
        print(f"     {j+1}. [score={r.score:.4f}] {text_preview}")

    # Quick Recall check against qrels
    qrels_df = pd.read_parquet(f"data/processed/qrels_{args.scale}.parquet")
    hits = 0
    total = 0
    for i, row in queries_df.iterrows():
        qid = row["query_id"]
        relevant = set(qrels_df[qrels_df["query_id"] == qid]["passage_id"])
        if not relevant:
            continue
        retrieved = {r.doc_id for r in all_results[i]}
        hits += len(retrieved & relevant)
        total += len(relevant)

    recall = hits / total if total > 0 else 0
    print(f"\n  🎯 Quick Recall@{args.top_k}: {recall:.4f} ({hits}/{total})")

    engine.cleanup()
