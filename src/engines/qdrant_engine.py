"""
Qdrant Search Engine — Local persistent mode (no Docker).

Implements the SearchEngine interface for benchmarking.
Uses Qdrant's local disk storage for Mac-friendly operation.

Features:
  - HNSW indexing with configurable m, ef_construct, ef_search
  - Batch upsert with progress
  - Cosine similarity (with pre-normalized vectors = inner product)
"""

import sys
import time
import shutil
from pathlib import Path

import numpy as np
import psutil
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    OptimizersConfigDiff,
    PointStruct,
    SearchParams,
    VectorParams,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.engines.base import SearchEngine, IndexStats, SearchResult


class QdrantEngine(SearchEngine):
    """Qdrant vector search engine using local persistent storage."""

    def __init__(self):
        self.storage_path = config.QDRANT_STORAGE_PATH
        self.collection_name = config.QDRANT_COLLECTION
        self.client = None
        self.doc_ids = None  # maps point index → real doc_id

    def name(self) -> str:
        return "Qdrant"

    def _init_client(self):
        """Initialize local Qdrant client."""
        self.client = QdrantClient(path=self.storage_path)
        print(f"  🔗 [Qdrant] Connected (local mode: {self.storage_path})")

    def index(
        self,
        doc_ids: list[int],
        embeddings: np.ndarray,
        texts: list[str] | None = None,
        metadata: dict[str, list] | None = None,
    ) -> IndexStats:
        """Create collection and upsert embeddings."""
        n, dim = embeddings.shape
        self.doc_ids = doc_ids
        self._metadata = metadata or {}

        # Initialize client
        self._init_client()

        # Delete existing collection if present
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            self.client.delete_collection(self.collection_name)

        # Create collection with HNSW config
        print(f"  📦 [Qdrant] Creating collection (dim={dim}, "
              f"m={config.QDRANT_HNSW_M}, ef_construct={config.QDRANT_HNSW_EF_CONSTRUCT})...")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(
                m=config.QDRANT_HNSW_M,
                ef_construct=config.QDRANT_HNSW_EF_CONSTRUCT,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=0,  # build index immediately
            ),
        )

        # Batch upsert
        print(f"  📤 [Qdrant] Upserting {n:,} points...")
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        start = time.time()

        batch_size = 500
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            points = []
            for idx in range(i, end):
                payload = {"doc_id": int(doc_ids[idx])}
                for field, values in self._metadata.items():
                    payload[field] = values[idx]
                points.append(PointStruct(
                    id=idx,
                    vector=embeddings[idx].tolist(),
                    payload=payload,
                ))
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            if (i // batch_size) % 5 == 0:
                progress = min(end, n) / n * 100
                print(f"     ... {progress:.0f}% ({min(end, n):,}/{n:,})")

        elapsed = time.time() - start
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        mem_used = max(0, mem_after - mem_before)

        # Wait for indexing to complete
        collection_info = self.client.get_collection(self.collection_name)
        print(f"  ✅ [Qdrant] Indexed in {elapsed:.2f}s, memory: ~{mem_used:.1f} MB")
        print(f"     Points: {collection_info.points_count:,}, "
              f"Status: {collection_info.status}")

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
        """Search using HNSW with configurable ef_search."""
        if self.client is None:
            raise RuntimeError("Client not initialized. Call index() first.")

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            search_params=SearchParams(
                hnsw_ef=config.QDRANT_HNSW_EF_SEARCH,
            ),
        )

        return [
            SearchResult(
                doc_id=int(hit.payload["doc_id"]),
                score=float(hit.score),
            )
            for hit in results.points
        ]

    def search_with_filter(
        self,
        query_embedding: np.ndarray,
        filter_field: str,
        filter_value: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Native HNSW search with payload filter."""
        if self.client is None:
            raise RuntimeError("Client not initialized. Call index() first.")

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            query_filter=Filter(
                must=[FieldCondition(key=filter_field, match=MatchValue(value=filter_value))]
            ),
            search_params=SearchParams(hnsw_ef=config.QDRANT_HNSW_EF_SEARCH),
        )

        return [
            SearchResult(
                doc_id=int(hit.payload["doc_id"]),
                score=float(hit.score),
            )
            for hit in results.points
        ]

    def cleanup(self) -> None:
        """Delete collection and close client."""
        if self.client:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.client.close()
            self.client = None

        # Clean up storage directory
        storage = Path(self.storage_path)
        if storage.exists():
            shutil.rmtree(storage)

        self.doc_ids = None
        print(f"  🧹 [Qdrant] Cleaned up")


# ─── CLI Test ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Qdrant Engine Test")
    parser.add_argument("--scale", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    # Load data
    passages_df = pd.read_parquet(f"data/processed/passages_{args.scale}.parquet")
    queries_df = pd.read_parquet(f"data/processed/queries_{args.scale}.parquet")
    passage_embs = np.load(f"data/embeddings/passages_{args.scale}.npy")
    query_embs = np.load(f"data/embeddings/queries_{args.scale}.npy")

    print(f"\n{'='*60}")
    print(f"  Qdrant Engine Test — {args.scale:,} passages")
    print(f"{'='*60}\n")

    # Build index
    engine = QdrantEngine()
    stats = engine.index(
        doc_ids=passages_df["passage_id"].tolist(),
        embeddings=passage_embs,
    )

    # Run queries with latency measurement
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

    # Quick Recall check
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
