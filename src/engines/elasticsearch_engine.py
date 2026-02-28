"""
Elasticsearch Search Engine — Vector + Hybrid search (ES 7.x compatible).

Implements the SearchEngine interface for benchmarking.
Requires a running Elasticsearch 7.x instance.

ES 7.x approach:
  - Vector search: script_score query with dotProduct function
  - Hybrid search: bool query combining match (BM25) + script_score (vector)
  - dense_vector field (non-indexed, brute-force — ES 7.x limitation)

Note: ES 7.x does NOT support native knn/HNSW. All vector search is brute-force
      via script_score. This is a known limitation and should be noted in the blog.
"""

import sys
import time
from pathlib import Path

import numpy as np
import psutil
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.engines.base import SearchEngine, IndexStats, SearchResult


class ElasticsearchEngine(SearchEngine):
    """Elasticsearch vector search engine with hybrid search support."""

    def __init__(self, hybrid: bool = False):
        """
        Args:
            hybrid: if True, use BM25 + vector combined scoring
        """
        self.hybrid = hybrid
        self.es = None
        self.index_name = config.ES_INDEX

    def name(self) -> str:
        mode = "Hybrid" if self.hybrid else "Vector"
        return f"ES-{mode}"

    def _connect(self):
        """Connect to Elasticsearch."""
        self.es = Elasticsearch(config.ES_HOST, request_timeout=60)
        info = self.es.info()
        version = info["version"]["number"]
        print(f"  🔗 [{self.name()}] Connected to ES {version}")

    def index(
        self,
        doc_ids: list[int],
        embeddings: np.ndarray,
        texts: list[str] | None = None,
        metadata: dict[str, list] | None = None,
    ) -> IndexStats:
        """Create index with dense_vector + text fields and bulk index."""
        n, dim = embeddings.shape
        self._metadata = metadata or {}

        self._connect()

        # Delete existing index
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)

        # Create index mapping (ES 7.x style)
        print(f"  📦 [{self.name()}] Creating index (dim={dim}, ES 7.x script_score mode)...")

        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "-1",
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "integer"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dim,
                    },
                    "text": {
                        "type": "text",
                        "analyzer": "standard",
                    },
                },
            },
        }
        # Add metadata fields as keyword type
        for field in self._metadata:
            mapping["mappings"]["properties"][field] = {"type": "keyword"}
        self.es.indices.create(index=self.index_name, body=mapping)

        # Bulk index
        print(f"  📤 [{self.name()}] Bulk indexing {n:,} documents...")
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        start = time.time()

        def generate_actions():
            for i in range(n):
                doc = {
                    "_index": self.index_name,
                    "_id": i,
                    "_source": {
                        "doc_id": int(doc_ids[i]),
                        "embedding": embeddings[i].tolist(),
                    },
                }
                if texts:
                    doc["_source"]["text"] = texts[i]
                for field, values in self._metadata.items():
                    doc["_source"][field] = values[i]
                yield doc

        indexed = 0
        for ok, info in streaming_bulk(
            self.es,
            generate_actions(),
            chunk_size=config.ES_BULK_SIZE,
            raise_on_error=True,
        ):
            indexed += 1
            if indexed % (config.ES_BULK_SIZE * 5) == 0:
                progress = indexed / n * 100
                print(f"     ... {progress:.0f}% ({indexed:,}/{n:,})")

        # Refresh to make docs searchable
        self.es.indices.refresh(index=self.index_name)

        elapsed = time.time() - start
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        mem_used = max(0, mem_after - mem_before)

        count = self.es.count(index=self.index_name)["count"]
        print(f"  ✅ [{self.name()}] Indexed {count:,} docs in {elapsed:.2f}s, memory: ~{mem_used:.1f} MB")

        return IndexStats(
            num_documents=count,
            index_time_sec=elapsed,
            memory_usage_mb=mem_used,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        query_text: str = None,
    ) -> list[SearchResult]:
        """
        Search using script_score (vector) or hybrid (BM25 + vector).

        ES 7.x uses script_score with dotProduct for vector similarity.
        Vectors are pre-normalized, so dotProduct ≈ cosine similarity.
        """
        if self.es is None:
            raise RuntimeError("Not connected. Call index() first.")

        query_vec = query_embedding.tolist()

        # script_score for vector similarity
        # dotProduct returns raw dot product; +1 to keep scores positive
        vector_script = {
            "source": "dotProduct(params.query_vector, 'embedding') + 1.0",
            "params": {"query_vector": query_vec},
        }

        if self.hybrid and query_text:
            # Hybrid: combine BM25 text match with vector score
            body = {
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": {
                            "bool": {
                                "should": [
                                    {"match": {"text": query_text}},
                                    {"match_all": {}},
                                ],
                            },
                        },
                        "script": vector_script,
                    },
                },
            }
        else:
            # Pure vector search via script_score
            body = {
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": vector_script,
                    },
                },
            }

        response = self.es.search(index=self.index_name, body=body)

        results = []
        for hit in response["hits"]["hits"]:
            results.append(SearchResult(
                doc_id=int(hit["_source"]["doc_id"]),
                score=float(hit["_score"]),
            ))

        return results

    def cleanup(self) -> None:
        """Delete index and close connection."""
        if self.es:
            try:
                if self.es.indices.exists(index=self.index_name):
                    self.es.indices.delete(index=self.index_name)
            except Exception:
                pass
            self.es.close()
            self.es = None
        self._metadata = {}
        print(f"  🧹 [{self.name()}] Cleaned up")

    def search_with_filter(
        self,
        query_embedding: np.ndarray,
        filter_field: str,
        filter_value: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Vector search with bool filter on a keyword field."""
        body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": [{"term": {filter_field: filter_value}}],
                        },
                    },
                    "script": {
                        "source": "dotProduct(params.qv, 'embedding') + 1.0",
                        "params": {"qv": query_embedding.tolist()},
                    },
                },
            },
        }
        response = self.es.search(index=self.index_name, body=body)
        return [
            SearchResult(doc_id=int(h["_source"]["doc_id"]), score=float(h["_score"]))
            for h in response["hits"]["hits"]
        ]

    def search_fulltext(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Pure BM25 full-text search."""
        body = {"size": top_k, "query": {"match": {"text": query_text}}}
        response = self.es.search(index=self.index_name, body=body)
        return [
            SearchResult(doc_id=int(h["_source"]["doc_id"]), score=float(h["_score"]))
            for h in response["hits"]["hits"]
        ]

    def search_hybrid(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Hybrid: BM25 + vector scoring."""
        body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "should": [
                                {"match": {"text": query_text}},
                                {"match_all": {}},
                            ],
                        },
                    },
                    "script": {
                        "source": "dotProduct(params.qv, 'embedding') + 1.0",
                        "params": {"qv": query_embedding.tolist()},
                    },
                },
            },
        }
        response = self.es.search(index=self.index_name, body=body)
        return [
            SearchResult(doc_id=int(h["_source"]["doc_id"]), score=float(h["_score"]))
            for h in response["hits"]["hits"]
        ]


# ─── CLI Test ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Elasticsearch Engine Test")
    parser.add_argument("--scale", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hybrid", action="store_true", help="Enable hybrid BM25+vector search")
    args = parser.parse_args()

    # Load data
    passages_df = pd.read_parquet(f"data/processed/passages_{args.scale}.parquet")
    queries_df = pd.read_parquet(f"data/processed/queries_{args.scale}.parquet")
    passage_embs = np.load(f"data/embeddings/passages_{args.scale}.npy")
    query_embs = np.load(f"data/embeddings/queries_{args.scale}.npy")

    mode = "HYBRID" if args.hybrid else "VECTOR"
    print(f"\n{'='*60}")
    print(f"  Elasticsearch Engine Test — {mode} — {args.scale:,} passages")
    print(f"{'='*60}\n")

    # Build index (always include texts)
    engine = ElasticsearchEngine(hybrid=args.hybrid)
    stats = engine.index(
        doc_ids=passages_df["passage_id"].tolist(),
        embeddings=passage_embs,
        texts=passages_df["passage_text"].tolist(),
    )

    # Run queries
    print(f"\n  🔍 Running {len(queries_df)} queries (top-{args.top_k}, mode={mode})...\n")
    latencies = []
    all_results = []

    for i, row in queries_df.iterrows():
        start = time.time()
        results = engine.search(
            query_embs[i],
            top_k=args.top_k,
            query_text=row["query_text"] if args.hybrid else None,
        )
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)
        all_results.append(results)

    latencies = np.array(latencies)

    print(f"  📊 Latency Stats ({len(latencies)} queries):")
    print(f"     P50:  {np.percentile(latencies, 50):.2f} ms")
    print(f"     P95:  {np.percentile(latencies, 95):.2f} ms")
    print(f"     Mean: {latencies.mean():.2f} ms")
    print(f"     Std:  {latencies.std():.2f} ms")

    # Sample result
    sample_query = queries_df.iloc[0]["query_text"]
    sample_results = all_results[0]
    print(f"\n  🔍 Sample query: \"{sample_query}\"")
    print(f"  📋 Top-{args.top_k} results:")
    for j, r in enumerate(sample_results[:5]):
        passage = passages_df[passages_df["passage_id"] == r.doc_id]["passage_text"].values
        text_preview = passage[0][:100] + "..." if len(passage) > 0 else "?"
        print(f"     {j+1}. [score={r.score:.4f}] {text_preview}")

    # Quick Recall
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
