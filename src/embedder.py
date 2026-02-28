"""
Embedding Generator — Passage & Query Encoding

Uses sentence-transformers to encode passages and queries with:
  - BAAI/bge-small-en-v1.5 (384-dim, modern, strong retrieval)
  - Query prefix for BGE instruction format
  - L2 normalization for fair cosine similarity across engines
  - Caching as .npy files per scale
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class Embedder:
    """Encode passages and queries using a sentence-transformer model."""

    def __init__(self, device: str = "cpu"):
        # MPS available but causes system slowdown on Mac — use CPU
        # Embeddings are cached as .npy, so this only runs once per scale
        self.device = device

        print(f"🔄 Loading model: {config.EMBEDDING_MODEL} (device={self.device})...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL, device=self.device)
        print(f"   ✅ Model loaded (dim={config.EMBEDDING_DIM}, device={self.device})")

    def encode_passages(
        self,
        texts: list[str],
        scale: int,
        force: bool = False,
    ) -> np.ndarray:
        """
        Encode passage texts → (N, dim) float32 array.

        BGE does NOT require a prefix for passages — only queries.

        Args:
            texts:  list of passage strings
            scale:  dataset scale (for cache filename)
            force:  if True, re-encode even if cache exists

        Returns:
            (N, dim) float32 numpy array, L2-normalized if configured
        """
        cache_path = self._cache_path("passages", scale)

        if cache_path.exists() and not force:
            print(f"📦 Loading cached passage embeddings ({scale:,})...")
            embeddings = np.load(cache_path)
            print(f"   ✅ Loaded shape={embeddings.shape}")
            return embeddings

        print(f"🔢 Encoding {len(texts):,} passages...")
        start = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
        )

        elapsed = time.time() - start
        throughput = len(texts) / elapsed

        print(f"   ✅ Encoded in {elapsed:.1f}s ({throughput:.0f} passages/sec)")
        print(f"   Shape: {embeddings.shape}, dtype: {embeddings.dtype}")

        # Save cache
        self._save_cache(embeddings, cache_path)
        return embeddings

    def encode_queries(
        self,
        texts: list[str],
        scale: int,
        force: bool = False,
    ) -> np.ndarray:
        """
        Encode query texts → (N, dim) float32 array.

        BGE requires an instruction prefix for queries:
        "Represent this sentence for searching relevant passages: {query}"

        Args:
            texts:  list of query strings
            scale:  dataset scale (for cache filename)
            force:  if True, re-encode even if cache exists

        Returns:
            (N, dim) float32 numpy array, L2-normalized if configured
        """
        cache_path = self._cache_path("queries", scale)

        if cache_path.exists() and not force:
            print(f"📦 Loading cached query embeddings ({scale:,})...")
            embeddings = np.load(cache_path)
            print(f"   ✅ Loaded shape={embeddings.shape}")
            return embeddings

        # Apply BGE query prefix
        prefixed = [config.QUERY_PREFIX + t for t in texts]
        print(f"🔢 Encoding {len(texts):,} queries (with BGE prefix)...")
        start = time.time()

        embeddings = self.model.encode(
            prefixed,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
        )

        elapsed = time.time() - start
        throughput = len(texts) / elapsed

        print(f"   ✅ Encoded in {elapsed:.1f}s ({throughput:.0f} queries/sec)")
        print(f"   Shape: {embeddings.shape}, dtype: {embeddings.dtype}")

        self._save_cache(embeddings, cache_path)
        return embeddings

    # ─── Internals ────────────────────────────────────────

    def _cache_path(self, prefix: str, scale: int) -> Path:
        p = Path(config.EMBEDDINGS_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{prefix}_{scale}.npy"

    def _save_cache(self, embeddings: np.ndarray, path: Path):
        np.save(path, embeddings)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"   💾 Cached to {path} ({size_mb:.1f} MB)")


# ─── CLI Test ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Embedding Generator")
    parser.add_argument(
        "--scale", type=int, default=10_000,
        help="Dataset scale to encode (default: 10000)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-encode even if cache exists",
    )
    args = parser.parse_args()

    # Load data
    passages_df = pd.read_parquet(
        Path(config.DATA_DIR) / "processed" / f"passages_{args.scale}.parquet"
    )
    queries_df = pd.read_parquet(
        Path(config.DATA_DIR) / "processed" / f"queries_{args.scale}.parquet"
    )

    # Encode
    embedder = Embedder()

    passage_embs = embedder.encode_passages(
        passages_df["passage_text"].tolist(),
        scale=args.scale,
        force=args.force,
    )
    query_embs = embedder.encode_queries(
        queries_df["query_text"].tolist(),
        scale=args.scale,
        force=args.force,
    )

    # Sanity checks
    print("\n" + "=" * 50)
    print("📊 Embedding Summary")
    print("=" * 50)
    print(f"  Passage embeddings: {passage_embs.shape}")
    print(f"  Query embeddings:   {query_embs.shape}")

    # Check normalization
    norms = np.linalg.norm(passage_embs[:100], axis=1)
    print(f"\n  L2 norm check (first 100 passages):")
    print(f"    Mean: {norms.mean():.6f}  (want ≈1.0)")
    print(f"    Std:  {norms.std():.8f}  (want ≈0.0)")

    # Quick similarity sanity: first query vs first 5 passages
    sims = query_embs[0] @ passage_embs[:5].T
    print(f"\n  Similarity: query[0] vs passages[0:5]:")
    for i, s in enumerate(sims):
        print(f"    passage[{i}]: {s:.4f}")
