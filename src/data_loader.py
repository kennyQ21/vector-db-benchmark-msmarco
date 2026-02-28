"""
Data Pipeline — MS MARCO Passage Dataset

Downloads MS MARCO v1.1 from HuggingFace, extracts:
  - passages (unique passage_id → passage_text)
  - queries  (query_id → query_text)
  - qrels    (query_id → set of relevant passage_ids)

Supports configurable subset sizes and caches processed data as Parquet.
"""

import os
import sys
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ─── Helpers ──────────────────────────────────────────────

def _cache_path(name: str, scale: int) -> Path:
    """Return path to a cached Parquet file."""
    p = Path(config.DATA_DIR) / "processed"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}_{scale}.parquet"


def _qrels_cache_path(scale: int) -> Path:
    p = Path(config.DATA_DIR) / "processed"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"qrels_{scale}.parquet"


# ─── Core Loading ─────────────────────────────────────────

def load_raw_dataset(split: str = "train"):
    """
    Load raw MS MARCO v1.1 from HuggingFace.

    Args:
        split: "train", "validation", or "test"

    Returns:
        HuggingFace Dataset object
    """
    print(f"📥 Loading MS MARCO {config.DATASET_VERSION} ({split} split)...")
    ds = load_dataset(config.DATASET_NAME, config.DATASET_VERSION, split=split)
    print(f"   ✅ Loaded {len(ds):,} rows")
    return ds


def extract_passages_and_qrels(ds, scale: int):
    """
    Extract unique passages and query→relevant_passage mappings from the dataset.

    MS MARCO v1.1 structure per row:
        query_id:   int
        query:      str
        passages:   { is_selected: [int], passage_text: [str], url: [str] }

    We assign each unique passage a stable integer ID via hashing.

    Args:
        ds:    HuggingFace Dataset (train or validation split)
        scale: number of unique passages to keep

    Returns:
        passages_df:  DataFrame with columns [passage_id, passage_text]
        queries_df:   DataFrame with columns [query_id, query_text]
        qrels_df:     DataFrame with columns [query_id, passage_id, relevance]
    """
    passages_cache = _cache_path("passages", scale)
    queries_cache = _cache_path("queries", scale)
    qrels_cache = _qrels_cache_path(scale)

    # Check cache
    if passages_cache.exists() and queries_cache.exists() and qrels_cache.exists():
        print(f"📦 Loading cached data for scale={scale:,}...")
        passages_df = pd.read_parquet(passages_cache)
        queries_df = pd.read_parquet(queries_cache)
        qrels_df = pd.read_parquet(qrels_cache)
        print(f"   ✅ {len(passages_df):,} passages, {len(queries_df):,} queries, {len(qrels_df):,} qrels")
        return passages_df, queries_df, qrels_df

    print(f"🔧 Extracting passages & qrels (target scale: {scale:,})...")
    random.seed(config.RANDOM_SEED)

    # Collect all unique passages with stable IDs
    passage_map = {}          # passage_text_hash → (passage_id, passage_text)
    qrels = []                # (query_id, passage_id, relevance)
    queries = {}              # query_id → query_text

    for row in tqdm(ds, desc="Processing rows"):
        query_id = row["query_id"]
        query_text = row["query"]
        queries[query_id] = query_text

        passage_texts = row["passages"]["passage_text"]
        is_selected = row["passages"]["is_selected"]

        for text, selected in zip(passage_texts, is_selected):
            # Stable passage ID from content hash (consistent across runs)
            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
            pid = int(text_hash, 16) % (2**31)  # positive int32

            if text_hash not in passage_map:
                passage_map[text_hash] = (pid, text)

            if selected == 1:
                qrels.append((query_id, pid, 1))

        # Early stop once we have enough unique passages
        if len(passage_map) >= scale:
            break

    # Build DataFrames
    passage_records = [
        {"passage_id": pid, "passage_text": text}
        for pid, text in passage_map.values()
    ]
    passages_df = pd.DataFrame(passage_records[:scale])

    # Keep only queries that have at least one relevant passage in our subset
    valid_passage_ids = set(passages_df["passage_id"])
    qrels_df = pd.DataFrame(qrels, columns=["query_id", "passage_id", "relevance"])
    qrels_df = qrels_df[qrels_df["passage_id"].isin(valid_passage_ids)]
    valid_query_ids = set(qrels_df["query_id"])

    query_records = [
        {"query_id": qid, "query_text": text}
        for qid, text in queries.items()
        if qid in valid_query_ids
    ]
    queries_df = pd.DataFrame(query_records)

    # Limit eval queries
    if len(queries_df) > config.NUM_EVAL_QUERIES:
        queries_df = queries_df.sample(
            n=config.NUM_EVAL_QUERIES,
            random_state=config.RANDOM_SEED,
        ).reset_index(drop=True)
        # Filter qrels to match sampled queries
        sampled_qids = set(queries_df["query_id"])
        qrels_df = qrels_df[qrels_df["query_id"].isin(sampled_qids)]

    # Cache to disk
    passages_df.to_parquet(passages_cache, index=False)
    queries_df.to_parquet(queries_cache, index=False)
    qrels_df.to_parquet(qrels_cache, index=False)

    print(f"   ✅ {len(passages_df):,} passages, {len(queries_df):,} queries, {len(qrels_df):,} qrels")
    print(f"   💾 Cached to {passages_cache.parent}/")
    return passages_df, queries_df, qrels_df


# ─── Convenience ──────────────────────────────────────────

def load_data(scale: int, split: str = "train"):
    """
    High-level loader: download (if needed), extract, subset, and cache.

    Args:
        scale: number of passages to extract
        split: dataset split to use

    Returns:
        (passages_df, queries_df, qrels_df)
    """
    # Check cache first (avoids loading the full dataset)
    passages_cache = _cache_path("passages", scale)
    if passages_cache.exists():
        return extract_passages_and_qrels(None, scale)

    ds = load_raw_dataset(split)
    return extract_passages_and_qrels(ds, scale)


# ─── CLI Test ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MS MARCO Data Loader")
    parser.add_argument(
        "--scale", type=int, default=10_000,
        help="Number of passages to extract (default: 10000)",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Dataset split (default: train)",
    )
    args = parser.parse_args()

    passages, queries, qrels = load_data(args.scale, args.split)

    print("\n" + "=" * 50)
    print("📊 Dataset Summary")
    print("=" * 50)
    print(f"  Passages: {len(passages):,}")
    print(f"  Queries:  {len(queries):,}")
    print(f"  Qrels:    {len(qrels):,}")

    print("\n📝 Sample passage:")
    sample = passages.iloc[0]
    print(f"  ID:   {sample['passage_id']}")
    print(f"  Text: {sample['passage_text'][:200]}...")

    print("\n🔍 Sample query:")
    sample_q = queries.iloc[0]
    print(f"  ID:   {sample_q['query_id']}")
    print(f"  Text: {sample_q['query_text']}")

    # Show relevance stats
    qrels_per_query = qrels.groupby("query_id").size()
    print(f"\n📈 Relevance stats:")
    print(f"  Avg relevant passages per query: {qrels_per_query.mean():.2f}")
    print(f"  Min: {qrels_per_query.min()}, Max: {qrels_per_query.max()}")
