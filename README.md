# 🏆 Vector Search Benchmark

**Elasticsearch vs FAISS vs Qdrant** — a systematic, reproducible benchmark using the MS MARCO Passage dataset.

## 📋 Overview

This project benchmarks three vector search approaches across five capabilities:
- **FAISS** — high-performance in-memory ANN library (Meta)
- **Qdrant** — modern vector-native database with payload filtering
- **Elasticsearch** — hybrid search engine with vector + full-text capabilities

### Metrics Measured
| Metric | Description |
|--------|-------------|
| Recall@k | Fraction of relevant docs in top-k results |
| MRR@10 | Mean Reciprocal Rank at 10 |
| Latency (P50/P95) | Query response time percentiles |
| Indexing Throughput | Passages indexed per second |
| Memory Footprint | RAM usage per engine |

### Feature Comparison Matrix
| Feature | FAISS | Qdrant | Elasticsearch |
|---------|:-----:|:------:|:-------------:|
| Pure Vector Speed | ★★★★★ | ★★★★ | ★★ |
| Hybrid Search (BM25+Vector) | — | ⚠️ partial | ★★★★★ |
| Full-Text Search | — | — | ★★★★★ |
| Metadata Filtering | ⚠️ post-filter | ★★★★ | ★★★★★ |
| Production Ecosystem | ★ | ★★★ | ★★★★★ |

> Scores reflect **production readiness** rather than raw algorithmic performance.

## 🧬 Embedding Model

**BAAI/bge-small-en-v1.5** (384-dim, normalized)

BGE-small was selected to balance embedding quality and computational cost. It produces normalized vectors suitable for cosine/inner-product similarity and is widely adopted in retrieval benchmarks.

## 📊 Dataset

**MS MARCO Passage Ranking** (Microsoft Machine Reading Comprehension)
- ~8.8M passages (we test subsets: 10k, 50k, 100k)
- Human relevance labels (qrels) for fair Recall/MRR evaluation
- Standard benchmark in information retrieval research

## ⚠️ Important Notes

**Elasticsearch version:** This benchmark uses ES 7.17, which performs **brute-force vector scoring** via `script_score`. ES 7.x does not support native ANN/HNSW indexing. ANN acceleration (the `knn` search API) is available in Elasticsearch 8.x+, which would significantly improve vector search latency.

**Qdrant mode:** Qdrant was evaluated in **embedded local mode** (disk-backed, single process). Distributed deployments with dedicated serving infrastructure may achieve lower latency.

**Synthetic metadata:** Category labels (tech/health/finance/general) were assigned randomly with a fixed seed solely to evaluate **filtering mechanics**. They do not affect semantic relevance evaluation.

**Hybrid search scoring:** In the ES hybrid mode, BM25 influences candidate ranking while the vector score provides semantic ordering. Documents matching on both keywords and semantics receive the highest combined scores.

**Reproducibility:** Results were stable across repeated runs with <3% variance on latency measurements. All random operations use `RANDOM_SEED=42`.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Elasticsearch
```bash
elasticsearch  # requires ES 7.x installed
```

### 3. Run Core Benchmark
```bash
python src/benchmark.py --scale 10000
```

### 4. Run Feature Comparison
```bash
python src/feature_benchmark.py --scale 10000 --num-queries 100
```

### 5. Generate Charts
```bash
python src/visualize.py
```

## 📁 Project Structure

```
├── config.py                   # All tunable parameters
├── requirements.txt            # Python dependencies
├── src/
│   ├── data_loader.py          # MS MARCO download & subset
│   ├── embedder.py             # Passage & query encoding (BGE-small)
│   ├── metrics.py              # Recall, MRR, latency stats
│   ├── benchmark.py            # Core benchmark orchestrator
│   ├── feature_benchmark.py    # 5-feature comparison orchestrator
│   ├── visualize.py            # Chart generation (14 charts)
│   └── engines/
│       ├── base.py             # Abstract interface + optional feature methods
│       ├── faiss_engine.py     # FAISS Flat/IVF + post-filter
│       ├── qdrant_engine.py    # Qdrant HNSW + native payload filter
│       └── elasticsearch_engine.py  # ES vector + BM25 + hybrid + filter
├── data/                       # Downloaded dataset & embeddings
└── results/                    # JSON metrics + PNG charts
```

## ⚙️ Configuration

All parameters are in `config.py` — embedding model, HNSW params, connection URLs, dataset scales, etc.

## 📉 Known Limitations

- Single embedding model (BGE-small); results may differ with larger models
- Single hardware environment (Apple M-series); server-grade hardware would shift latency ratios
- ES 7.x lacks native ANN — ES 8.x would narrow the vector speed gap significantly
- Synthetic metadata for filtering tests (not content-derived categories)
- Qdrant tested in local embedded mode, not distributed deployment

