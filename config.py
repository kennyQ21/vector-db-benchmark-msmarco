"""
Central configuration for the Vector Search Benchmark.
All tunable parameters in one place.
"""

# ─── Reproducibility ──────────────────────────────────────
RANDOM_SEED = 42

# ─── Dataset ──────────────────────────────────────────────
DATASET_NAME = "ms_marco"
DATASET_VERSION = "v1.1"
DATA_DIR = "data"
DATASET_SCALES = [10_000, 50_000, 100_000, 250_000]
NUM_EVAL_QUERIES = 500

# ─── Embedding Model ─────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 128
EMBEDDINGS_DIR = "data/embeddings"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
NORMALIZE_EMBEDDINGS = True
SIMILARITY_METRIC = "cosine"  # cosine with normalized vectors = inner product

# ─── FAISS ────────────────────────────────────────────────
FAISS_INDEX_TYPES = ["flat"]          # IVF disabled: segfaults on faiss-cpu ARM/Python 3.14
# FAISS_INDEX_TYPES = ["flat", "ivf"]  # uncomment when faiss-cpu ARM is fixed
FAISS_IVF_NLIST = 100
FAISS_IVF_NPROBE = 10

# ─── Qdrant ───────────────────────────────────────────────
# Using local persistent mode (no Docker needed)
QDRANT_STORAGE_PATH = "data/qdrant_storage"
# QDRANT_HOST = "localhost"     # uncomment for Docker mode
# QDRANT_PORT = 6333
QDRANT_COLLECTION = "msmarco_benchmark"
QDRANT_HNSW_M = 16
QDRANT_HNSW_EF_CONSTRUCT = 100
QDRANT_HNSW_EF_SEARCH = 64           # search-time recall/latency tradeoff

# ─── Elasticsearch ────────────────────────────────────────
ES_HOST = "http://localhost:9200"
ES_INDEX = "msmarco_benchmark"
ES_HNSW_M = 16
ES_HNSW_EF_CONSTRUCTION = 100
ES_NUM_CANDIDATES_LIST = [50, 100, 200]   # enables latency vs recall curves
ES_BULK_SIZE = 500

# ─── Evaluation ───────────────────────────────────────────
TOP_K_VALUES = [10, 50, 100]
MRR_K = 10

# ─── System Logging ───────────────────────────────────────
SYSTEM_INFO_LOG = True                # capture CPU, RAM, GPU info for blog

# ─── Output ───────────────────────────────────────────────
RESULTS_DIR = "results"
