"""
Abstract base class for all search engines.
Every engine (FAISS, Qdrant, Elasticsearch) implements this interface
so the benchmark orchestrator can treat them uniformly.
"""

from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

import numpy as np


@dataclass
class IndexStats:
    """Stats returned after indexing."""
    num_documents: int
    index_time_sec: float
    memory_usage_mb: float


@dataclass
class SearchResult:
    """A single search result."""
    doc_id: int
    score: float


class SearchEngine(ABC):
    """Abstract interface for a vector search engine."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name (e.g. 'FAISS-Flat', 'Qdrant', 'Elasticsearch')."""
        ...

    @abstractmethod
    def index(
        self,
        doc_ids: list[int],
        embeddings: np.ndarray,
        texts: list[str] | None = None,
        metadata: dict[str, list] | None = None,
    ) -> IndexStats:
        """
        Index documents with their embeddings.

        Args:
            doc_ids:    list of document IDs
            embeddings: (N, dim) float32 array
            texts:      optional passage texts (needed for hybrid search)
            metadata:   optional dict of field_name → list of values

        Returns:
            IndexStats with timing and memory info
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Search for nearest neighbors.

        Args:
            query_embedding: (dim,) float32 array
            top_k:           number of results to return

        Returns:
            List of SearchResult sorted by relevance (best first)
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Delete index and free resources."""
        ...

    # ─── Optional Feature Methods ─────────────────────────

    def search_with_filter(
        self,
        query_embedding: np.ndarray,
        filter_field: str,
        filter_value: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search with metadata filter. Override in subclass if supported."""
        raise NotImplementedError(f"{self.name()} does not support filtered search")

    def search_fulltext(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Pure BM25 full-text search. Override in subclass if supported."""
        raise NotImplementedError(f"{self.name()} does not support full-text search")

    def search_hybrid(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Hybrid BM25 + vector search. Override in subclass if supported."""
        raise NotImplementedError(f"{self.name()} does not support hybrid search")

