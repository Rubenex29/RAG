# Standard library
import sys
from typing import Any, Dict, List, cast

# Third-party dependencies
import bm25s  # type: ignore[import-untyped]
import faiss
import numpy as np

# Application modules
from .embeddings import EmbeddingModel, Tokenizer
from .storage import ChunkStore


class Retriever:
    """Retrieve chunks using BM25, vector search, and rank fusion."""

    def __init__(self, store: ChunkStore, tokenizer: Tokenizer):
        """Load the persisted chunks and retrieval indexes."""

        self.store = store
        self.tokenizer = tokenizer
        self.model = EmbeddingModel()
        self.chunks = self.store.load_chunks()
        self.bm25_index = bm25s.BM25.load(self.store.bm25_index_path)
        self.index = faiss.read_index(str(self.store.vector_index_path))

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Return the top BM25 chunks for a query."""

        query_tokens = self.tokenizer.tokenize([query])
        try:
            bm25_index = bm25s.BM25.load(self.store.bm25_index_path)
        except FileNotFoundError:
            print(
                f"Error: Index file '{self.store.bm25_index_path}' not found."
            )
            sys.exit(1)
        results, scores = bm25_index.retrieve(query_tokens, k=k)
        if np.all(scores == 0):
            print("Error: No data found for the given query.")
            sys.exit(1)
        retrieved_chunks = []
        for chunk_id in results[0]:
            retrieved_chunks.append(self.chunks[chunk_id])
        return retrieved_chunks

    def vector_retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Return the top vector-similarity chunks for a query."""

        query_embedding = self.model.encode([query])
        scores, indices = self.index.search(
            np.array(query_embedding, dtype="float32"), k
        )
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(chunk)
        return results

    def hybrid_retrieve(
        self, query: str, k: int = 5, k_rrf: int = 60
    ) -> List[Dict[str, Any]]:
        """Fuse weighted BM25 and vector rankings into the top results."""

        candidate_k = min(max(k * 4, 20), len(self.chunks))
        bm25_results = self.retrieve(query, candidate_k)
        vector_results = self.vector_retrieve(query, candidate_k)

        for i, chunk in enumerate(self.chunks):
            chunk["id"] = i

        def chunk_to_id(chunk: Dict[str, Any]) -> int:
            """Return the identifier assigned to a retrieved chunk."""

            return cast(int, chunk["id"])

        scores: Dict[int, float] = {}
        chunk_by_id: Dict[int, Dict[str, Any]] = {}

        for rank, chunk in enumerate(bm25_results):
            cid = chunk_to_id(chunk)
            scores[cid] = scores.get(cid, 0.0) + 2 / (k_rrf + rank + 1)
            chunk_by_id[cid] = chunk

        for rank, chunk in enumerate(vector_results):
            cid = chunk_to_id(chunk)
            scores[cid] = scores.get(cid, 0.0) + 1 / (k_rrf + rank + 1)
            chunk_by_id[cid] = chunk

        ranked_ids = sorted(
            scores.keys(), key=lambda cid: scores[cid], reverse=True
        )
        ranked_chunks = [chunk_by_id[cid] for cid in ranked_ids]

        return ranked_chunks[:k]
