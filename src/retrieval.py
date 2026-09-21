# Standard library
from typing import Any, Dict, List, cast

# Third-party dependencies
import bm25s  # type: ignore[import-untyped]
import faiss
import numpy as np

# Application modules
from .embeddings import EmbeddingModel, Tokenizer
from .errors import RAGError
from .storage import ChunkStore


class Retriever:
    """Retrieve chunks using BM25, vector search, and rank fusion."""

    def __init__(self, store: ChunkStore, tokenizer: Tokenizer):
        """Load the persisted chunks and retrieval indexes."""

        self.store = store
        self.tokenizer = tokenizer
        self.chunks = self.store.load_chunks()
        missing_bm25_files = self.store.missing_bm25_files()
        if missing_bm25_files:
            missing_names = ", ".join(
                path.name for path in missing_bm25_files
            )
            raise RAGError(
                f"BM25 index '{self.store.bm25_index_path}' is missing or "
                f"incomplete (missing: {missing_names}). Run the 'index' "
                "command first."
            )
        if not self.store.vector_index_path.is_file():
            raise RAGError(
                f"Vector index '{self.store.vector_index_path}' was not "
                "found. Run the 'index' command first."
            )
        try:
            self.bm25_index = bm25s.BM25.load(
                self.store.bm25_index_path
            )
        except Exception as exc:
            raise RAGError(
                f"BM25 index '{self.store.bm25_index_path}' is incomplete "
                "or invalid. Run the 'index' command again."
            ) from exc
        try:
            self.index = faiss.read_index(
                str(self.store.vector_index_path)
            )
        except Exception as exc:
            raise RAGError(
                f"Vector index '{self.store.vector_index_path}' is invalid. "
                "Run the 'index' command again."
            ) from exc
        if self.index.ntotal != len(self.chunks):
            raise RAGError(
                "The chunks file and vector index contain different numbers "
                "of entries. Run the 'index' command again."
            )
        self.model = EmbeddingModel()

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Return the top BM25 chunks for a query."""

        query_tokens = self.tokenizer.tokenize([query])
        try:
            results, scores = self.bm25_index.retrieve(query_tokens, k=k)
        except Exception as exc:
            raise RAGError(
                "Could not query the BM25 index. Run the 'index' command "
                "again."
            ) from exc
        if np.all(scores == 0):
            raise RAGError("No indexed data matched the given query.")
        retrieved_chunks = []
        for chunk_id in results[0]:
            try:
                retrieved_chunks.append(self.chunks[chunk_id])
            except IndexError as exc:
                raise RAGError(
                    "The chunks file and BM25 index are inconsistent. Run "
                    "the 'index' command again."
                ) from exc
        return retrieved_chunks

    def vector_retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Return the top vector-similarity chunks for a query."""

        query_embedding = self.model.encode([query])
        if query_embedding.shape[1] != self.index.d:
            raise RAGError(
                "The embedding model and vector index dimensions do not "
                "match. Run the 'index' command again."
            )
        scores, indices = self.index.search(
            np.array(query_embedding, dtype="float32"), k
        )
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            try:
                chunk = self.chunks[idx]
            except IndexError as exc:
                raise RAGError(
                    "The chunks file and vector index are inconsistent. Run "
                    "the 'index' command again."
                ) from exc
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
