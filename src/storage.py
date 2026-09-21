# Standard library
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

# Third-party dependencies
import numpy as np

# Application modules
from .errors import RAGError


def hash_file(path: Path) -> str:
    """Return the SHA-256 digest for a file's bytes."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


class ChunkStore:
    """Manage persisted chunks, embeddings, and retrieval indexes."""

    BM25_INDEX_FILES = (
        "data.csc.index.npy",
        "indices.csc.index.npy",
        "indptr.csc.index.npy",
        "params.index.json",
        "vocab.index.json",
    )

    def __init__(self, processed_dir: Path):
        """Define the paths used for all processed retrieval artifacts."""

        self.processed_dir = processed_dir
        self.bm25_index_path = self.processed_dir / "bm25_index"
        self.vector_index_path = self.processed_dir / "vector_index.faiss"
        self.chunks_path = self.processed_dir / "chunks.json"
        self.embeddings_path = self.processed_dir / "embeddings.npz"
        self.manifest_path = self.processed_dir / "manifest.json"

    def missing_bm25_files(self) -> List[Path]:
        """Return the BM25 component files that are not present."""

        return [
            self.bm25_index_path / filename
            for filename in self.BM25_INDEX_FILES
            if not (self.bm25_index_path / filename).is_file()
        ]

    def load_chunks(self) -> Any:
        """Load chunks from disk or exit with a descriptive error."""

        if not self.chunks_path.is_file():
            raise RAGError(
                f"Chunks file '{self.chunks_path}' was not found. "
                "Run the 'index' command first."
            )
        try:
            with open(self.chunks_path, "r") as f:
                chunks = json.load(f)
        except json.JSONDecodeError as exc:
            raise RAGError(
                f"Chunks file '{self.chunks_path}' contains invalid JSON. "
                "Run the 'index' command again."
            ) from exc
        except OSError as exc:
            raise RAGError(
                f"Could not read chunks file '{self.chunks_path}': {exc}"
            ) from exc
        if not isinstance(chunks, list) or not chunks:
            raise RAGError(
                f"Chunks file '{self.chunks_path}' contains no chunks. "
                "Run the 'index' command again."
            )
        required_metadata = {
            "file_path",
            "first_character_index",
            "last_character_index",
        }
        for chunk in chunks:
            if (
                not isinstance(chunk, dict)
                or not isinstance(chunk.get("content"), str)
                or not isinstance(chunk.get("metadata"), dict)
                or not required_metadata.issubset(chunk["metadata"])
            ):
                raise RAGError(
                    f"Chunks file '{self.chunks_path}' has an invalid "
                    "format. Run the 'index' command again."
                )
        return chunks

    def save_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Persist chunks as formatted JSON."""

        try:
            with open(self.chunks_path, "w") as f:
                json.dump(chunks, f, indent=2)
        except OSError as exc:
            raise RAGError(
                f"Could not write chunks file '{self.chunks_path}': {exc}"
            ) from exc

    def load_embeddings(self) -> Dict[str, np.ndarray]:
        """Load cached embeddings, returning an empty cache when absent."""

        if not self.embeddings_path.exists():
            return {}
        try:
            with np.load(self.embeddings_path, allow_pickle=False) as data:
                keys = data["keys"].tolist()
                vectors = data["vectors"]
        except (KeyError, OSError, ValueError) as exc:
            raise RAGError(
                f"Embedding cache '{self.embeddings_path}' is invalid. "
                "Delete it and run the 'index' command again."
            ) from exc
        if len(keys) != len(vectors):
            raise RAGError(
                f"Embedding cache '{self.embeddings_path}' is inconsistent. "
                "Delete it and run the 'index' command again."
            )
        return {key: vector for key, vector in zip(keys, vectors)}

    def save_embeddings(self, embeddings: Dict[str, np.ndarray]) -> None:
        """Persist embeddings and their keys in a NumPy archive."""

        keys = list(embeddings)
        vectors = np.asarray(
            [embeddings[key] for key in keys], dtype="float32"
        )
        try:
            np.savez(
                self.embeddings_path,
                keys=np.asarray(keys),
                vectors=vectors,
            )
        except OSError as exc:
            raise RAGError(
                f"Could not write embedding cache "
                f"'{self.embeddings_path}': {exc}"
            ) from exc
