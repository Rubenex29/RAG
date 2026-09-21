# Standard library
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Third-party dependencies
import numpy as np


def hash_file(path: Path) -> str:
    """Return the SHA-256 digest for a file's bytes."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


class ChunkStore:
    """Manage persisted chunks, embeddings, and retrieval indexes."""

    def __init__(self, processed_dir: Path):
        """Define the paths used for all processed retrieval artifacts."""

        self.processed_dir = processed_dir
        self.bm25_index_path = self.processed_dir / "bm25_index"
        self.vector_index_path = self.processed_dir / "vector_index.faiss"
        self.chunks_path = self.processed_dir / "chunks.json"
        self.embeddings_path = self.processed_dir / "embeddings.npz"
        self.manifest_path = self.processed_dir / "manifest.json"

    def load_chunks(self) -> Any:
        """Load chunks from disk or exit with a descriptive error."""

        try:
            with open(self.chunks_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Chunks file '{self.chunks_path}' not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {self.chunks_path}: {e}")
            sys.exit(1)

    def save_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Persist chunks as formatted JSON."""

        with open(self.chunks_path, "w") as f:
            json.dump(chunks, f, indent=2)

    def load_embeddings(self) -> Dict[str, np.ndarray]:
        """Load cached embeddings, returning an empty cache when absent."""

        if not self.embeddings_path.exists():
            return {}
        data = np.load(self.embeddings_path, allow_pickle=False)
        keys = data["keys"].tolist()
        vectors = data["vectors"]
        return {key: vector for key, vector in zip(keys, vectors)}

    def save_embeddings(self, embeddings: Dict[str, np.ndarray]) -> None:
        """Persist embeddings and their keys in a NumPy archive."""

        keys = list(embeddings)
        vectors = np.asarray(
            [embeddings[key] for key in keys], dtype="float32"
        )
        np.savez(self.embeddings_path, keys=np.asarray(keys), vectors=vectors)
