# Standard library
from typing import Any, List

# Third-party dependencies
import bm25s  # type: ignore[import-untyped]
import numpy as np
import Stemmer  # type: ignore[import-not-found]
from sentence_transformers import SentenceTransformer

# Application modules
from .errors import RAGError


class EmbeddingModel:
    """Encode text into normalized vectors for semantic retrieval."""

    def __init__(self) -> None:
        """Load the embedding model and set its maximum sequence length."""

        model_name = "BAAI/bge-small-en-v1.5"
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RAGError(
                f"Could not load embedding model '{model_name}'. Ensure "
                "that it is cached locally or that internet access is "
                "available."
            ) from exc
        self.model.max_seq_length = 450

    def encode(self, texts: List[str],
               show_progress_bar: bool = False,
               batch_size: int = 64) -> np.ndarray:
        """Encode text using the configured batching and progress settings."""

        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )


class Tokenizer:
    """Tokenize English text for the BM25 index."""

    def __init__(self) -> None:
        """Initialize the English stemmer used during tokenization."""

        self.stemmer = Stemmer.Stemmer("english")

    def tokenize(self, texts: List[str]) -> Any:
        """Tokenize and stem text while removing English stop words."""

        return bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer)
