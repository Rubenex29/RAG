# Standard library
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Third-party dependencies
import bm25s  # type: ignore[import-untyped]
import faiss
import numpy as np
from tqdm import tqdm  # type: ignore[import-untyped]

# Application modules
from .chunking import Chunker
from .embeddings import EmbeddingModel, Tokenizer
from .generation import QwenModel
from .retrieval import Retriever
from .schemas import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .storage import ChunkStore


class RAGService:
    """Coordinate indexing, retrieval, answer generation, and persistence."""

    def __init__(self) -> None:
        """Initialize paths and collaborators for the RAG workflow."""

        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent
        self.repo = (self.project_root / "data/raw/vllm-0.10.1").resolve()
        self.processed_dir = self.project_root / "data/processed"

        self.tokenizer = Tokenizer()
        self.retriever: Retriever | None = None
        self.store = ChunkStore(self.processed_dir)
        self.chunker = Chunker(self.project_root, self.repo)

    def index(self, max_chunk_size: int = 2000) -> None:
        """Rebuild indexes for changed repository files and their chunks."""

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = self.processed_dir / "chunks.json"
        first_index = not chunks_path.exists()
        files = self.chunker.process_files(chunks_path)
        if chunks_path.exists():
            with open(chunks_path, "r") as f:
                all_chunks = json.load(f)
        else:
            all_chunks = []
        if not files and all_chunks and self.store.embeddings_path.exists():
            print(
                "Index is already up to date; "
                "all chunks and embeddings are cached."
            )
            return
        if not all_chunks:
            first_index = True
        new_chunk_count = 0
        for _, file in tqdm(
            enumerate(files),
            total=len(files),
            desc="Indexing files" if first_index else "Reindexing files",
        ):
            chunks = self.chunker.chunk_file(file, max_chunk_size)
            new_chunk_count += len(chunks)
            for chunk in chunks:
                all_chunks.append(chunk)
        action = "Indexed" if first_index else "Reindexed"
        print(
            f"{action} {len(files)} files and generated "
            f"{new_chunk_count} new chunks."
        )
        texts = []
        for item in tqdm(all_chunks, desc="Tokenizing"):
            texts.append(item["content"])
        tokenized_data = self.tokenizer.tokenize(texts)
        bm25_index = bm25s.BM25()
        bm25_index.index(tokenized_data)
        bm25_index.save(self.store.bm25_index_path)

        def chunk_key(chunk: Dict[str, Any]) -> str:
            """Build the cache key for a chunk embedding."""

            metadata = chunk["metadata"]
            return "|".join(
                [
                    chunk["hash"],
                    metadata["file_path"],
                    str(metadata["first_character_index"]),
                    str(metadata["last_character_index"]),
                ]
            )

        embedding_cache = self.store.load_embeddings()
        missing_chunks = [
            chunk for chunk in all_chunks
            if chunk_key(chunk) not in embedding_cache
        ]
        if missing_chunks:
            self.model: EmbeddingModel | QwenModel = EmbeddingModel()
            new_embeddings = self.model.encode(
                [chunk["content"] for chunk in missing_chunks],
                show_progress_bar=True,
                batch_size=64,
            )
            for chunk, embedding in zip(missing_chunks, new_embeddings):
                embedding_cache[chunk_key(chunk)] = embedding

        self.doc_embeddings = np.asarray(
            [embedding_cache[chunk_key(chunk)] for chunk in all_chunks],
            dtype="float32",
        )
        self.store.save_embeddings(
            {chunk_key(chunk): embedding_cache[chunk_key(chunk)]
             for chunk in all_chunks}
        )
        dim = self.doc_embeddings.shape[1]
        self.vector_index = faiss.IndexFlatIP(dim)
        self.vector_index.add(
            np.array(self.doc_embeddings, dtype="float32")
        )
        faiss.write_index(
            self.vector_index,
            str(self.store.vector_index_path.with_suffix(".faiss"))
        )
        self.store.save_chunks(all_chunks)
        print(
            "Ingestion complete! Indexed "
            "{} chunks under data/processed.".format(len(all_chunks))
        )

        print("Vector index saved to data/processed/vector_index.faiss.")
        print("BM25 index saved to data/processed/bm25_index.")

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """Retrieve sources, using the disk cache when available."""

        answer_path = self.processed_dir / "search_cache.json"
        cache_key = query + f"__k={k}"
        if answer_path.exists():
            with open(answer_path, "r") as f:
                try:
                    existing_results = json.load(f)
                except json.JSONDecodeError:
                    existing_results = {}
            if cache_key in existing_results:
                return [
                    MinimalSource(**result)
                    for result in existing_results[cache_key]
                ]

        if self.retriever is None:
            self.retriever = Retriever(self.store, self.tokenizer)
        results = self.retriever.hybrid_retrieve(query, k)
        result_dict = []
        for result in results:
            first_char_index = result["metadata"]["first_character_index"]
            last_char_index = result["metadata"]["last_character_index"]
            result_dict.append(MinimalSource(
                file_path=result["metadata"]["file_path"],
                first_character_index=first_char_index,
                last_character_index=last_char_index,
            ))
        answer_path = self.processed_dir / "search_cache.json"
        if answer_path.exists():
            with open(answer_path, "r") as f:
                try:
                    existing_results = json.load(f)
                except json.JSONDecodeError:
                    existing_results = {}
        else:
            existing_results = {}
        cache_key = query + f"__k={k}"
        existing_results[cache_key] = [result.dict() for result in result_dict]
        with open(answer_path, "w") as f:
            json.dump(existing_results, f, indent=2)
        return result_dict

    def search_dataset(self, dataset_path: Path, k: int,
                       save_directory: Path) -> StudentSearchResults:
        """Search every question in a dataset and save the results."""

        dataset_path = Path(dataset_path)
        save_directory = Path(save_directory)
        try:
            with open(dataset_path, "r") as f:
                dataset = RagDataset.model_validate_json(f.read())
        except FileNotFoundError:
            print(f"Error: File '{dataset_path}' not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {dataset_path}: {e}")
            sys.exit(1)

        questions = dataset.rag_questions

        search_results = []
        for item in questions:
            result = MinimalSearchResults(
                question_id=item.question_id,
                question=item.question,
                retrieved_sources=self.search(item.question, k),
            )
            search_results.append(result)
        full_results = StudentSearchResults(
            search_results=search_results,
            k=k,
        )
        save_directory.mkdir(parents=True, exist_ok=True)

        output_name = dataset_path.name.replace("_private", "_public")
        output_path = save_directory / output_name
        print("OUTPUT PATH:", output_path)

        with open(output_path, "w") as f:
            json.dump(full_results.model_dump(), f, indent=2)

        return full_results

    def get_snippets(self, chunks: List[Dict[str, Any]],
                     sources: List[MinimalSource]) -> List[str]:
        """Return chunk text matching the supplied source locations."""

        snippets = []

        for source in sources:
            source_file_path = source.file_path
            source_first = source.first_character_index
            source_last = source.last_character_index

            for chunk in chunks:
                metadata = chunk["metadata"]

                if (
                    metadata["file_path"] == source_file_path
                    and metadata["first_character_index"] == source_first
                    and metadata["last_character_index"] == source_last
                ):
                    snippets.append(chunk["content"])
                    break

        return snippets

    def answer(self, query: str, k: int) -> str:
        """Generate an answer for a query using its retrieved sources."""
        self.model = QwenModel()
        search_results = self.search(query, k)
        chunks = self.store.load_chunks()
        snippets = self.get_snippets(chunks, search_results)
        answer = self.model.generate_answer(query, snippets)
        return answer

    def answer_dataset(
        self,
        student_search_results_path: Path,
        save_directory: Path,
    ) -> StudentSearchResultsAndAnswer:
        """Generate and save answers for retrieved dataset results."""

        try:
            with open(student_search_results_path, "r") as f:
                search_results = StudentSearchResults.model_validate_json(
                    f.read())
        except FileNotFoundError:
            print(f"Error: File '{student_search_results_path}' not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print("Error decoding JSON from" +
                  f" {student_search_results_path}: {e}")
            sys.exit(1)
        self.model = QwenModel()
        chunks = self.store.load_chunks()
        answered_questions = []
        start_time = time.time()

        for item in tqdm(
            search_results.search_results,
            desc="Answering questions",
            dynamic_ncols=True,
        ):
            snippets = self.get_snippets(
                chunks,
                item.retrieved_sources,
            )

            answer = self.model.generate_answer(
                item.question,
                snippets,
            )
            answered_questions.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=item.retrieved_sources,
                    answer=answer,
                )
            )

        full_results = StudentSearchResultsAndAnswer(
            search_results=answered_questions,
            k=search_results.k,
        )

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        with open(save_directory / "answered_questions.json", "w") as f:
            json.dump(full_results.model_dump(), f, indent=2)

        end_time = time.time()
        print(
            f"Answered {len(answered_questions)} questions "
            f"in {end_time - start_time:.2f} seconds."
        )

        return full_results
