# Standard library
import json
import time
from pathlib import Path
from typing import Any, Dict, List

# Third-party dependencies
import bm25s  # type: ignore[import-untyped]
import faiss
import numpy as np
from pydantic import ValidationError
from tqdm import tqdm  # type: ignore[import-untyped]

# Application modules
from .chunking import Chunker
from .embeddings import EmbeddingModel, Tokenizer
from .errors import RAGError
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

        if not self.repo.is_dir():
            raise RAGError(
                f"vLLM source repository '{self.repo}' was not found. "
                "Download vLLM 0.10.1 and place it at "
                "'data/raw/vllm-0.10.1'."
            )
        if max_chunk_size <= 0:
            raise RAGError("max_chunk_size must be a positive integer.")
        try:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RAGError(
                f"Could not create processed data directory "
                f"'{self.processed_dir}': {exc}"
            ) from exc
        chunks_path = self.processed_dir / "chunks.json"
        missing_bm25_files = self.store.missing_bm25_files()
        if not chunks_path.is_file():
            print(
                f"Chunks file '{chunks_path}' was not found; rebuilding "
                "chunks from the source repository."
            )
        if not self.store.manifest_path.is_file():
            print(
                f"Manifest file '{self.store.manifest_path}' was not found; "
                "rebuilding it."
            )
        if not self.store.embeddings_path.is_file():
            print(
                f"Embedding cache '{self.store.embeddings_path}' was not "
                "found; regenerating embeddings."
            )
        if missing_bm25_files:
            if self.store.bm25_index_path.is_dir():
                missing_names = ", ".join(
                    path.name for path in missing_bm25_files
                )
                print(
                    f"BM25 index '{self.store.bm25_index_path}' is "
                    f"incomplete (missing: {missing_names}); rebuilding it."
                )
            else:
                print(
                    f"BM25 index '{self.store.bm25_index_path}' was not "
                    "found; rebuilding it."
                )
        if not self.store.vector_index_path.is_file():
            print(
                f"Vector index '{self.store.vector_index_path}' was not "
                "found; rebuilding it."
            )
        first_index = not chunks_path.exists()
        files = self.chunker.process_files(chunks_path, max_chunk_size)
        try:
            with open(chunks_path, "r") as f:
                all_chunks = json.load(f)
        except json.JSONDecodeError as exc:
            raise RAGError(
                f"Chunks file '{chunks_path}' contains invalid JSON. Delete "
                "data/processed and run the 'index' command again."
            ) from exc
        except OSError as exc:
            raise RAGError(
                f"Could not read chunks file '{chunks_path}': {exc}"
            ) from exc
        if not isinstance(all_chunks, list):
            raise RAGError(
                f"Chunks file '{chunks_path}' has an invalid format. Delete "
                "data/processed and run the 'index' command again."
            )
        indexes_exist = (
            self.store.embeddings_path.is_file()
            and not missing_bm25_files
            and self.store.vector_index_path.is_file()
        )
        if not files and all_chunks and indexes_exist:
            print(
                "Index is already up to date; "
                "all retrieval artifacts are cached."
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
        if not all_chunks:
            raise RAGError(
                f"No supported source files were found in '{self.repo}'. "
                "Expected .py, .md, .rst, or .txt files from vLLM 0.10.1."
            )
        texts = []
        for item in tqdm(all_chunks, desc="Tokenizing"):
            texts.append(item["content"])
        tokenized_data = self.tokenizer.tokenize(texts)
        bm25_index = bm25s.BM25()
        bm25_index.index(tokenized_data)
        try:
            bm25_index.save(self.store.bm25_index_path)
        except Exception as exc:
            raise RAGError(
                f"Could not save BM25 index to "
                f"'{self.store.bm25_index_path}': {exc}"
            ) from exc

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
        try:
            faiss.write_index(
                self.vector_index,
                str(self.store.vector_index_path.with_suffix(".faiss"))
            )
        except Exception as exc:
            raise RAGError(
                f"Could not save vector index to "
                f"'{self.store.vector_index_path}': {exc}"
            ) from exc
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
        existing_results: Dict[str, Any] = {}
        if answer_path.is_file():
            try:
                with open(answer_path, "r") as f:
                    existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = {}
            except OSError as exc:
                raise RAGError(
                    f"Could not read search cache '{answer_path}': {exc}"
                ) from exc
            if not isinstance(existing_results, dict):
                existing_results = {}
            if cache_key in existing_results:
                try:
                    return [
                        MinimalSource(**result)
                        for result in existing_results[cache_key]
                    ]
                except (TypeError, ValidationError):
                    del existing_results[cache_key]

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
        cache_key = query + f"__k={k}"
        existing_results[cache_key] = [
            result.model_dump() for result in result_dict
        ]
        try:
            with open(answer_path, "w") as f:
                json.dump(existing_results, f, indent=2)
        except OSError as exc:
            raise RAGError(
                f"Could not write search cache '{answer_path}': {exc}"
            ) from exc
        return result_dict

    def search_dataset(self, dataset_path: Path, k: int,
                       save_directory: Path) -> StudentSearchResults:
        """Search every question in a dataset and save the results."""

        dataset_path = Path(dataset_path)
        save_directory = Path(save_directory)
        try:
            with open(dataset_path, "r") as f:
                dataset = RagDataset.model_validate_json(f.read())
        except FileNotFoundError as exc:
            raise RAGError(
                f"Dataset file '{dataset_path}' was not found."
            ) from exc
        except (json.JSONDecodeError, ValidationError) as exc:
            raise RAGError(
                f"Dataset file '{dataset_path}' has invalid content: {exc}"
            ) from exc
        except OSError as exc:
            raise RAGError(
                f"Could not read dataset file '{dataset_path}': {exc}"
            ) from exc

        questions = dataset.rag_questions

        search_results = []
        for item in tqdm(
            questions,
            desc="Searching questions",
            dynamic_ncols=True,
        ):
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
        try:
            save_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RAGError(
                f"Could not create output directory '{save_directory}': "
                f"{exc}"
            ) from exc

        output_name = dataset_path.name.replace("_private", "_public")
        output_path = save_directory / output_name
        print("OUTPUT PATH:", output_path)

        try:
            with open(output_path, "w") as f:
                json.dump(full_results.model_dump(), f, indent=2)
        except OSError as exc:
            raise RAGError(
                f"Could not write search results '{output_path}': {exc}"
            ) from exc

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
        search_results = self.search(query, k)
        chunks = self.store.load_chunks()
        self.model = QwenModel()
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
        except FileNotFoundError as exc:
            raise RAGError(
                "Search-results file "
                f"'{student_search_results_path}' was not found."
            ) from exc
        except (json.JSONDecodeError, ValidationError) as exc:
            raise RAGError(
                f"Search-results file '{student_search_results_path}' has "
                f"invalid content: {exc}"
            ) from exc
        except OSError as exc:
            raise RAGError(
                f"Could not read search-results file "
                f"'{student_search_results_path}': {exc}"
            ) from exc
        chunks = self.store.load_chunks()
        self.model = QwenModel()
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
        try:
            save_directory.mkdir(parents=True, exist_ok=True)
            answer_path = save_directory / "answered_questions.json"
            with open(answer_path, "w") as f:
                json.dump(full_results.model_dump(), f, indent=2)
        except OSError as exc:
            raise RAGError(
                f"Could not write answers to '{save_directory}': {exc}"
            ) from exc

        end_time = time.time()
        print(
            f"Answered {len(answered_questions)} questions "
            f"in {end_time - start_time:.2f} seconds."
        )

        return full_results
