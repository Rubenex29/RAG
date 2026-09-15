import ast
import sys
import json
import numpy as np
from pathlib import Path
from pydantic import BaseModel, Field
import uuid
from typing import Any, Dict, List
import bm25s
import Stemmer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import time
import hashlib


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QwenModel:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )

    def generate_answer(self, query: str, snippets: List[str]) -> Any:
        context = "\n\n".join(
            f"[Snippet {i + 1}]\n{snippet[:600]}"
            for i, snippet in enumerate(snippets)
        )
        prompt = f"""Answer the question using the provided context.

        Use the context as your main source of information. You may combine information
        from different parts of the context and make reasonable logical inferences.

        Do not introduce facts that are unrelated to or unsupported by the context.
        If the context does not contain enough information to answer the question,
        say: "I don't have enough information in the provided context to answer this question."

        Answer the question directly and concisely.

        Question:
        {query}

        Context:
        {context}

        Answer:
        """

        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=150,
        )

        output_ids = generated_ids[0][len(inputs.input_ids[0]):]

        answer = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        ).strip()

        return answer


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5",
        )
        self.model.max_seq_length = 450

    def encode(self, texts: List[str],
               show_progress_bar: bool = False,
               batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )


class Tokenizer:
    def __init__(self) -> None:
        self.stemmer = Stemmer.Stemmer("english")

    def tokenize(self, texts: List[str]) -> Any:
        return bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer)


class ChunkStore:
    def __init__(self, processed_dir: Path):
        self.processed_dir = processed_dir
        self.bm25_index_path = self.processed_dir / "bm25_index"
        self.vector_index_path = self.processed_dir / "vector_index.faiss"
        self.chunks_path = self.processed_dir / "chunks.json"
        self.embeddings_path = self.processed_dir / "embeddings.npz"
        self.manifest_path = self.processed_dir / "manifest.json"

    def load_chunks(self) -> Any:
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
        with open(self.chunks_path, "w") as f:
            json.dump(chunks, f, indent=2)

    def load_embeddings(self) -> Dict[str, np.ndarray]:
        if not self.embeddings_path.exists():
            return {}
        data = np.load(self.embeddings_path, allow_pickle=False)
        keys = data["keys"].tolist()
        vectors = data["vectors"]
        return {key: vector for key, vector in zip(keys, vectors)}

    def save_embeddings(self, embeddings: Dict[str, np.ndarray]) -> None:
        keys = list(embeddings)
        vectors = np.asarray([embeddings[key] for key in keys], dtype="float32")
        np.savez(self.embeddings_path, keys=np.asarray(keys), vectors=vectors)


class Retriever:
    def __init__(self, store: ChunkStore, tokenizer: Tokenizer):
        self.store = store
        self.tokenizer = tokenizer
        self.model = EmbeddingModel()
        self.chunks = self.store.load_chunks()
        self.bm25_index = bm25s.BM25.load(self.store.bm25_index_path)
        self.index = faiss.read_index(str(self.store.vector_index_path))

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        query_tokens = self.tokenizer.tokenize([query])
        try:
            bm25_index = bm25s.BM25.load(self.store.bm25_index_path)
        except FileNotFoundError:
            print(f"Error: Index file '{self.store.bm25_index_path}' not found.")
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
        query_embedding = self.model.encode([query])
        scores, indices = self.index.search(np.array(query_embedding, dtype="float32"), k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:  # FAISS devolve -1 se não encontrar k resultados suficientes
                continue
            chunk = self.chunks[idx]  # self.chunks tem de ser a lista original com "content" + "metadata"
            results.append(chunk)
        return results

    def hybrid_retrieve(self, query: str, k: int = 5, k_rrf: int = 60) -> List[Dict[str, Any]]:
        candidate_k = min(max(k * 4, 20), len(self.chunks))
        bm25_results = self.retrieve(query, candidate_k)
        vector_results = self.vector_retrieve(query, candidate_k)

        for i, chunk in enumerate(self.chunks):
            chunk["id"] = i

        def chunk_to_id(chunk):
            return chunk["id"]

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

        ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        ranked_chunks = [chunk_by_id[cid] for cid in ranked_ids]

        return ranked_chunks[:k]


class Chunker:
    def __init__(self, project_root: Path, repo: Path):
        self.project_root = project_root
        self.repo = repo

    def process_files(self, chunk_path: Path) -> List[Path]:
        files = []
        manifest_path = chunk_path.with_name("manifest.json")
        old_manifest = {}
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                old_manifest = json.load(f)
        elif chunk_path.exists():
            with open(chunk_path, "r") as f:
                old_chunks = json.load(f)
            if old_chunks and all("hash" in chunk for chunk in old_chunks):
                old_manifest = {
                    chunk["metadata"]["file_path"]: chunk["hash"]
                    for chunk in old_chunks
                }
            else:
                old_manifest = {}

        current_files = [
            file for file in self.repo.rglob("*")
            if file.is_file()
            and file.suffix in {".md", ".rst", ".txt", ".py"}
        ]
        current_manifest = {
            str(file.relative_to(self.project_root)): hash_file(file)
            for file in current_files
        }

        if chunk_path.exists():
            with open(chunk_path, "r") as f:
                chunks = json.load(f)
        else:
            chunks = []
        if chunks and any("hash" not in chunk for chunk in chunks):
            chunks = []
        if not chunks:
            old_manifest = {}

        changed_paths = {
            path for path, digest in current_manifest.items()
            if old_manifest.get(path) != digest
        }
        deleted_paths = set(old_manifest) - set(current_manifest)
        stale_paths = changed_paths | deleted_paths
        if stale_paths:
            chunks = [
                chunk for chunk in chunks
                if chunk["metadata"]["file_path"] not in stale_paths
            ]

        files = [
            file for file in current_files
            if str(file.relative_to(self.project_root)) in changed_paths
        ]
        with open(chunk_path, "w") as f:
            json.dump(chunks, f, indent=2)
        with open(manifest_path, "w") as f:
            json.dump(current_manifest, f, indent=2)
        return files

    def recursive_chunking(self, text: str, file: Path,
                           chunk_size: int = 2000) -> List[Dict[str, Any]]:
        chunks = []
        stride = max(1, int(chunk_size * 0.45))
        for start in range(0, len(text), stride):
            content = text[start:start + chunk_size]
            if not content:
                continue
            end = start + len(content)
            chunks.append(
                {
                    "content": content,
                    "hash": hash_file(file),
                    "metadata": {
                        "type": "text",
                        "file_path": str(file.relative_to(self.project_root)),
                        "first_character_index": start,
                        "last_character_index": end,
                    },
                }
            )
        return chunks

    def python_chunking(self, source_code: str, file: Path,
                        chunk_size: int = 2000) -> List[Dict[str, Any]]:
        tree = ast.parse(source_code)

        lines = source_code.splitlines(keepends=True)

        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line))

        chunks = []
        parents = {
            child: node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }

        def character_offset(lineno: Any, col_offset: Any) -> Any:
            line = lines[lineno - 1]
            prefix = line.encode("utf-8")[:col_offset].decode("utf-8")
            return line_offsets[lineno - 1] + len(prefix)

        MAX_CHUNK_LENGTH = chunk_size

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            code = ast.get_source_segment(source_code, node)
            file_path = str(file.relative_to(self.project_root))

            if code is None:
                continue

            class_names = []
            parent = parents.get(node)
            while parent is not None:
                if isinstance(parent, ast.ClassDef):
                    class_names.append(parent.name)
                parent = parents.get(parent)
            context = "\n".join(
                f"class {name}" for name in reversed(class_names)
            )
            indexed_code = f"{context}\n{code}" if context else code

            # Function fits in one chunk
            if len(indexed_code) <= MAX_CHUNK_LENGTH:
                chunks.append(
                    {
                        "content": indexed_code,
                        "hash": hash_file(file),
                        "metadata": {
                            "type": "function",
                            "file_path": file_path,
                            "first_character_index": character_offset(
                                node.lineno, node.col_offset
                            ),
                            "last_character_index": character_offset(
                                node.end_lineno, node.end_col_offset
                            ),
                        },
                    }
                )
                continue

            # Function is too large: split it into chunks
            start = character_offset(node.lineno, node.col_offset)
            for chunk_start in range(0, len(code), MAX_CHUNK_LENGTH):
                chunk_end = min(chunk_start + MAX_CHUNK_LENGTH, len(code))
                chunk_content = code[chunk_start:chunk_end]
                if context:
                    chunk_content = f"{context}\n{chunk_content}"

                chunks.append(
                    {
                        "content": chunk_content,
                        "hash": hash_file(file),
                        "metadata": {
                            "type": "function",
                            "file_path": file_path,
                            "first_character_index": start + chunk_start,
                            "last_character_index": start + chunk_end,
                        },
                    }
                )

        return chunks

    def chunk_file(self, file: Path,
                   max_chunk_size: int) -> List[Dict[str, Any]]:
        if file.suffix in {".md", ".rst", ".txt"}:
            text = file.read_text(encoding="utf-8", errors="ignore")
            return self.recursive_chunking(text, file, max_chunk_size)

        if file.suffix in {".py"}:
            text = file.read_text(encoding="utf-8", errors="ignore")
            return self.python_chunking(text, file, max_chunk_size)

        return []


class RAGService:
    def __init__(self) -> None:
        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent
        self.repo = (self.project_root / "data/raw/vllm-0.10.1").resolve()
        self.processed_dir = self.project_root / "data/processed"

        self.tokenizer = Tokenizer()
        self.retriever: Retriever | None = None
        self.store = ChunkStore(self.processed_dir)
        self.chunker = Chunker(self.project_root, self.repo)

    def index(self, max_chunk_size: int = 2000) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = self.processed_dir / "chunks.json"
        first_index = not chunks_path.exists()
        files = self.chunker.process_files(chunks_path)
        # Load chunks that were NOT changed
        if chunks_path.exists():
            with open(chunks_path, "r") as f:
                all_chunks = json.load(f)
        else:
            all_chunks = []
        if not files and all_chunks and self.store.embeddings_path.exists():
            print("Index is already up to date; all chunks and embeddings are cached.")
            return
        if not all_chunks:
            first_index = True
        # Add new chunks from changed/new files
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
            self.model = EmbeddingModel()
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
        return result_dict

    def search_dataset(self, dataset_path: Path, k: int,
                       save_directory: Path) -> StudentSearchResults:
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

    def answer(self, query: str, k: int) -> Any:
        """Answer a query using the RAG system."""
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


def parse_k(k: int) -> None:
    if not isinstance(k, int):
        print(f"Error: k must be an integer, got {type(k).__name__}")
        sys.exit(1)


def path_to_str(name_path: str, path: Path) -> None:
    if not isinstance(path, str):
        print(f"Error: {name_path} must be a string PAth, got " +
              f"{type(path).__name__}")
        sys.exit(1)


class RAG:
    def __init__(self) -> None:
        self.service = RAGService()

    def index(self, max_chunk_size: int = 2000) -> None:
        if not isinstance(max_chunk_size, int):
            print("Error: max_chunk_size must be an integer, got " +
                  f"{type(max_chunk_size).__name__}")
            sys.exit(1)
        self.service.index(max_chunk_size=max_chunk_size)

    def search(self, query: str, k: int) -> None:
        if not isinstance(k, int):
            print(f"Error: k must be an integer, got {type(k).__name__}")
            sys.exit(1)
        if not isinstance(query, str):
            print(f"Error: query must be a string, got {type(query).__name__}")
            sys.exit(1)
        if k <= 0:
            print("Error: k must be a positive integer.")
            sys.exit(1)
        results = self.service.search(query, k)
        for idx, entry in enumerate(results):
            print(f"Result {idx + 1}:")
            print(f"File Path: {entry.file_path}")
            print(f"First Character Index: {entry.first_character_index}")
            print(f"Last Character Index: {entry.last_character_index}")
            print("-" * 40)

    def search_dataset(self, dataset_path: Path, k: int,
                       save_directory: Path) -> None:
        if not isinstance(k, int):
            print(f"Error: k must be an integer, got {type(k).__name__}")
            sys.exit(1)
        path_to_str("dataset_path", dataset_path)
        path_to_str("save_directory", save_directory)
        self.service.search_dataset(dataset_path, k, save_directory)

    """Answer a query using the RAG system."""
    def answer(self, query: str, k: int) -> None:
        if not isinstance(k, int):
            print(f"Error: k must be an integer, got {type(k).__name__}")
            sys.exit(1)
        if not isinstance(query, str):
            print(f"Error: query must be a string, got {type(query).__name__}")
            sys.exit(1)
        if k <= 0:
            print("Error: k must be a positive integer.")
            sys.exit(1)
        print(self.service.answer(query, k))

    def answer_dataset(self, student_search_results_path: Path,
                       save_directory: Path) -> None:
        path_to_str("student_search_results_path", student_search_results_path)
        path_to_str("save_directory", save_directory)
        self.service.answer_dataset(student_search_results_path,
                                    save_directory)


"""
export HF_HOME=/sgoinfre/$(whoami)/hf_cache
export UV_CACHE_DIR=/sgoinfre/$USER/uv_cache
uv run python -m src index --max_chunk_size 2000
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions
"""
