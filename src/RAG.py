
import ast
import json
from pathlib import Path
from pydantic import BaseModel, Field
import uuid
from typing import List
import bm25s
import Stemmer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


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
    question_str: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int


class QwenModel:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def generate_answer(self, query: str, snippets: List[str]) -> str:
        context = "\n\n".join(
            f"[Snippet {i + 1}]\n{snippet}"
            for i, snippet in enumerate(snippets)
        )

        prompt = f"""Answer the question using the provided context.

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
            max_new_tokens=64,
        )

        output_ids = generated_ids[0][len(inputs.input_ids[0]):]

        answer = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        ).strip()

        return answer


class Tokenizer:
    def __init__(self):
        self.stemmer = Stemmer.Stemmer("english")

    def tokenize(self, texts):
        return bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer)


class ChunkStore:
    def __init__(self, processed_dir: Path):
        self.processed_dir = processed_dir
        self.index_path = self.processed_dir / "bm25_index"
        self.chunks_path = self.processed_dir / "chunks.json"

    def load_chunks(self):
        with open(self.chunks_path, "r") as f:
            return json.load(f)

    def save_chunks(self, chunks):
        with open(self.chunks_path, "w") as f:
            json.dump(chunks, f, indent=2)


class Retriever:
    def __init__(self, store: ChunkStore, tokenizer: Tokenizer):
        self.store = store
        self.tokenizer = tokenizer

    def retrieve(self, query: str, k: int):
        query_tokens = self.tokenizer.tokenize([query])
        all_chunks = self.store.load_chunks()
        bm25_index = bm25s.BM25.load(self.store.index_path)
        results, scores = bm25_index.retrieve(query_tokens, k=k)
        retrieved_chunks = []
        for chunk_id in results[0]:
            retrieved_chunks.append(all_chunks[chunk_id])
        return retrieved_chunks


class Chunker:
    def __init__(self, project_root: Path, repo: Path):
        self.project_root = project_root
        self.repo = repo

    def process_files(self):
        files = []
        for file in self.repo.rglob("*"):
            if file.is_file():
                files.append(file)
        return files

    def recursive_chunking(self, text: str, file: Path, chunk_size=2000):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 10,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            add_start_index=True,
        )
        chunks = []
        docs = text_splitter.create_documents([text])
        for doc in docs:
            start: int = doc.metadata["start_index"]
            end: int = start + len(doc.page_content)
            chunks.append(
                {
                    "content": doc.page_content,
                    "metadata": {
                        "type": "text",
                        "file_path": str(file.relative_to(self.project_root)),
                        "first_character_index": start,
                        "last_character_index": end,
                    },
                }
            )
        return chunks

    def python_chunking(self, source_code: str, file: Path):
        tree = ast.parse(source_code)
        chunks = []

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                code = ast.get_source_segment(source_code, node)
                chunks.append(
                    {
                        "content": code,
                        "metadata": {
                            "type": "function",
                            "file_path": str(
                                file.relative_to(self.project_root)
                            ),
                            "first_character_index": node.lineno,
                            "last_character_index": node.end_lineno,
                        },
                    }
                )
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        code = ast.get_source_segment(source_code, child)
                        chunks.append(
                            {
                                "content": code,
                                "metadata": {
                                    "type": "method",
                                    "file_path": str(
                                        file.relative_to(self.project_root)
                                    ),
                                    "first_character_index": child.lineno,
                                    "last_character_index": child.end_lineno,
                                },
                            }
                        )

        return chunks

    def chunk_file(self, file: Path, max_chunk_size: int):
        if file.suffix in {".md", ".rst", ".txt"}:
            text = file.read_text(encoding="utf-8", errors="ignore")
            return self.recursive_chunking(text, file, max_chunk_size)

        if file.suffix in {".py"}:
            text = file.read_text(encoding="utf-8", errors="ignore")
            return self.python_chunking(text, file)

        return []


class RAGService:
    def __init__(self):
        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent
        self.repo = (self.project_root / "data/raw/vllm-0.10.1").resolve()
        self.processed_dir = self.project_root / "data/processed"

        self.tokenizer = Tokenizer()
        self.store = ChunkStore(self.processed_dir)
        self.chunker = Chunker(self.project_root, self.repo)
        self.retriever = Retriever(self.store, self.tokenizer)
        self.model = QwenModel()

    def index(self, max_chunk_size=2000):
        files = self.chunker.process_files()

        all_chunks = []
        for _, file in tqdm(
            enumerate(files), total=len(files), desc="Chunking"
        ):
            chunks = self.chunker.chunk_file(file, max_chunk_size)
            for chunk in chunks:
                all_chunks.append(chunk)

        texts_to_tokenize = []
        for item in tqdm(all_chunks, desc="Tokenizing"):
            texts_to_tokenize.append(item["content"])

        tokenized_data = self.tokenizer.tokenize(texts_to_tokenize)

        bm25_index = bm25s.BM25()
        bm25_index.index(tokenized_data)
        bm25_index.save(self.store.index_path)

        self.store.save_chunks(all_chunks)
        print(
            "Ingestion complete! Indexed "
            "{} chunks under data/processed.".format(len(all_chunks))
        )
        print("BM25 index saved to data/processed/bm25_index.")

    def search(self, query: str, k: int):
        results = self.retriever.retrieve(query, k)
        result_dict = []
        for result in results:
            first_char_index = result["metadata"]["first_character_index"]
            last_char_index = result["metadata"]["last_character_index"]
            result_dict.append({
                "file_path": result["metadata"]["file_path"],
                "first_character_index": first_char_index,
                "last_character_index": last_char_index,
            })
        return result_dict

    def search_dataset(self, dataset_path: Path, k: int, save_directory: Path):
        save_directory = Path(save_directory)
        with open(dataset_path, "r") as f:
            questions = json.load(f)

        questions = questions["rag_questions"]

        search_results = []

        for item in questions:
            result = MinimalSearchResults(
                question_id=item["question_id"],
                question_str=item["question"],
                retrieved_sources=self.search(item["question"], k),
            )

            search_results.append(result)

        full_results = StudentSearchResults(
            search_results=search_results,
            k=k,
        )
        print("OUTPUT PATH:", save_directory / "dataset_docs_public.json")
        save_directory.mkdir(parents=True, exist_ok=True)

        with open(save_directory / "dataset_docs_public.json", "w") as f:
            json.dump(full_results.model_dump(), f, indent=2)

        return full_results

    def answer(self, query: str, k: int):
        search_results = self.search(query, k)
        # wanna access the chunks in self.store where file_path in search_results and first_character_index and last_character_index match
        chunks = self.store.load_chunks()
        relevant_chunks = []
        for result in search_results:
            for chunk in chunks:
                if (
                    chunk["metadata"]["file_path"] == result["file_path"]
                    and chunk["metadata"]["first_character_index"]
                    == result["first_character_index"]
                    and chunk["metadata"]["last_character_index"]
                    == result["last_character_index"]
                ):
                    relevant_chunks.append(chunk["content"])
        return self.model.generate_answer(query, relevant_chunks)

    def answer_dataset(
        self,
        student_search_results_path: Path,
        save_directory: Path,
    ):
        with open(student_search_results_path, "r") as f:
            search_results = json.load(f)

        answered_questions = []
        chunks = self.store.load_chunks()
        start_time = time.time()
        for i, item in enumerate(search_results["search_results"], 1):
            print(f"Answering question {i}/{len(search_results['search_results'])}")

            first = item["retrieved_sources"][0]["first_character_index"]
            last = item["retrieved_sources"][0]["last_character_index"]

            snippets = []
            for chunk in chunks:
                if (
                    chunk["metadata"]["file_path"] == item["retrieved_sources"][0]["file_path"]
                    and chunk["metadata"]["first_character_index"] == first
                    and chunk["metadata"]["last_character_index"] == last
                ):
                    snippets.append(chunk["content"])
            answer = self.model.generate_answer(
                item["question_str"],
                snippets,
            )

            answered_question = AnsweredQuestion(
                question_id=item["question_id"],
                question=item["question_str"],
                sources=item["retrieved_sources"],
                answer=answer,
            )

            answered_questions.append(answered_question)

        full_results = RagDataset(
            rag_questions=answered_questions
        )

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        with open(save_directory / "answered_questions.json", "w") as f:
            json.dump(full_results.model_dump(), f, indent=2)
        end_time = time.time()
        print(f"Answered {len(answered_questions)} questions in {end_time - start_time:.2f} seconds.")
        return full_results


class RAG:
    def __init__(self):
        self.service = RAGService()

    def index(self, max_chunk_size=2000):
        self.service.index(max_chunk_size=max_chunk_size)

    def search(self, query: str, k: int):
        results = self.service.search(query, k)
        for idx, entry in enumerate(results):
            print(f"Result {idx + 1}:")
            print(f"File Path: {entry['file_path']}")
            print(f"First Character Index: {entry['first_character_index']}")
            print(f"Last Character Index: {entry['last_character_index']}")
            print("-" * 40)

    def search_dataset(self, dataset_path: Path, k: int, save_directory: Path):
        self.service.search_dataset(dataset_path, k, save_directory)

    def answer(self, query: str, k: int):
        print(self.service.answer(query, k))

    def answer_dataset(self, student_search_results_path: Path, save_directory: Path):
        self.service.answer_dataset(student_search_results_path, save_directory)

# export HF_HOME=/sgoinfre/$(whoami)/hf_cache
