
import ast
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bm25s
import Stemmer
from tqdm import tqdm


class RAG:
    def __init__(self):
        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent
        self.repo = (self.project_root / "data/raw/vllm-0.10.1").resolve()

    def index(self, max_chunk_size=2000):
        def process_files():
            files = []
            for file in self.repo.rglob("*"):
                if file.is_file():
                    files.append(file)
            return files

        def recursive_chunking(text: str, file: str, chunk_size=2000):
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
                chunks.append({
                    "content": doc.page_content,
                    "metadata": {
                        "type": "text",
                        "file_path": str(file.relative_to(self.project_root)),
                        "first_character_index": start,
                        "last_character_index": end,
                    }
                })
            return chunks

        def python_chunking(source_code: str, file: str):
            tree = ast.parse(source_code)

            chunks = []

            for node in tree.body:

                # Top-level functions
                if isinstance(node, ast.FunctionDef):
                    code = ast.get_source_segment(source_code, node)
                    chunks.append({
                        "content": code,
                        "metadata": {
                            "type": "function",
                            "file_path": str(file.relative_to(self.project_root)),
                            "name": node.name,
                            "first_character_index": node.lineno,
                            "last_character_index": node.end_lineno,
                        }
                    })

                # Classes
                elif isinstance(node, ast.ClassDef):

                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):

                            code = ast.get_source_segment(source_code, child)

                            chunks.append({
                                "content": code,
                                "metadata": {
                                    "type": "method",
                                    "file_path": str(file.relative_to(self.project_root)),
                                    "class": node.name,
                                    "name": child.name,

                                    "first_character_index": child.lineno,
                                    "last_character_index": child.end_lineno,
                                }
                            })

            return chunks

        def run():
            chunks = []
            self.all_chunks = []
            for i, file in tqdm(enumerate(self.files), total=len(self.files), desc="Processing files"):
                if file.suffix in {".md", ".rst", ".txt"}:
                    text = file.read_text(encoding="utf-8", errors="ignore")
                    chunks = recursive_chunking(text, file, max_chunk_size)
                elif file.suffix in {".py"}:
                    text = file.read_text(encoding="utf-8", errors="ignore")
                    chunks = python_chunking(text, file)

                for chunk in chunks:
                    self.all_chunks.append(chunk)

        def build_index():
            english_stemmer = Stemmer.Stemmer('english')
            self.tokenizer = lambda texts: bm25s.tokenize(
                texts, 
                stopwords="en", 
                stemmer=english_stemmer
            )
            texts_to_tokenize = []

            for item in self.all_chunks:
                if isinstance(item, dict):
                    # If it's a dictionary (like your code files), extract the 'content'
                    texts_to_tokenize.append(item['content'])
                elif isinstance(item, str):
                    # If it's already a string (like your markdown files), just use it
                    texts_to_tokenize.append(item)

            # Now pass the clean list of strings to the tokenizer
            self.tokenized_data = self.tokenizer(texts_to_tokenize)

            self.bm25_index = bm25s.BM25()
            self.bm25_index.index(self.tokenized_data)
            self.bm25_index.save("data/processed/bm25_index")

        def search_index():
            # 4. Search the index
            query = "How do I use Cmake to build a project?"
            # Remember: You must tokenize the query using the exact same tokenizer!
            query_tokens = self.tokenizer([query])

            # Retrieve the top 1 result
            results, scores = self.bm25_index.retrieve(query_tokens, k=1)

            # 1. Get the integer ID of the best match
            best_match_id = results[0][0]

            # 2. Get the score
            best_score = scores[0][0]

            # 3. Retrieve the actual content from your original list
            winning_chunk = self.all_chunks[best_match_id]

        self.files = process_files()
        run()
        build_index()
        search_index()

    def test(self):
        print(len(self.all_chunks))
        for i, chunk in enumerate(self.all_chunks):
            if "metadata" in chunk:
                print(f"Chunk {i+1}, size: {len(chunk['content'])} characters")
                print(chunk)
                print("-" * 40)
            if i == 920:
                break


