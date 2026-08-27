from pathlib import Path
import sys
import fire

from .RAG import RAG

    
class CLI:
    def __init__(self):
        self.service = RAG()

    def index(self, max_chunk_size=2000):
        self.service.index(max_chunk_size=max_chunk_size)

    def search(self, query: str, k: int):
        self.service.search(query, k)

    def search_dataset(self, dataset_path: Path, k: int, save_directory: Path):
        self.service.search_dataset(dataset_path, k, save_directory)

    def answer(self, query: str, k: int):
        self.service.answer(query, k)

    def answer_dataset(self, student_search_results_path: Path, save_directory: Path):
        self.service.answer_dataset(student_search_results_path, save_directory)


VALID_COMMANDS = {
    "answer",
    "answer_dataset",
    "index",
    "search",
    "search_dataset",
}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in VALID_COMMANDS:
        print(f"Error: Invalid command: {sys.argv[1]}")
        sys.exit(1)

    fire.Fire(CLI())
