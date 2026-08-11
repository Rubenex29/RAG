from pathlib import Path

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


if __name__ == "__main__":
    fire.Fire(CLI())
