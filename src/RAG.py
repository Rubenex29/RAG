# Standard library
import sys
from pathlib import Path

# Application modules
from .service import RAGService


def parse_k(k: int) -> None:
    """Validate that a result count is an integer."""

    if not isinstance(k, int):
        print(f"Error: k must be an integer, got {type(k).__name__}")
        sys.exit(1)


def path_to_str(name_path: str, path: Path) -> None:
    """Validate that a command-line path value is a string."""

    if not isinstance(path, str):
        print(f"Error: {name_path} must be a string path, got " +
              f"{type(path).__name__}")
        sys.exit(1)


class RAG:
    """Provide the validated public interface for the RAG service."""

    def __init__(self) -> None:
        """Initialize the underlying RAG service."""

        self.service = RAGService()

    def index(self, max_chunk_size: int = 2000) -> None:
        """Index repository files using the requested chunk size."""

        if not isinstance(max_chunk_size, int):
            print("Error: max_chunk_size must be an integer, got " +
                  f"{type(max_chunk_size).__name__}")
            sys.exit(1)
        if max_chunk_size <= 0:
            print("Error: max_chunk_size must be a positive integer.")
            sys.exit(1)
        self.service.index(max_chunk_size=max_chunk_size)

    def search(self, query: str, k: int) -> None:
        """Search for a query and print each retrieved source."""

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
        """Search a dataset and write the results to disk."""

        if not isinstance(k, int):
            print(f"Error: k must be an integer, got {type(k).__name__}")
            sys.exit(1)
        if k <= 0:
            print("Error: k must be a positive integer.")
            sys.exit(1)
        path_to_str("dataset_path", dataset_path)
        path_to_str("save_directory", save_directory)
        self.service.search_dataset(dataset_path, k, save_directory)

    def answer(self, query: str, k: int) -> None:
        """Generate and print an answer for a query."""

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
        """Generate and save answers for a retrieved dataset."""

        path_to_str("student_search_results_path", student_search_results_path)
        path_to_str("save_directory", save_directory)
        self.service.answer_dataset(student_search_results_path,
                                    save_directory)
