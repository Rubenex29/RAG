# Standard library
from pathlib import Path
import sys

# Third-party dependencies
import fire  # type: ignore[import-untyped]

# Application modules
from .RAG import RAG
from .errors import RAGError


class CLI:
    """Expose RAG operations through the Fire command-line interface."""

    def __init__(self) -> None:
        """Initialize the public RAG interface."""

        self.service = RAG()

    def index(self, max_chunk_size: int = 2000) -> None:
        """Index the repository using the requested chunk size."""

        self.service.index(max_chunk_size=max_chunk_size)

    def search(self, query: str, k: int) -> None:
        """Search for a query and print its source locations."""

        self.service.search(query, k)

    def search_dataset(
        self, dataset_path: Path, k: int, save_directory: Path
    ) -> None:
        """Search all questions in a dataset and save the output."""

        self.service.search_dataset(dataset_path, k, save_directory)

    def answer(self, query: str, k: int) -> None:
        """Generate and print an answer for a query."""

        self.service.answer(query, k)

    def answer_dataset(self, student_search_results_path: Path,
                       save_directory: Path) -> None:
        """Generate and save answers for prior search results."""

        self.service.answer_dataset(
            student_search_results_path, save_directory
        )


VALID_COMMANDS = {
    "answer",
    "answer_dataset",
    "index",
    "search",
    "search_dataset",
}


def main() -> None:
    """Validate the command name and start the Fire CLI."""

    if len(sys.argv) > 1 and sys.argv[1] not in VALID_COMMANDS:
        print(f"Error: Invalid command: {sys.argv[1]}")
        sys.exit(1)

    try:
        fire.Fire(CLI())
    except RAGError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
