# Standard library
import ast
import json
from pathlib import Path
from typing import Any, Dict, List

# Application modules
from .storage import hash_file
from .errors import RAGError


class Chunker:
    """Find changed files and split their contents into searchable chunks."""

    def __init__(self, project_root: Path, repo: Path):
        """Store the project root and repository to be indexed."""

        self.project_root = project_root
        self.repo = repo

    def process_files(self, chunk_path: Path) -> List[Path]:
        """Update manifests and return files that need reindexing."""

        files = []
        manifest_path = chunk_path.with_name("manifest.json")
        old_manifest = {}
        if manifest_path.exists():
            old_manifest = self._load_json(manifest_path)
            if not isinstance(old_manifest, dict):
                raise RAGError(
                    f"Manifest '{manifest_path}' has an invalid format. "
                    "Delete data/processed and run the 'index' command again."
                )
        elif chunk_path.exists():
            old_chunks = self._load_json(chunk_path)
            if not isinstance(old_chunks, list):
                raise RAGError(
                    f"Chunks file '{chunk_path}' has an invalid format. "
                    "Delete data/processed and run the 'index' command again."
                )
            if old_chunks and all("hash" in chunk for chunk in old_chunks):
                try:
                    old_manifest = {
                        chunk["metadata"]["file_path"]: chunk["hash"]
                        for chunk in old_chunks
                    }
                except (KeyError, TypeError) as exc:
                    raise RAGError(
                        f"Chunks file '{chunk_path}' has an invalid format. "
                        "Delete data/processed and run the 'index' command "
                        "again."
                    ) from exc
            else:
                old_manifest = {}

        try:
            current_files = [
                file for file in self.repo.rglob("*")
                if file.is_file()
                and file.suffix in {".md", ".rst", ".txt", ".py"}
            ]
            current_manifest = {
                str(file.relative_to(self.project_root)): hash_file(file)
                for file in current_files
            }
        except OSError as exc:
            raise RAGError(
                f"Could not read source repository '{self.repo}': {exc}"
            ) from exc

        if chunk_path.exists():
            chunks = self._load_json(chunk_path)
            if not isinstance(chunks, list):
                raise RAGError(
                    f"Chunks file '{chunk_path}' has an invalid format. "
                    "Delete data/processed and run the 'index' command again."
                )
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
        try:
            with open(chunk_path, "w") as f:
                json.dump(chunks, f, indent=2)
            with open(manifest_path, "w") as f:
                json.dump(current_manifest, f, indent=2)
        except OSError as exc:
            raise RAGError(
                f"Could not write processed index metadata: {exc}"
            ) from exc
        return files

    @staticmethod
    def _load_json(path: Path) -> Any:
        """Load a JSON file and report a concise user-facing error."""

        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            raise RAGError(
                f"JSON file '{path}' is invalid. Delete data/processed and "
                "run the 'index' command again."
            ) from exc
        except OSError as exc:
            raise RAGError(f"Could not read file '{path}': {exc}") from exc

    def recursive_chunking(self, text: str, file: Path,
                           chunk_size: int = 2000) -> List[Dict[str, Any]]:
        """Split plain text into overlapping chunks with source metadata."""

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
        """Split Python functions into chunks while retaining class context."""

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
            """Convert an AST line and byte offset to a character offset."""

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
        """Chunk a supported text or Python source file."""

        try:
            if file.suffix in {".md", ".rst", ".txt"}:
                text = file.read_text(encoding="utf-8", errors="ignore")
                return self.recursive_chunking(text, file, max_chunk_size)

            if file.suffix == ".py":
                text = file.read_text(encoding="utf-8", errors="ignore")
                return self.python_chunking(text, file, max_chunk_size)
        except SyntaxError as exc:
            raise RAGError(
                f"Could not parse Python file '{file}' at line "
                f"{exc.lineno}: {exc.msg}"
            ) from exc
        except OSError as exc:
            raise RAGError(f"Could not read source file '{file}': {exc}") \
                from exc

        return []
