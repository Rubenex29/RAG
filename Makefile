ifneq ($(wildcard /sgoinfre),)
SGOINFRE_USER := $(shell whoami)
export HF_HOME := /sgoinfre/$(SGOINFRE_USER)/hf_cache
export UV_CACHE_DIR := /sgoinfre/$(SGOINFRE_USER)/uv_cache
endif

.PHONY: install run debug clean clean-venv lint lint-strict

install:
	uv sync

# Available commands:
# make run ARGS='index --max_chunk_size 2000'
# make run ARGS='search "what is a RAG?" --k 10'
# make run ARGS='search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions'
# make run ARGS='answer "what is a RAG?" --k 10'
# make run ARGS='answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions'
run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	rm -rf .mypy_cache .pytest_cache .ruff_cache

clean-venv:
	rm -rf .venv

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src
	uv run mypy src --strict
