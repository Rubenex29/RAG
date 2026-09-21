# Standard library
import uuid
from typing import List

# Third-party dependencies
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Identify one retrieved location within a source file."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a question that has not yet been answered."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question together with its answer and source locations."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Contain the questions used by a RAG dataset."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Store the sources retrieved for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Store a generated answer alongside its retrieved sources."""

    answer: str


class StudentSearchResults(BaseModel):
    """Contain search results produced for an entire dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Contain answered search results produced for an entire dataset."""

    search_results: List[MinimalAnswer]
    k: int
