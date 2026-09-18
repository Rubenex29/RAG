from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .RAG import RAG

app = FastAPI(title="RAG API")
rag = RAG()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, gt=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_class=PlainTextResponse)
def search(payload: QueryRequest):
    try:
        results = rag.service.search(payload.query, payload.k)
        print_lines = []
        for idx, entry in enumerate(results):
            print_lines.extend(
                [
                    f"Result {idx + 1}:",
                    f"File Path: {entry.file_path}",
                    f"First Character Index: {entry.first_character_index}",
                    f"Last Character Index: {entry.last_character_index}",
                    "-" * 40,
                ]
            )
        return "\n".join(print_lines)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/answer", response_class=PlainTextResponse)
def answer(payload: QueryRequest):
    try:
        answer = rag.service.answer(payload.query, payload.k)
        return answer
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
