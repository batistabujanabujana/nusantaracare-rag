from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import AskRequest, AskResponse
from app.services.agent import AnswerGenerationUnavailable, generate_answer
from app.services.rag import KnowledgeBaseMissing, Retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.retriever = Retriever()
        app.state.kb_error = None
    except KnowledgeBaseMissing as exc:
        app.state.retriever = None
        app.state.kb_error = str(exc)
    yield


app = FastAPI(title="NusantaraCare Internal Assistant", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request_handler(_request, _exc):
    return JSONResponse(
        status_code=422,
        content={
            "answer": "Format permintaan tidak valid. Kirim JSON dengan field question berupa teks 3 sampai 2000 karakter.",
            "confidence_label": "low",
            "reason_code": "invalid_request",
        },
    )


@app.get("/health")
def health():
    retriever = app.state.retriever
    ready = retriever is not None
    genai_ready = bool(ready and retriever.client and retriever.embeddings is not None)
    return {
        "status": "ok" if ready and genai_ready else "degraded",
        "knowledge_base_loaded": ready,
        "semantic_search_enabled": genai_ready,
        "llm_enabled": genai_ready,
        # Safe diagnostic only: exception class/status, never request details
        # or credential values.
        "genai_initialization_error": (
            retriever.embedding_error if ready and not genai_ready else None
        ),
    }


@app.post("/ask", response_model=AskResponse, responses={422: {"model": AskResponse}})
def ask(payload: AskRequest):
    if app.state.retriever is None:
        return JSONResponse(
            status_code=503,
            content={
                "answer": "Knowledge base belum tersedia; periksa pemasangan dokumen sumber.",
                "confidence_label": "low",
                "reason_code": "service_unavailable",
            },
        )
    try:
        return generate_answer(app.state.retriever, payload.question)
    except AnswerGenerationUnavailable:
        return JSONResponse(
            status_code=503,
            content={
                "answer": "Layanan model sedang tidak tersedia. Coba lagi nanti.",
                "confidence_label": "low",
                "reason_code": "service_unavailable",
            },
        )
