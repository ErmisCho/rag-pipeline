import contextvars
import logging
import os
import time
import uuid
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.core import answer_with_docs, run_llm, search_docs
from backend.selection import run_answer_with_selection_and_retry
from backend.ingestion import crawl_and_ingest, ingest_text
from .schemas import (
    AskRequest,
    AskResponse,
    Citation,
    CrawlRequest,
    CrawlResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)

request_id_var = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("api")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s request_id=%(request_id)s %(message)s"
    )
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = setup_logging()
app = FastAPI(title="RAG API", version="0.1.0")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    token = request_id_var.set(request_id)
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["x-request-id"] = request_id
    return response


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@app.post("/ingest", response_model=IngestResponse)
async def ingest(payload: IngestRequest):
    start = time.perf_counter()
    try:
        chunks = await ingest_text(
            doc_id=payload.doc_id,
            text=payload.text,
            metadata=payload.metadata,
        )
    except KeyError as e:
        logger.exception("stage=ingest error=missing_env")
        raise HTTPException(
            status_code=500, detail=f"Missing env var: {e}") from e
    except Exception as e:
        logger.exception("stage=ingest error=internal")
        raise HTTPException(status_code=500, detail="Ingest failed") from e

    latency_ms = int((time.perf_counter() - start) * 1000)
    # Log counts only to avoid large/PII payloads.
    logger.info(
        "stage=ingest latency_ms=%s chunks_count=%s doc_id=%s",
        latency_ms,
        chunks,
        payload.doc_id,
    )
    return IngestResponse(status="ok", chunks=chunks)


@app.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest):
    start = time.perf_counter()
    try:
        results = search_docs(payload.query, top_k=payload.top_k)
    except Exception as e:
        logger.exception("stage=search error=internal")
        raise HTTPException(status_code=500, detail="Search failed") from e

    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info(f"stage=search latency_ms={latency_ms} top_k={payload.top_k}")

    citations: List[Citation] = []
    for i, (doc, score) in enumerate(results):
        metadata = doc.metadata or {}
        doc_id = str(metadata.get("doc_id")
                     or metadata.get("source") or "unknown")
        chunk_id = metadata.get("chunk_id", i)
        citations.append(
            Citation(
                doc_id=doc_id,
                chunk_id=chunk_id,
                score=float(score),
                text_snippet=doc.page_content[:200],
            )
        )
    return SearchResponse(results=citations)


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest):
    search_start = time.perf_counter()
    try:
        results = search_docs(payload.query, top_k=payload.top_k)
    except Exception as e:
        logger.exception("stage=search error=internal")
        raise HTTPException(status_code=500, detail="Search failed") from e
    search_latency_ms = int((time.perf_counter() - search_start) * 1000)
    logger.info(
        f"stage=search latency_ms={search_latency_ms} top_k={payload.top_k}")
    if not results:
        logger.info(
            "stage=guardrail triggered=true reason=empty_results top_score=0 sources=0")
        return AskResponse(
            answer="Not enough evidence in the provided documents.",
            citations=[],
        )

    llm_start = time.perf_counter()
    try:
        note_threshold = float(os.environ.get("NOTE_SCORE_THRESHOLD", "0.6"))
        note_margin = float(os.environ.get("NOTE_SCORE_MARGIN", "0.05"))
        llm_result, docs_for_answer, results = run_answer_with_selection_and_retry(
            query=payload.query,
            results=results,
            note_threshold=note_threshold,
            note_margin=note_margin,
            max_docs=4,
            answer_fn=lambda docs: answer_with_docs(
                payload.query,
                documents=docs,
                chat_history=[],
            ),
            retry_search_fn=search_docs,
            retry_top_ks=[
                max(payload.top_k * 2, 20),
                max(payload.top_k * 3, 30),
            ],
            retry_max_docs=6,
            debug_log=logger.debug,
        )
    except KeyError as e:
        logger.exception("stage=llm error=missing_env")
        raise HTTPException(
            status_code=500, detail=f"Missing env var: {e}") from e
    except Exception as e:
        logger.exception("stage=llm error=internal")
        raise HTTPException(status_code=500, detail="LLM failed") from e

    llm_latency_ms = int((time.perf_counter() - llm_start) * 1000)
    logger.info(f"stage=llm latency_ms={llm_latency_ms}")

    docs_for_answer = llm_result.get("source_documents", docs_for_answer)
    allowed_ids = {id(doc) for doc in docs_for_answer}
    score_by_id = {id(doc): float(score) for doc, score in results}
    citations: List[Citation] = []
    for i, (doc, score) in enumerate(results):
        if id(doc) not in allowed_ids:
            continue
        metadata = doc.metadata or {}
        doc_id = str(metadata.get("doc_id")
                     or metadata.get("source") or "unknown")
        chunk_id = metadata.get("chunk_id", i)
        citations.append(
            Citation(
                doc_id=doc_id,
                chunk_id=chunk_id,
                score=float(score),
                text_snippet=doc.page_content[:200],
            )
        )

    note_docs_used = [
        doc for doc in docs_for_answer if (doc.metadata or {}).get("doc_id")
    ]
    if note_docs_used and not citations:
        doc = note_docs_used[0]
        metadata = doc.metadata or {}
        doc_id = str(metadata.get("doc_id")
                     or metadata.get("source") or "unknown")
        chunk_id = metadata.get("chunk_id", 0)
        score = score_by_id.get(id(doc), 1.0)
        citations.append(
            Citation(
                doc_id=doc_id,
                chunk_id=chunk_id,
                score=float(score),
                text_snippet=doc.page_content[:200],
            )
        )

    min_sources = int(os.environ.get("MIN_SOURCES", "1"))
    min_top_score = float(os.environ.get("MIN_TOP_SCORE", "0.3"))
    top_score = max((score for _, score in results), default=0.0)
    effective_min_sources = 1 if note_docs_used else min_sources
    if top_score < min_top_score or len(citations) < effective_min_sources:
        reason = "low_score" if top_score < min_top_score else "insufficient_sources"
        logger.info(
            "stage=guardrail triggered=true reason=%s top_score=%s sources=%s",
            reason,
            f"{top_score:.4f}",
            len(citations),
        )
        return AskResponse(
            answer="Not enough evidence in the provided documents.",
            citations=[],
        )

    answer = llm_result.get("result", "")
    return AskResponse(answer=answer, citations=citations)


@app.post("/crawl", response_model=CrawlResponse)
async def crawl(payload: CrawlRequest):
    start = time.perf_counter()
    try:
        stats = await crawl_and_ingest(
            url=payload.url,
            max_depth=payload.max_depth,
            extract_depth=payload.extract_depth,
        )
    except Exception as e:
        logger.exception("stage=crawl error=internal")
        raise HTTPException(status_code=500, detail="Crawl failed") from e

    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "stage=crawl latency_ms=%s documents=%s chunks=%s",
        latency_ms,
        stats["documents"],
        stats["chunks"],
    )
    return CrawlResponse(
        status="ok",
        documents=stats["documents"],
        chunks=stats["chunks"],
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
