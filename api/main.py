import contextvars
import logging
import os
import re
import time
import uuid
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.core import answer_with_docs, run_llm, search_docs
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
    logger.setLevel(logging.INFO)
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
        raise HTTPException(status_code=500, detail=f"Missing env var: {e}") from e
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
        doc_id = str(metadata.get("doc_id") or metadata.get("source") or "unknown")
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
    logger.info(f"stage=search latency_ms={search_latency_ms} top_k={payload.top_k}")

    llm_start = time.perf_counter()
    try:
        query_terms = set(re.findall(r"[a-zA-Z]{3,}", payload.query.lower()))
        reranked_results = []
        for doc, score in results:
            metadata = doc.metadata or {}
            source = str(metadata.get("source") or metadata.get("doc_id") or "")
            source_lower = source.lower()
            content_lower = (doc.page_content or "").lower()
            term_hits = sum(1 for term in query_terms if term in content_lower)
            bonus = 0.02 * term_hits
            if "overview" in source_lower and "langchain" in source_lower:
                bonus += 0.15
            reranked_results.append((doc, score + bonus))
        results = sorted(reranked_results, key=lambda item: item[1], reverse=True)

        note_threshold = float(os.environ.get("NOTE_SCORE_THRESHOLD", "0.6"))
        note_margin = float(os.environ.get("NOTE_SCORE_MARGIN", "0.05"))
        note_docs = []
        best_note_score = None
        best_other_score = None
        for doc, score in results:
            if (doc.metadata or {}).get("doc_id"):
                note_docs.append(doc)
                best_note_score = score if best_note_score is None else max(best_note_score, score)
            else:
                best_other_score = score if best_other_score is None else max(best_other_score, score)
        note_term_match = any(
            term in (doc.page_content or "").lower()
            for doc in note_docs
            for term in query_terms
        ) if query_terms else False
        use_notes_only = (
            best_note_score is not None
            and best_note_score >= note_threshold
            and (best_other_score is None or best_note_score >= best_other_score + note_margin)
            and note_term_match
        )
        docs_for_answer = note_docs if use_notes_only else [doc for doc, _ in results[:4]]
        llm_result = answer_with_docs(
            payload.query, documents=docs_for_answer, chat_history=[]
        )
        answer_text = str(llm_result.get("result", "")).strip()
        answer_lower = answer_text.lower()
        has_term_overlap = any(term in answer_lower for term in query_terms) if query_terms else True
        if (len(answer_text) < 40) or (not has_term_overlap):
            retry_top_k = max(payload.top_k * 2, 20)
            retry_results = search_docs(payload.query, top_k=retry_top_k)
            reranked_retry = []
            for doc, score in retry_results:
                metadata = doc.metadata or {}
                source = str(metadata.get("source") or metadata.get("doc_id") or "")
                source_lower = source.lower()
                content_lower = (doc.page_content or "").lower()
                term_hits = sum(1 for term in query_terms if term in content_lower)
                bonus = 0.02 * term_hits
                if "overview" in source_lower and "langchain" in source_lower:
                    bonus += 0.15
                reranked_retry.append((doc, score + bonus))
            reranked_retry = sorted(reranked_retry, key=lambda item: item[1], reverse=True)
            docs_for_answer = [doc for doc, _ in reranked_retry[:6]]
            llm_result = answer_with_docs(
                payload.query, documents=docs_for_answer, chat_history=[]
            )
    except KeyError as e:
        logger.exception("stage=llm error=missing_env")
        raise HTTPException(status_code=500, detail=f"Missing env var: {e}") from e
    except Exception as e:
        logger.exception("stage=llm error=internal")
        raise HTTPException(status_code=500, detail="LLM failed") from e

    llm_latency_ms = int((time.perf_counter() - llm_start) * 1000)
    logger.info(f"stage=llm latency_ms={llm_latency_ms}")

    docs_for_answer = llm_result.get("source_documents", docs_for_answer)
    allowed_ids = {id(doc) for doc in docs_for_answer}
    citations: List[Citation] = []
    for i, (doc, score) in enumerate(results):
        if id(doc) not in allowed_ids:
            continue
        metadata = doc.metadata or {}
        doc_id = str(metadata.get("doc_id") or metadata.get("source") or "unknown")
        chunk_id = metadata.get("chunk_id", i)
        citations.append(
            Citation(
                doc_id=doc_id,
                chunk_id=chunk_id,
                score=float(score),
                text_snippet=doc.page_content[:200],
            )
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
