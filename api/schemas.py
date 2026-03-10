from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class IngestRequest(BaseModel):
    doc_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None


class JobStatusResponse(BaseModel):
    job_id: str
    kind: str
    status: str
    queue: str
    created_at: str
    updated_at: str
    error: Optional[str] = None


class IngestResponse(JobStatusResponse):
    pass


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=50)


class Citation(BaseModel):
    doc_id: str
    chunk_id: Union[str, int]
    score: float
    text_snippet: str


class SearchResponse(BaseModel):
    results: List[Citation]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=50)


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]


class CrawlRequest(BaseModel):
    url: str = Field(..., min_length=1)
    max_depth: int = Field(5, ge=1, le=10)
    extract_depth: str = Field("advanced")


class CrawlResponse(JobStatusResponse):
    pass
