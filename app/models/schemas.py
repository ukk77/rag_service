"""Pydantic schemas for the RAG service API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class IngestRequest(BaseModel):
    sources: List[str] = ["sentiment", "risk", "trades", "market_data", "articles"]
    incremental: bool = True


class IngestResult(BaseModel):
    source: str
    docs_added: int
    docs_skipped: int
    watermark_advanced_to: Optional[str] = None


class IngestResponse(BaseModel):
    status: str
    results: List[IngestResult]
    total_docs_added: int
    duration_seconds: float


class AskRequest(BaseModel):
    query: str
    ticker: Optional[str] = None
    date_from: Optional[str] = None   # YYYY-MM-DD
    date_to: Optional[str] = None     # YYYY-MM-DD
    top_k: Optional[int] = None


class SourceChunk(BaseModel):
    doc_id: str
    text: str
    source: str
    ticker: Optional[str]
    date: Optional[str]
    score: float


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceChunk]
    model_used: str


class SummarizeRequest(BaseModel):
    run_report: Dict[str, Any]


class SummarizeResponse(BaseModel):
    narrative: str
    sources: List[SourceChunk]
    model_used: str


class ContextResponse(BaseModel):
    ticker: str
    chunks: List[SourceChunk]
    summary: Optional[str] = None


class HealthResponse(BaseModel):
    status: str                        # "healthy" | "degraded" | "unhealthy"
    chroma_ok: bool
    embedding_ok: bool
    llm_ok: bool
    doc_count: int
    timestamp: datetime = datetime.utcnow()
    detail: Optional[str] = None
