"""RAG Market Intelligence Service — FastAPI app on port 8200.

Endpoints:
  GET  /api/health           — ChromaDB + embedding + LLM status
  POST /api/ingest           — incremental ingest from all 4 Tier-1 sources
  POST /api/ask              — free-text query → grounded answer
  POST /api/summarize        — run-report dict → RAG-grounded narrative
  GET  /api/context/{ticker} — recent context chunks for a ticker
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .config import get_settings
from .ingestor import get_collection, run_ingestion
from .models.schemas import (
    AskRequest, AskResponse,
    ContextResponse,
    HealthResponse,
    IngestRequest, IngestResponse,
    SummarizeRequest, SummarizeResponse,
)
from .retriever import retrieve
from .synthesizer import synthesize_answer, synthesize_run_summary

log = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Market Intelligence Service",
    version="1.0.0",
    description="Retrieval-Augmented Generation layer for the trading platform.",
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    chroma_ok = False
    doc_count = 0
    embedding_ok = False
    llm_ok = False
    detail = None

    try:
        col = get_collection(settings)
        doc_count = col.count()
        chroma_ok = True
        embedding_ok = True
    except Exception as e:
        detail = f"ChromaDB/embedding error: {e}"

    try:
        if settings.llm_provider == "openai":
            import openai  # noqa: F401
            llm_ok = bool(settings.openai_api_key)
        else:
            import requests
            r = requests.get(f"{settings.llm_base_url}/api/tags", timeout=5)
            llm_ok = r.status_code == 200
    except Exception:
        llm_ok = False

    if chroma_ok and embedding_ok:
        status = "healthy" if llm_ok else "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        chroma_ok=chroma_ok,
        embedding_ok=embedding_ok,
        llm_ok=llm_ok,
        doc_count=doc_count,
        detail=detail,
    )


# ── Ingest ────────────────────────────────────────────────────────────────────

@app.post("/api/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    settings = get_settings()
    t0 = time.time()
    results = run_ingestion(sources=req.sources, settings=settings)
    total = sum(r.docs_added for r in results)
    return IngestResponse(
        status="ok",
        results=results,
        total_docs_added=total,
        duration_seconds=round(time.time() - t0, 2),
    )


# ── Ask ───────────────────────────────────────────────────────────────────────

@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    settings = get_settings()
    chunks = retrieve(
        query=req.query,
        ticker=req.ticker,
        date_from=req.date_from,
        date_to=req.date_to,
        top_k=req.top_k,
        settings=settings,
    )
    answer, model_used = synthesize_answer(req.query, chunks, settings)
    return AskResponse(query=req.query, answer=answer, sources=chunks, model_used=model_used)


# ── Summarize (operator oversight, called by harness reporter) ────────────────

@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    settings = get_settings()
    run_report = req.run_report

    # Retrieve context relevant to the run — top tickers by activity
    tickers_in_run = list(set(
        run_report.get("top_buys", [])[:3] + run_report.get("top_sells", [])[:3]
    ))
    all_chunks = []
    for ticker in tickers_in_run:
        chunks = retrieve(
            query=f"recent sentiment risk activity for {ticker}",
            ticker=ticker,
            top_k=3,
            settings=settings,
        )
        all_chunks.extend(chunks)

    # Deduplicate and rank
    seen: set[str] = set()
    deduped = []
    for c in sorted(all_chunks, key=lambda x: x.score, reverse=True):
        if c.doc_id not in seen:
            seen.add(c.doc_id)
            deduped.append(c)

    narrative, model_used = synthesize_run_summary(run_report, deduped[:8], settings)
    return SummarizeResponse(narrative=narrative, sources=deduped[:8], model_used=model_used)


# ── Context for a ticker (signal enrichment) ──────────────────────────────────

@app.get("/api/context/{ticker}", response_model=ContextResponse)
def context(ticker: str, days: int = 7) -> ContextResponse:
    from datetime import timedelta
    settings = get_settings()
    date_from = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    chunks = retrieve(
        query=f"recent sentiment risk trades market data for {ticker.upper()}",
        ticker=ticker.upper(),
        date_from=date_from,
        settings=settings,
    )
    summary = None
    if chunks:
        summary, _ = synthesize_answer(
            f"Summarise recent context for {ticker.upper()} in 2 sentences.",
            chunks,
            settings,
        )
    return ContextResponse(ticker=ticker.upper(), chunks=chunks, summary=summary)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    settings = get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
