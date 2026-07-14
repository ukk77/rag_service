"""RAG service configuration — all values overridable via environment variables."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings

_TRADING_ROOT = Path(__file__).resolve().parents[2]
_RAG_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8200

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: str = str(_RAG_ROOT / "chromadb_data")
    chroma_collection: str = "trading_platform"

    # ── Embedding model ───────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── LLM synthesis (same provider pattern as harness A2) ──────────────────
    llm_provider: str = os.environ.get("RAG_LLM_PROVIDER", "ollama")
    llm_model: str = os.environ.get("LLM_MODEL", "llama3.2:3b")
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "http://localhost:11434")
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")

    # ── Data source paths ─────────────────────────────────────────────────────
    sentiment_db: str = str(_TRADING_ROOT / "sentiment_analysis" / "backend" / "sentiment_history.db")
    risk_db: str = str(_TRADING_ROOT / "risk_calculator" / "backend" / "risk_history.db")
    harness_db: str = str(_TRADING_ROOT / "harness" / "harness_trades.db")
    market_data_dir: str = str(_TRADING_ROOT / "market_data")

    # ── Watermark file ────────────────────────────────────────────────────────
    watermark_file: str = str(_RAG_ROOT / "rag_ingestion_watermark.json")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 8
    context_window_days: int = 30          # default lookback for context queries

    # ── Cold-start OHLCV limit ────────────────────────────────────────────────
    ohlcv_cold_start_days: int = 90        # limit on first ingest to avoid timeout

    model_config = {"env_file": str(_TRADING_ROOT / ".env"), "extra": "ignore"}


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
