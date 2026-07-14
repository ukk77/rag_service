"""RAG ingestion layer — verbalizes data from all 4 Tier-1 sources into ChromaDB.

Sources (Tier 1):
  1. sentiment_history.db  — per-ticker sentiment snapshots
  2. risk_history.db       — per-ticker risk snapshots
  3. harness_trades.db     — executed trade log
  4. market_data/*.parquet — OHLCV (daily) with computed indicators

Each document is verbalized as a human-readable string, embedded, and stored with
structured metadata (source, ticker, date) for filtered retrieval.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import pandas as pd

from .config import Settings, get_settings
from .models.schemas import IngestResult
from .watermark import WatermarkStore

log = logging.getLogger(__name__)


# ── ChromaDB + embedding singleton ───────────────────────────────────────────

_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[Any] = None
_embedding_fn: Optional[Any] = None


def _get_embedding_fn(settings: Settings):
    global _embedding_fn
    if _embedding_fn is None:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        _embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
    return _embedding_fn


def get_collection(settings: Optional[Settings] = None) -> Any:
    global _chroma_client, _collection
    if settings is None:
        settings = get_settings()
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = _chroma_client.get_or_create_collection(
            name=settings.chroma_collection,
            embedding_function=_get_embedding_fn(settings),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Verbalization helpers ─────────────────────────────────────────────────────

def _verbalize_sentiment(row: sqlite3.Row) -> str:
    ticker = row["ticker"]
    captured = (row["captured_at"] or "")[:10]
    overall = row["overall_sentiment"] if row["overall_sentiment"] else "neutral"
    avg_score = row["avg_sentiment"] if row["avg_sentiment"] is not None else 0.0
    confidence = row["confidence"] if row["confidence"] is not None else 0.0
    article_count = row["total_articles"] if "total_articles" in row.keys() and row["total_articles"] is not None else "?"
    keys = row.keys()
    extra_parts = []
    if "contrarian_signal" in keys and row["contrarian_signal"]:
        extra_parts.append(f"contrarian={row['contrarian_signal']}")
    if "percentile_vs_sector" in keys and row["percentile_vs_sector"] is not None:
        extra_parts.append(f"pct_vs_sector={row['percentile_vs_sector']:.0f}")
    extra_str = f" ({', '.join(extra_parts)})" if extra_parts else ""
    return (
        f"[SENTIMENT] {ticker} on {captured}: overall sentiment {overall.upper()} "
        f"(avg_score={avg_score:+.3f}, confidence={confidence:.2f}, articles={article_count}).{extra_str}"
    )


def _verbalize_risk(row: sqlite3.Row) -> str:
    ticker = row["ticker"]
    as_of = (row["as_of"] or row["captured_at"] or "")[:10]
    keys = row.keys()
    score = row["composite_risk_score"] if "composite_risk_score" in keys and row["composite_risk_score"] is not None else "?"
    bucket = row["risk_bucket"] if "risk_bucket" in keys and row["risk_bucket"] else ""
    var95 = row["var_95_hist_1d"] if "var_95_hist_1d" in keys and row["var_95_hist_1d"] is not None else None
    beta = row["beta"] if "beta" in keys and row["beta"] is not None else None
    kelly = row["kelly_fraction_capped"] if "kelly_fraction_capped" in keys and row["kelly_fraction_capped"] is not None else None
    stop_pct = row["suggested_stop_loss_pct"] if "suggested_stop_loss_pct" in keys and row["suggested_stop_loss_pct"] is not None else None
    sharpe = row["sharpe"] if "sharpe" in keys and row["sharpe"] is not None else None
    parts = [f"[RISK] {ticker} on {as_of}: composite_score={score}"]
    if bucket:
        parts.append(f"({bucket} risk)")
    if var95 is not None:
        parts.append(f"VaR-95={var95:.1%}")
    if beta is not None:
        parts.append(f"beta={beta:.2f}")
    if sharpe is not None:
        parts.append(f"Sharpe={sharpe:.2f}")
    if kelly is not None:
        parts.append(f"Kelly={kelly:.3f}")
    if stop_pct is not None:
        parts.append(f"stop-loss={stop_pct:.1%}")
    return ". ".join(parts) + "."


def _verbalize_trade(row: sqlite3.Row) -> str:
    ticker = row["ticker"]
    action = row["action"]
    shares = row["shares"]
    price = row["price"]
    strategy = row["strategy"]
    executed_at = (row["executed_at"] or "")[:10]
    pnl = row["realized_pnl"] if row["realized_pnl"] is not None else 0.0
    pnl_str = f", realized_pnl=${pnl:+,.2f}" if pnl != 0 else ""
    return (
        f"[TRADE] {executed_at}: harness/{strategy} {action} "
        f"{shares:.2f} shares of {ticker} at ${price:.2f}{pnl_str}."
    )


def _verbalize_ohlcv(ticker: str, date: str, row: pd.Series) -> str:
    o = row.get("open", float("nan"))
    h = row.get("high", float("nan"))
    l = row.get("low", float("nan"))
    c = row.get("close", float("nan"))
    v = row.get("volume", float("nan"))
    vwap = row.get("vwap", None)
    parts = [f"[MARKET] {ticker} on {date}: O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f}"]
    if not (math.isnan(v) if isinstance(v, float) else False):
        parts.append(f"V={v/1e6:.1f}M")
    if vwap and not (math.isnan(float(vwap)) if vwap is not None else False):
        parts.append(f"VWAP={float(vwap):.2f}")
    return " ".join(parts) + "."


def _verbalize_article(row: sqlite3.Row) -> str:
    """Verbalize a single article from article_sentiments table."""
    ticker = row["ticker"]
    published = (row["published_at"] or "")[:10]
    event_type = row["event_type"] or "other"
    source = row["source"] or "unknown"
    title = row["title"] or ""
    summary = (row["summary"] or "")[:200]
    sentiment = row["sentiment"] or "neutral"
    score = row["score"] if row["score"] is not None else 0.0
    confidence = row["confidence"] if row["confidence"] is not None else 0.0
    impact = row["impact_score"] if row["impact_score"] is not None else 0.0
    text_parts = [
        f"[ARTICLE] {published} | {ticker} | {event_type} | {source}",
        f"{title}.",
    ]
    if summary:
        text_parts.append(summary)
    text_parts.append(f"sentiment={sentiment} score={score:+.2f} confidence={confidence:.2f} impact={impact:.2f}")
    return " ".join(text_parts)


# ── Ingestion per source ──────────────────────────────────────────────────────

def _db_conn(path: str) -> Optional[sqlite3.Connection]:
    p = Path(path)
    if not p.exists():
        log.warning("DB not found: %s", path)
        return None
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    return conn


def ingest_sentiment(
    collection: Any,
    watermark: WatermarkStore,
    settings: Settings,
) -> IngestResult:
    conn = _db_conn(settings.sentiment_db)
    if conn is None:
        return IngestResult(source="sentiment", docs_added=0, docs_skipped=0)

    last_wm = watermark.get_global("sentiment") or "1970-01-01T00:00:00Z"
    try:
        rows = conn.execute(
            "SELECT * FROM sentiment_snapshots WHERE captured_at > ? ORDER BY captured_at",
            (last_wm,),
        ).fetchall()
    except Exception as e:
        log.warning("sentiment_snapshots query failed: %s", e)
        rows = []
    finally:
        conn.close()

    if not rows:
        return IngestResult(source="sentiment", docs_added=0, docs_skipped=0, watermark_advanced_to=last_wm)

    docs, ids, metas = [], [], []
    new_wm = last_wm
    for row in rows:
        try:
            doc_id = f"sentiment_{row['ticker']}_{row['captured_at']}"
            text = _verbalize_sentiment(row)
            date = (row["captured_at"] or "")[:10]
            docs.append(text)
            ids.append(doc_id)
            metas.append({"source": "sentiment", "ticker": row["ticker"], "date": date})
            if row["captured_at"] > new_wm:
                new_wm = row["captured_at"]
        except Exception as e:
            log.debug("sentiment row skip: %s", e)

    _batch_upsert(collection, docs, ids, metas)
    watermark.set_global("sentiment", new_wm)
    return IngestResult(source="sentiment", docs_added=len(docs), docs_skipped=0, watermark_advanced_to=new_wm)


def ingest_risk(
    collection: Any,
    watermark: WatermarkStore,
    settings: Settings,
) -> IngestResult:
    conn = _db_conn(settings.risk_db)
    if conn is None:
        return IngestResult(source="risk", docs_added=0, docs_skipped=0)

    last_wm = watermark.get_global("risk") or "1970-01-01T00:00:00Z"
    try:
        rows = conn.execute(
            "SELECT * FROM risk_snapshots WHERE captured_at > ? ORDER BY captured_at",
            (last_wm,),
        ).fetchall()
    except Exception as e:
        log.warning("risk_snapshots query failed: %s", e)
        rows = []
    finally:
        conn.close()

    if not rows:
        return IngestResult(source="risk", docs_added=0, docs_skipped=0, watermark_advanced_to=last_wm)

    docs, ids, metas = [], [], []
    new_wm = last_wm
    for row in rows:
        try:
            ts_field = row["as_of"] or row["captured_at"]
            doc_id = f"risk_{row['ticker']}_{ts_field}"
            text = _verbalize_risk(row)
            date = (ts_field or "")[:10]
            docs.append(text)
            ids.append(doc_id)
            metas.append({"source": "risk", "ticker": row["ticker"], "date": date})
            ts_val = row["as_of"] or row["captured_at"]
            if ts_val and ts_val > new_wm:
                new_wm = ts_val
        except Exception as e:
            log.debug("risk row skip: %s", e)

    _batch_upsert(collection, docs, ids, metas)
    watermark.set_global("risk", new_wm)
    return IngestResult(source="risk", docs_added=len(docs), docs_skipped=0, watermark_advanced_to=new_wm)


def ingest_trades(
    collection: Any,
    watermark: WatermarkStore,
    settings: Settings,
) -> IngestResult:
    conn = _db_conn(settings.harness_db)
    if conn is None:
        return IngestResult(source="trades", docs_added=0, docs_skipped=0)

    last_wm = watermark.get_global("trades") or "1970-01-01T00:00:00Z"
    try:
        rows = conn.execute(
            "SELECT * FROM trades WHERE executed_at > ? ORDER BY executed_at",
            (last_wm,),
        ).fetchall()
    except Exception as e:
        log.warning("trades query failed: %s", e)
        rows = []
    finally:
        conn.close()

    if not rows:
        return IngestResult(source="trades", docs_added=0, docs_skipped=0, watermark_advanced_to=last_wm)

    docs, ids, metas = [], [], []
    new_wm = last_wm
    for row in rows:
        try:
            doc_id = f"trade_{row['id']}"
            text = _verbalize_trade(row)
            date = (row["executed_at"] or "")[:10]
            docs.append(text)
            ids.append(doc_id)
            metas.append({"source": "trade", "ticker": row["ticker"], "date": date})
            if row["executed_at"] > new_wm:
                new_wm = row["executed_at"]
        except Exception as e:
            log.debug("trade row skip: %s", e)

    _batch_upsert(collection, docs, ids, metas)
    watermark.set_global("trades", new_wm)
    return IngestResult(source="trades", docs_added=len(docs), docs_skipped=0, watermark_advanced_to=new_wm)


def ingest_market_data(
    collection: Any,
    watermark: WatermarkStore,
    settings: Settings,
) -> IngestResult:
    market_dir = Path(settings.market_data_dir)
    if not market_dir.exists():
        return IngestResult(source="market_data", docs_added=0, docs_skipped=0)

    added = 0
    skipped = 0
    parquet_files = list(market_dir.glob("*.parquet"))

    for parquet_path in parquet_files:
        ticker = parquet_path.stem.upper()
        last_wm = watermark.get("market_data", ticker) or "1970-01-01"
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            log.debug("parquet read failed %s: %s", parquet_path, e)
            continue

        if df.empty:
            continue

        # Normalize index to date strings
        if hasattr(df.index, "date"):
            df.index = df.index.strftime("%Y-%m-%d")
        elif not isinstance(df.index[0], str):
            df.index = df.index.astype(str).str[:10]

        # Filter to new rows only (after watermark)
        new_rows = df[df.index > last_wm]

        # Cold-start: limit to recent N days on first ingest
        if last_wm == "1970-01-01" and len(new_rows) > settings.ohlcv_cold_start_days:
            new_rows = new_rows.iloc[-settings.ohlcv_cold_start_days:]

        if new_rows.empty:
            skipped += 1
            continue

        docs, ids, metas = [], [], []
        new_wm = last_wm
        for date, row in new_rows.iterrows():
            try:
                doc_id = f"ohlcv_{ticker}_{date}"
                text = _verbalize_ohlcv(ticker, str(date), row)
                docs.append(text)
                ids.append(doc_id)
                metas.append({"source": "market_data", "ticker": ticker, "date": str(date)})
                if str(date) > new_wm:
                    new_wm = str(date)
            except Exception as e:
                log.debug("ohlcv row skip %s %s: %s", ticker, date, e)

        if docs:
            _batch_upsert(collection, docs, ids, metas)
            watermark.set("market_data", ticker, new_wm)
            added += len(docs)

    return IngestResult(source="market_data", docs_added=added, docs_skipped=skipped)


def ingest_articles(
    collection: Any,
    watermark: WatermarkStore,
    settings: Settings,
) -> IngestResult:
    """Tier 2: ingest per-article sentiment data from article_sentiments table."""
    conn = _db_conn(settings.sentiment_db)
    if conn is None:
        return IngestResult(source="articles", docs_added=0, docs_skipped=0)

    last_wm = watermark.get_global("articles") or "1970-01-01T00:00:00Z"
    try:
        rows = conn.execute(
            "SELECT * FROM article_sentiments WHERE captured_at > ? ORDER BY captured_at",
            (last_wm,),
        ).fetchall()
    except Exception as e:
        log.warning("article_sentiments query failed: %s", e)
        rows = []
    finally:
        conn.close()

    if not rows:
        return IngestResult(source="articles", docs_added=0, docs_skipped=0, watermark_advanced_to=last_wm)

    docs, ids, metas = [], [], []
    new_wm = last_wm
    for row in rows:
        try:
            # Dedup key matches DB constraint: (ticker, url_hash)
            url_hash = row["url_hash"] if "url_hash" in row.keys() else str(hash(row["title"]))
            doc_id = f"article_{row['ticker']}_{url_hash}"
            text = _verbalize_article(row)
            date = (row["published_at"] or row["captured_at"] or "")[:10]
            docs.append(text)
            ids.append(doc_id)
            metas.append({
                "source": "article",
                "ticker": row["ticker"],
                "date": date,
                "event_type": row["event_type"] or "other",
                "sentiment": row["sentiment"] or "neutral",
            })
            if row["captured_at"] and row["captured_at"] > new_wm:
                new_wm = row["captured_at"]
        except Exception as e:
            log.debug("article row skip: %s", e)

    _batch_upsert(collection, docs, ids, metas)
    watermark.set_global("articles", new_wm)
    return IngestResult(source="articles", docs_added=len(docs), docs_skipped=0, watermark_advanced_to=new_wm)


def _dedup_by_id(
    docs: List[str], ids: List[str], metas: List[dict]
) -> Tuple[List[str], List[str], List[dict]]:
    """Keep last occurrence of each id (rows ordered ASC so last = most recent)."""
    seen: Dict[str, int] = {}
    for i, doc_id in enumerate(ids):
        seen[doc_id] = i
    order = sorted(seen.values())
    return [docs[i] for i in order], [ids[i] for i in order], [metas[i] for i in order]


def _batch_upsert(collection: Any, docs: List[str], ids: List[str], metas: List[dict], batch_size: int = 100) -> None:
    """Upsert documents in batches, deduplicating IDs to avoid ChromaDB DuplicateIDError."""
    docs, ids, metas = _dedup_by_id(docs, ids, metas)
    for i in range(0, len(docs), batch_size):
        try:
            collection.upsert(
                documents=docs[i:i + batch_size],
                ids=ids[i:i + batch_size],
                metadatas=metas[i:i + batch_size],
            )
        except Exception as e:
            log.warning("chroma upsert batch [%d:%d] failed: %s", i, i + batch_size, e)


# ── Main ingestion orchestrator ───────────────────────────────────────────────

_SOURCE_MAP = {
    "sentiment": ingest_sentiment,
    "risk": ingest_risk,
    "trades": ingest_trades,
    "market_data": ingest_market_data,
    "articles": ingest_articles,
}


def run_ingestion(
    sources: List[str],
    settings: Optional[Settings] = None,
) -> List[IngestResult]:
    if settings is None:
        settings = get_settings()
    collection = get_collection(settings)
    watermark = WatermarkStore(settings.watermark_file)

    results = []
    for source in sources:
        fn = _SOURCE_MAP.get(source)
        if fn is None:
            log.warning("Unknown ingest source: %s", source)
            continue
        try:
            result = fn(collection, watermark, settings)
            log.info("ingest[%s]: +%d docs (wm=%s)", source, result.docs_added, result.watermark_advanced_to)
            results.append(result)
        except Exception as e:
            log.error("ingest[%s] failed: %s", source, e)
            results.append(IngestResult(source=source, docs_added=0, docs_skipped=0))

    return results
