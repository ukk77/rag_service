"""ChromaDB retrieval pipeline with multi-query expansion and metadata filtering."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .config import Settings, get_settings
from .ingestor import get_collection
from .models.schemas import SourceChunk

log = logging.getLogger(__name__)


def _build_where(
    ticker: Optional[str] = None,
    source: Optional[str] = None,
) -> Optional[Dict]:
    """Build ChromaDB $where filter dict.

    NOTE: ChromaDB only supports $gte/$lte on int/float, not strings.
    Date filtering is applied post-retrieval in Python (see retrieve()).
    """
    clauses = []
    if ticker:
        clauses.append({"ticker": {"$eq": ticker.upper()}})
    if source:
        clauses.append({"source": {"$eq": source}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _expand_queries(query: str, ticker: Optional[str]) -> List[str]:
    """Generate query variants targeting each source-document format.

    Documents are stored with prefixed formats: [RISK], [SENTIMENT], [TRADE],
    [MARKET], [ARTICLE].  Source-format-aware variants ensure the embedding model
    can match against each doc type regardless of how the user phrases the query.
    """
    variants = [query]
    if ticker:
        variants.append(f"[RISK] {ticker} composite_score risk VaR beta Sharpe Kelly")
        variants.append(f"[SENTIMENT] {ticker} sentiment overall confidence score")
        variants.append(f"[TRADE] {ticker} trade buy sell shares price")
        variants.append(f"[ARTICLE] {ticker} news headline earnings guidance event")
    else:
        variants.append("risk sentiment trade regime market allocation signal")
        variants.append("[ARTICLE] news headline event earnings guidance analyst")
    return variants


def retrieve(
    query: str,
    ticker: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    top_k: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> List[SourceChunk]:
    """Multi-query ChromaDB retrieval with deduplication."""
    if settings is None:
        settings = get_settings()
    k = top_k or settings.retrieval_top_k
    collection = get_collection(settings)
    # Date filters applied post-retrieval (ChromaDB $gte/$lte requires numeric types)
    where = _build_where(ticker=ticker)

    try:
        doc_count = collection.count()
    except Exception:
        doc_count = 0

    if doc_count == 0:
        return []

    # Over-retrieve when date filters are active: ChromaDB returns top-N by
    # similarity, but many top-similarity docs may fall outside the date window
    # and get pruned by the post-retrieval filter, leaving too few results.
    fetch_multiplier = 5 if (date_from or date_to) else 1
    effective_k = min(k * fetch_multiplier, doc_count)

    query_variants = _expand_queries(query, ticker)
    seen_ids: set[str] = set()
    chunks: List[SourceChunk] = []

    for variant in query_variants:
        try:
            query_kwargs: Dict[str, Any] = {
                "query_texts": [variant],
                "n_results": effective_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_kwargs["where"] = where

            results = collection.query(**query_kwargs)
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, distances):
                doc_id = f"{meta.get('source','')}_{meta.get('ticker','')}_{meta.get('date','')}_{hash(doc)}"
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                score = max(0.0, 1.0 - float(dist))
                chunks.append(SourceChunk(
                    doc_id=doc_id,
                    text=doc,
                    source=meta.get("source", "unknown"),
                    ticker=meta.get("ticker"),
                    date=meta.get("date"),
                    score=round(score, 4),
                ))
        except Exception as e:
            log.warning("retrieval variant failed ('%s'): %s", variant, e)

    # Post-retrieval date filter (string ISO date comparison is safe in Python)
    if date_from or date_to:
        filtered = []
        for c in chunks:
            d = c.date or ""
            if date_from and d and d < date_from:
                continue
            if date_to and d and d > date_to:
                continue
            filtered.append(c)
        chunks = filtered

    # Source-diversified ranking: round-robin across source types so that
    # risk, sentiment, and trade docs all appear (not just the highest-scoring
    # source type, which is usually sentiment).
    by_source: Dict[str, List[SourceChunk]] = {}
    for c in chunks:
        by_source.setdefault(c.source, []).append(c)
    for group in by_source.values():
        group.sort(key=lambda c: c.score, reverse=True)
    sources = sorted(by_source.keys())  # deterministic order
    result: List[SourceChunk] = []
    idx = {s: 0 for s in sources}
    while len(result) < k:
        added = False
        for s in sources:
            if idx[s] < len(by_source[s]):
                result.append(by_source[s][idx[s]])
                idx[s] += 1
                added = True
                if len(result) >= k:
                    break
        if not added:
            break
    return result
