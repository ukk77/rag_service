"""LLM synthesis — builds prompts from retrieved chunks and calls the configured LLM.

Uses the same ollama/openai provider pattern as harness/llm_client.py.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from .config import Settings, get_settings
from .models.schemas import SourceChunk

log = logging.getLogger(__name__)

_ASK_SYSTEM = (
    "You are a market intelligence assistant for an algorithmic trading platform. "
    "Answer the user's question using ONLY the provided context chunks. "
    "Always attribute your answer to specific data points (ticker, date, source). "
    "If the context is insufficient, say so — do not hallucinate."
)

_SUMMARIZE_SYSTEM = (
    "You are an operator oversight assistant for an algorithmic trading platform. "
    "Write a concise 2-3 paragraph narrative summarising the trading run, "
    "grounded in both the run statistics and the retrieved historical context provided. "
    "Focus on: market regime, allocation rationale, signal quality, and portfolio health."
)


def _format_chunks(chunks: List[SourceChunk]) -> str:
    if not chunks:
        return "(no context retrieved)"
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] ({c.source}, {c.ticker or 'N/A'}, {c.date or 'N/A'}) {c.text}")
    return "\n".join(lines)


def _call_ollama(prompt: str, settings: Settings, system: str) -> Optional[str]:
    import requests
    url = f"{settings.llm_base_url}/api/chat"
    payload = {
        "model": settings.llm_model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        log.warning("Ollama call failed: %s", e)
        return None


def _call_openai(prompt: str, settings: Settings, system: str) -> Optional[str]:
    try:
        import openai
        client = openai.OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.llm_model if "gpt" in settings.llm_model else "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning("OpenAI call failed: %s", e)
        return None


def synthesize_answer(
    query: str,
    chunks: List[SourceChunk],
    settings: Optional[Settings] = None,
    system_prompt: Optional[str] = None,
) -> tuple[str, str]:
    """Call LLM with retrieved context. Returns (answer, model_used)."""
    if settings is None:
        settings = get_settings()

    context_str = _format_chunks(chunks)
    system = system_prompt or _ASK_SYSTEM
    prompt = f"CONTEXT:\n{context_str}\n\nQUESTION: {query}\n\nANSWER:"

    model_used = f"{settings.llm_provider}/{settings.llm_model}"

    if settings.llm_provider == "openai":
        answer = _call_openai(prompt, settings, system)
    else:
        answer = _call_ollama(prompt, settings, system)

    if answer is None:
        answer = (
            "[LLM unavailable — raw context below]\n\n"
            + "\n".join(c.text for c in chunks[:3])
        )
        model_used = "fallback/raw_context"

    return answer, model_used


def synthesize_run_summary(
    run_report: dict,
    chunks: List[SourceChunk],
    settings: Optional[Settings] = None,
) -> tuple[str, str]:
    """Build run summary prompt from report dict + retrieved context."""
    if settings is None:
        settings = get_settings()

    regime = run_report.get("regime", "unknown")
    alloc = run_report.get("allocation_summary", {})
    executed = run_report.get("executed_trades", 0)
    conflicts = run_report.get("conflicts_blocked", 0)
    unrealized_pnl = run_report.get("unrealized_pnl", 0)
    top_buys = run_report.get("top_buys", [])
    top_sells = run_report.get("top_sells", [])
    open_positions = run_report.get("open_positions", 0)

    alloc_str = ", ".join(f"{k}=${v:,.0f}" for k, v in alloc.items()) if alloc else "N/A"
    buys_str = ", ".join(top_buys[:5]) if top_buys else "none"
    sells_str = ", ".join(top_sells[:5]) if top_sells else "none"

    run_stats = (
        f"REGIME: {regime}\n"
        f"ALLOCATION: {alloc_str}\n"
        f"EXECUTED TRADES: {executed} (BUY/SELL actions)\n"
        f"CONFLICTS BLOCKED: {conflicts}\n"
        f"TOP BUYS: {buys_str}\n"
        f"TOP SELLS: {sells_str}\n"
        f"OPEN POSITIONS: {open_positions}\n"
        f"UNREALIZED P&L: ${unrealized_pnl:+,.2f}\n"
    )

    context_str = _format_chunks(chunks)
    prompt = (
        f"RUN STATISTICS:\n{run_stats}\n\n"
        f"HISTORICAL CONTEXT:\n{context_str}\n\n"
        "Write a 2-3 paragraph operator oversight narrative grounded in the above:"
    )

    return synthesize_answer(prompt, chunks=[], settings=settings, system_prompt=_SUMMARIZE_SYSTEM)
