"""Watermark tracking for incremental RAG ingestion.

Each source+ticker combination tracks its last-ingested timestamp/date,
identical in pattern to market_data/*.freshness sidecar files.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class WatermarkStore:
    """Persistent JSON watermark file: {source: {ticker: last_ingested_iso}}."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("Watermark load failed (%s) — starting fresh: %s", self._path, e)
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def get(self, source: str, ticker: str = "_global") -> Optional[str]:
        """Return last-ingested ISO timestamp for (source, ticker), or None."""
        return self._data.get(source, {}).get(ticker)

    def get_global(self, source: str) -> Optional[str]:
        """Return global (non-ticker-scoped) watermark for a source."""
        return self.get(source, "_global")

    def set(self, source: str, ticker: str, value: str) -> None:
        """Update watermark and persist."""
        if source not in self._data:
            self._data[source] = {}
        self._data[source][ticker] = value
        self._save()

    def set_global(self, source: str, value: str) -> None:
        self.set(source, "_global", value)

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
