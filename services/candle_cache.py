"""
File-based cache for raw candle data fetched from the Breeze API.

Backtests otherwise re-download the same historical candles from Breeze on every
run, which is slow for fine-grained (1-second) intervals because the request is
paged into many small market-hours chunks. This cache persists the *raw* fetched
candles to Parquet keyed by the full request signature (symbol, interval, date
range, etc.). On a subsequent run with the same signature the data is loaded from
disk instead of hitting the API.

Raw 1-second candles are cached (not the resampled output), so the same cached
data can be re-resampled to any interval without re-fetching.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime

import pandas as pd

# Default cache directory; override with the BACKTEST_CACHE_DIR env var.
_DEFAULT_CACHE_DIR = os.environ.get(
    "BACKTEST_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".candle_cache"),
)


def _json_default(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)


class CandleCache:
    """Persist and reuse raw candle lists keyed by their request parameters."""

    def __init__(self, cache_dir: str | None = None, enabled: bool = True):
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        # Allow disabling via env var (e.g. BACKTEST_CACHE_DISABLED=1) or argument.
        self.enabled = enabled and os.environ.get("BACKTEST_CACHE_DISABLED", "") not in ("1", "true", "True")
        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    # ── key handling ──────────────────────────────────────────────────────────

    @staticmethod
    def make_key(**params) -> str:
        """Build a stable hash key from request parameters."""
        blob = json.dumps(params, sort_keys=True, default=_json_default)
        digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
        # Include a readable prefix for easy inspection of the cache directory.
        symbol = str(params.get("stock_code") or params.get("symbol") or "data")
        interval = str(params.get("interval") or "")
        return f"{symbol}_{interval}_{digest}".strip("_")

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.parquet")

    # ── load / save ─────────────────────────────────────────────────────────--

    def load(self, key: str) -> list[dict] | None:
        """Return cached candles for key, or None if absent/disabled."""
        if not self.enabled:
            return None
        path = self._path(key)
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        return df.to_dict("records")

    def save(self, key: str, candles: list[dict]) -> None:
        """Persist candles for key. No-op if disabled or candles is empty."""
        if not self.enabled or not candles:
            return
        df = pd.DataFrame(candles)
        df.to_parquet(self._path(key), index=False)

    def get_or_fetch(self, fetch_fn, **params) -> list[dict]:
        """
        Return cached candles for the given params, otherwise call ``fetch_fn()``
        (which must return a list[dict]), cache the result, and return it.
        """
        key = self.make_key(**params)
        cached = self.load(key)
        if cached is not None:
            return cached
        candles = fetch_fn()
        self.save(key, candles)
        return candles
