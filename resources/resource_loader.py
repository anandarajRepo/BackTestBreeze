"""
Loader utilities for the resources folder.

- resources/stocks.json           : all stock lists in one file, keyed by group
                                    name (sector / watchlist), each value a list
                                    of Breeze stock codes (e.g. "NSE:TCS-EQ").
- resources/futures_contracts.json: futures contract details, a list of objects
                                    with keys: scrip, expiry, lot_size, margin_per_lot.
- resources/indices.json          : index definitions with name, symbol,
                                    breeze_code and exchange.

Usage:
    from resources.resource_loader import (
        load_stocks, load_all_stocks, load_futures_contracts, load_indices
    )

    banking      = load_stocks("banking")          # list[str] for one group
    all_symbols  = load_all_stocks()               # merged, de-duplicated list
    contracts    = load_futures_contracts()        # dict[scrip] -> FutureContract
    indices      = load_indices()                  # dict[breeze_code] -> Index
"""

import json
from dataclasses import dataclass
from pathlib import Path

RESOURCES_DIR = Path(__file__).resolve().parent
STOCKS_FILE = RESOURCES_DIR / "stocks.json"
FUTURES_FILE = RESOURCES_DIR / "futures_contracts.json"
INDICES_FILE = RESOURCES_DIR / "indices.json"


@dataclass(frozen=True)
class FutureContract:
    scrip: str
    expiry: str
    lot_size: int
    margin_per_lot: float


@dataclass(frozen=True)
class Index:
    name: str
    symbol: str
    breeze_code: str
    exchange: str


def _load_stock_groups() -> dict[str, list[str]]:
    with STOCKS_FILE.open() as fh:
        return json.load(fh)


def list_stock_files() -> list[str]:
    """Names of the available stock groups in stocks.json."""
    return sorted(_load_stock_groups())


def load_stocks(name: str) -> list[str]:
    """Load one stock group, e.g. load_stocks("banking")."""
    groups = _load_stock_groups()
    if name not in groups:
        raise KeyError(
            f"No stock list '{name}'. Available: {', '.join(sorted(groups))}"
        )
    return list(groups[name])


def load_all_stocks() -> list[str]:
    """Merge every stock group, preserving order and dropping duplicates."""
    seen: dict[str, None] = {}
    for name in list_stock_files():
        for symbol in load_stocks(name):
            seen.setdefault(symbol, None)
    return list(seen)


def load_futures_contracts() -> dict[str, FutureContract]:
    """Load futures contract details keyed by scrip code."""
    with FUTURES_FILE.open() as fh:
        rows = json.load(fh)
    return {row["scrip"]: FutureContract(**row) for row in rows}


def load_indices() -> dict[str, Index]:
    """Load index definitions keyed by Breeze code."""
    with INDICES_FILE.open() as fh:
        rows = json.load(fh)["indices"]
    return {row["breeze_code"]: Index(**row) for row in rows}


if __name__ == "__main__":
    print(f"Stock lists: {', '.join(list_stock_files())}")
    print(f"Total unique stocks: {len(load_all_stocks())}")
    print(f"Futures contracts:   {len(load_futures_contracts())}")
    print(f"Indices:             {len(load_indices())}")
