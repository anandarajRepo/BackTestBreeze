"""
Loader utilities for the resources folder.

- resources/stocks/*.txt      : one Breeze stock code per line (e.g. "NSE:TCS-EQ"),
                                grouped into separate files (by sector / watchlist).
- resources/futures_contracts.txt : tab-separated futures contract details with
                                columns: Scrip, Expiry, Lot Size, Margin/Lot.

Usage:
    from resources.resource_loader import load_stocks, load_all_stocks, load_futures_contracts

    banking      = load_stocks("banking")          # list[str] from stocks/banking.txt
    all_symbols  = load_all_stocks()               # merged, de-duplicated list from every txt file
    contracts    = load_futures_contracts()        # dict[scrip] -> FutureContract
"""

import csv
from dataclasses import dataclass
from pathlib import Path

RESOURCES_DIR = Path(__file__).resolve().parent
STOCKS_DIR = RESOURCES_DIR / "stocks"
FUTURES_FILE = RESOURCES_DIR / "futures_contracts.txt"


@dataclass(frozen=True)
class FutureContract:
    scrip: str
    expiry: str
    lot_size: int
    margin_per_lot: float


def _read_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def list_stock_files() -> list[str]:
    """Names (without extension) of the available stock list files."""
    return sorted(p.stem for p in STOCKS_DIR.glob("*.txt"))


def load_stocks(name: str) -> list[str]:
    """Load one stock list, e.g. load_stocks("banking")."""
    path = STOCKS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"No stock list '{name}'. Available: {', '.join(list_stock_files())}"
        )
    return _read_lines(path)


def load_all_stocks() -> list[str]:
    """Merge every stock list, preserving order and dropping duplicates."""
    seen: dict[str, None] = {}
    for name in list_stock_files():
        for symbol in load_stocks(name):
            seen.setdefault(symbol, None)
    return list(seen)


def load_futures_contracts() -> dict[str, FutureContract]:
    """Load futures contract details keyed by scrip code."""
    contracts: dict[str, FutureContract] = {}
    with FUTURES_FILE.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            scrip = row["Scrip"].strip()
            contracts[scrip] = FutureContract(
                scrip=scrip,
                expiry=row["Expiry"].strip(),
                lot_size=int(row["Lot Size"]),
                margin_per_lot=float(row["Margin/Lot"]),
            )
    return contracts


if __name__ == "__main__":
    print(f"Stock lists: {', '.join(list_stock_files())}")
    print(f"Total unique stocks: {len(load_all_stocks())}")
    print(f"Futures contracts:   {len(load_futures_contracts())}")
