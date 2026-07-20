"""
Buy-Open / Exit-Close intraday backtest.

Strategy : Buy every trading day at the open price and exit the same day
           at the close (EOD) price.
Universe : Indices from resources/indices.json and/or stock groups from
           resources/stocks.json — choose with UNIVERSE / STOCK_GROUP below
           or via command-line options.
Report   : Per-instrument day-wise table (Date | Entry Price | Exit Price |
           PnL %) followed by the total profit for each instrument.

Usage:
    python BuyOpenExitCloseStrategy.py                          # defaults below
    python BuyOpenExitCloseStrategy.py --universe indices
    python BuyOpenExitCloseStrategy.py --universe stocks
    python BuyOpenExitCloseStrategy.py --universe stocks --group banking
    python BuyOpenExitCloseStrategy.py --universe both
    python BuyOpenExitCloseStrategy.py --list                   # show stock groups
"""

import argparse
import os
from dataclasses import dataclass
from datetime import datetime

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from resources.resource_loader import (
    list_stock_files,
    load_all_stocks,
    load_indices,
    load_stocks,
)

# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE = "01-Jan-2026 9:15:00"
END_DATE   = "30-Jun-2026 15:29:59"
INTERVAL   = "30minute"
TAKE_PROFIT_PCT = 1.0  # reset "Wait for x%" once cumulative PnL exceeds this

# Default universe when no command-line options are given:
#   UNIVERSE = "indices" → index definitions from resources/indices.json
#   UNIVERSE = "stocks"  → stock lists from resources/stocks.json
#   UNIVERSE = "both"    → indices followed by stocks
# STOCK_GROUP narrows "stocks" to a single group (e.g. "banking", "it");
# leave it as None to run every group.

UNIVERSE    = "both"     # "indices" | "stocks" | "both"
STOCK_GROUP = None       # e.g. "banking"; None = all groups


@dataclass(frozen=True)
class Instrument:
    """A backtestable instrument: an index or a stock."""
    name: str
    breeze_code: str
    exchange: str
    product_type: str = "cash"


@dataclass
class DayTrade:
    date: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    wait_for_x_pct: float
    take_profit: bool


# ── Universe selection ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--universe",
        choices=("indices", "stocks", "both"),
        default=UNIVERSE,
        help=f"which resources to backtest (default: {UNIVERSE})",
    )
    parser.add_argument(
        "--group",
        default=STOCK_GROUP,
        metavar="NAME",
        help="restrict stocks to one group from stocks.json (e.g. banking); "
             "default: all groups",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the available stock groups and exit",
    )
    return parser.parse_args()


def stock_instruments(group: str | None) -> list[Instrument]:
    """Build instruments from stocks.json ('NSE:TCS-EQ' → TCS on NSE)."""
    symbols = load_stocks(group) if group else load_all_stocks()
    instruments = []
    for symbol in symbols:
        exchange, rest = symbol.split(":", 1)
        code = rest.removesuffix("-EQ")
        instruments.append(Instrument(name=symbol, breeze_code=code, exchange=exchange))
    return instruments


def index_instruments() -> list[Instrument]:
    return [
        Instrument(name=idx.name, breeze_code=idx.breeze_code, exchange=idx.exchange)
        for idx in load_indices().values()
    ]


def load_universe(universe: str, group: str | None) -> list[Instrument]:
    instruments: list[Instrument] = []
    if universe in ("indices", "both"):
        instruments += index_instruments()
    if universe in ("stocks", "both"):
        instruments += stock_instruments(group)
    return instruments


# ── Backtest ──────────────────────────────────────────────────────────────────

def fetch_daily_candles(instrument: Instrument) -> list[dict]:
    """Fetch OHLC candles for the instrument over the backtest period."""
    resp = breeze.get_historical_data_v2(
        interval=INTERVAL,
        from_date=datetime.strptime(START_DATE, "%d-%b-%Y %H:%M:%S"),
        to_date=datetime.strptime(END_DATE, "%d-%b-%Y %H:%M:%S"),
        stock_code=instrument.breeze_code,
        exchange_code=instrument.exchange,
        product_type=instrument.product_type,
    )
    candles = resp.get("Success") or []
    if not candles:
        raise ValueError(f"No historical data returned: {resp}")
    return candles


def run_backtest(instrument: Instrument) -> list[DayTrade]:
    """Buy at open, sell at close for every daily candle."""
    trades: list[DayTrade] = []
    cumulative = 0.0
    for candle in fetch_daily_candles(instrument):
        entry = float(candle["open"])
        exit_ = float(candle["close"])
        if entry <= 0:
            continue
        date_str = str(candle["datetime"]).split(" ")[0].split("T")[0]
        pnl_pct = round((exit_ - entry) / entry * 100, 2)

        cumulative = round(cumulative + pnl_pct, 2)
        take_profit = cumulative > TAKE_PROFIT_PCT
        wait_for_x = cumulative
        if take_profit:
            cumulative = 0.0

        trades.append(DayTrade(date_str, entry, exit_, pnl_pct, wait_for_x, take_profit))
    return trades


def print_instrument_report(instrument: Instrument, trades: list[DayTrade]) -> float:
    """Print the day-wise table for one instrument and return its total PnL %."""
    header = (
        f"{'Date':<12} | {'Entry Price':>12} | {'Exit Price':>12} | {'PnL %':>8} | "
        f"{'Wait for x%':>12}"
    )
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print(f"  {instrument.name}  ({instrument.exchange}:{instrument.breeze_code})")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for t in trades:
        pnl_str = f"{'+' if t.pnl_pct >= 0 else ''}{t.pnl_pct:.2f}"
        wait_str = f"{'+' if t.wait_for_x_pct >= 0 else ''}{t.wait_for_x_pct:.2f}"
        if t.take_profit:
            wait_str += " TP"
        print(
            f"{t.date:<12} | {t.entry_price:>12.2f} | {t.exit_price:>12.2f} | "
            f"{pnl_str:>8} | {wait_str:>12}"
        )

    total_pnl = round(sum(t.pnl_pct for t in trades), 2)
    wins = sum(1 for t in trades if t.pnl_pct > 0)

    print(sep)
    print(
        f"Total: {len(trades)} days  |  Wins: {wins}  |  Losses: {len(trades) - wins}  |  "
        f"Total Profit: {'+' if total_pnl >= 0 else ''}{total_pnl:.2f}%"
    )
    return total_pnl


def print_summary(totals: dict[str, float]) -> None:
    header = f"{'Instrument':<28} | {'Total Profit %':>14}"
    print(f"\n{'='*len(header)}")
    print("  TOTAL PROFIT — INSTRUMENT-WISE SUMMARY")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for name, total in sorted(totals.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{name:<28} | {'+' if total >= 0 else ''}{total:>13.2f}")
    print(f"{'='*len(header)}\n")


# ── Run ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.list:
        print("Available stock groups:")
        for name in list_stock_files():
            print(f"  {name}")
        return

    instruments = load_universe(args.universe, args.group)

    load_dotenv()
    global breeze
    breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
    breeze.generate_session(
        api_secret=os.getenv("BREEZE_API_SECRET"),
        session_token=os.getenv("BREEZE_SESSION_TOKEN"),
    )
    print("Session Generated Successfully\n")

    scope = args.universe + (f" (group: {args.group})" if args.group else "")
    print(f"Running Buy-Open/Exit-Close backtest for {len(instruments)} instruments [{scope}]...")
    print(f"Period: {START_DATE}  →  {END_DATE}")

    totals: dict[str, float] = {}
    for instrument in instruments:
        try:
            trades = run_backtest(instrument)
            totals[instrument.name] = print_instrument_report(instrument, trades)
        except Exception as exc:
            print(
                f"\n  [ERROR] {instrument.name} "
                f"({instrument.exchange}:{instrument.breeze_code}): {exc}"
            )

    if totals:
        print_summary(totals)


if __name__ == "__main__":
    main()
