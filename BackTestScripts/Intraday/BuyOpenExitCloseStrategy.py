"""
Buy-Open / Exit-Close intraday backtest.

Strategy : Buy every trading day at the open price and exit the same day
           at the close (EOD) price.
Universe : All indices defined in resources/indices.json.
Report   : Per-index day-wise table (Date | Entry Price | Exit Price | PnL %)
           followed by the total profit for each index.
"""

import os
from dataclasses import dataclass
from datetime import datetime

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from resources.resource_loader import Index, load_indices

load_dotenv()

# ── Session ───────────────────────────────────────────────────────────────────

breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN"),
)
print("Session Generated Successfully\n")

# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE = "01-Jan-2026 9:15:00"
END_DATE   = "30-Jun-2026 15:29:59"
INTERVAL   = "1day"
TAKE_PROFIT_PCT = 1.0  # reset "Wait for x%" once cumulative PnL exceeds this


@dataclass
class DayTrade:
    date: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    wait_for_x_pct: float
    take_profit: bool


def fetch_daily_candles(index: Index) -> list[dict]:
    """Fetch daily OHLC candles for the index over the backtest period."""
    resp = breeze.get_historical_data_v2(
        interval=INTERVAL,
        from_date=datetime.strptime(START_DATE, "%d-%b-%Y %H:%M:%S"),
        to_date=datetime.strptime(END_DATE, "%d-%b-%Y %H:%M:%S"),
        stock_code=index.breeze_code,
        exchange_code=index.exchange,
        product_type="cash",
    )
    candles = resp.get("Success") or []
    if not candles:
        raise ValueError(f"No historical data returned: {resp}")
    return candles


def run_backtest(index: Index) -> list[DayTrade]:
    """Buy at open, sell at close for every daily candle."""
    trades: list[DayTrade] = []
    cumulative = 0.0
    for candle in fetch_daily_candles(index):
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


def print_index_report(index: Index, trades: list[DayTrade]) -> float:
    """Print the day-wise table for one index and return its total PnL %."""
    header = (
        f"{'Date':<12} | {'Entry Price':>12} | {'Exit Price':>12} | {'PnL %':>8} | "
        f"{'Wait for x%':>12}"
    )
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print(f"  {index.name}  ({index.exchange}:{index.breeze_code})")
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


# ── Run ───────────────────────────────────────────────────────────────────────

indices = list(load_indices().values())
print(f"Running Buy-Open/Exit-Close backtest for {len(indices)} indices...")
print(f"Period: {START_DATE}  →  {END_DATE}")

totals: dict[str, float] = {}
for index in indices:
    try:
        trades = run_backtest(index)
        totals[index.name] = print_index_report(index, trades)
    except Exception as exc:
        print(f"\n  [ERROR] {index.name} ({index.exchange}:{index.breeze_code}): {exc}")

if totals:
    header = f"{'Index':<28} | {'Total Profit %':>14}"
    print(f"\n{'='*len(header)}")
    print("  TOTAL PROFIT — INDEX-WISE SUMMARY")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for name, total in sorted(totals.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{name:<28} | {'+' if total >= 0 else ''}{total:>13.2f}")
    print(f"{'='*len(header)}\n")
