import csv
import os
from dataclasses import dataclass

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.gap_trend_service import GapTrendService
from strategy.gap_strategy import GapStrategy
from strategy.order_manager import OrderManager

load_dotenv()

# ── Session ───────────────────────────────────────────────────────────────────

breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN"),
)
print("Session Generated Successfully\n")

# ── Mode ──────────────────────────────────────────────────────────────────────
# Set MULTI_SYMBOL = True  to run all symbols in SYMBOLS list with a report.
# Set MULTI_SYMBOL = False to run a single symbol defined by STOCK_CODE.

MULTI_SYMBOL = True

# ── Single-Symbol Configuration ───────────────────────────────────────────────

STOCK_CODE    = "OLAELE"
EXCHANGE_CODE = "NSE"

# ── Multi-Symbol List ─────────────────────────────────────────────────────────

SYMBOLS = [
    # Oil & Gas — PSU Upstream
    "NSE:ONGC-EQ",
    "NSE:OIL-EQ",
    "NSE:GAIL-EQ",

    # Renewables — Structural Beneficiaries
    "NSE:ADANIGREEN-EQ",
    "NSE:TATAPOWER-EQ",
    "NSE:CESC-EQ",

    # City Gas / LNG Distribution
    "NSE:IGL-EQ",
    "NSE:MGL-EQ",
    "NSE:GUJGASLTD-EQ",
    "NSE:PETRONET-EQ",

    # Defence
    "NSE:HAL-EQ",
    "NSE:BEL-EQ",
    "NSE:MAZDOCK-EQ",
    "NSE:DATAPATTNS-EQ",

    # Sugar - Ethanol
    "NSE:EIDPARRY-EQ",
    "NSE:BALRAMCHIN-EQ",
    "NSE:TRIVENI-EQ",

    # Pharmaceuticals
    "NSE:SUNPHARMA-EQ",
    "NSE:DIVISLAB-EQ",
    "NSE:CIPLA-EQ",

    # Petroleum (Oil Marketing Companies)
    "NSE:IOC-EQ",
    "NSE:BPCL-EQ",
    "NSE:HINDPETRO-EQ",

    # Airlines
    "NSE:INDIGO-EQ",

    # Paints
    "NSE:ASIANPAINT-EQ",
    "NSE:BERGEPAINT-EQ",
    "NSE:KANSAINER-EQ",
    "NSE:INDIGOPNTS-EQ",

    # Tyres
    "NSE:CEATLTD-EQ",
    "NSE:MRF-EQ",
    "NSE:APOLLOTYRE-EQ",
    "NSE:JKTYRE-EQ",
    "NSE:BALKRISIND-EQ",

    # Autos (Nifty Auto)
    "NSE:MARUTI-EQ",
    "NSE:M&M-EQ",
    "NSE:BAJAJ-AUTO-EQ",
    "NSE:EICHERMOT-EQ",
    "NSE:TVSMOTOR-EQ",

    # IPO Stocks
    "NSE:VIKRAMSOLR-EQ",
    "NSE:ATLANTAELE-EQ",
    "NSE:SOLARWORLD-EQ",
    "NSE:RUBICON-EQ",
    "NSE:MIDWESTLTD-EQ",

    # Jewellery
    "NSE:TITAN-EQ",
    "NSE:KALYANKJIL-EQ",
    "NSE:PCJEWELLER-EQ",
    "NSE:PNGBL-EQ",
    "NSE:THANGAMAYL-EQ",
    "NSE:SENCO-EQ",
    "NSE:RJIL-EQ",
    "NSE:SKYGOLD-EQ",
    "NSE:GOLDIAM-EQ",
    "NSE:DIVHJL-EQ",
    "NSE:ZODIACJL-EQ",
    "NSE:NARBADAG-EQ",
    "NSE:MOKSH-EQ",
    "NSE:SWARN-EQ",

    # Favourite Stocks
    "NSE:STLTECH-EQ",
    "NSE:AXISCADES-EQ",
]

# ── Shared Strategy Configuration ─────────────────────────────────────────────

QUANTITY      = 1
GAP_PCT       = 0.5
MAX_GAP_PCT   = 5.0
TARGET_PCT    = 5.0
STOP_LOSS_PCT = 5.0
START_DATE    = "01-Jan-2026 9:15:00"
END_DATE      = "28-Apr-2026 15:29:59"
INTERVAL      = "1day"

BEHAVIOR_LOOKBACK_DAYS  = 30
MIN_GAP_HISTORY         = 5
CONTINUATION_THRESHOLD  = 60.0
REVERSAL_THRESHOLD      = 60.0

REPORT_CSV = "backtest_report.csv"

# ── Multi-Symbol Helpers ───────────────────────────────────────────────────────

@dataclass
class SymbolSummary:
    symbol: str
    stock_code: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    max_win: float
    max_loss: float
    error: str = ""


def parse_symbol(symbol: str) -> tuple[str, str]:
    """Parse 'NSE:ONGC-EQ' → ('ONGC', 'NSE')."""
    exchange, rest = symbol.split(":", 1)
    stock_code = rest.removesuffix("-EQ")
    return stock_code, exchange


def resolve_breeze_code(stock_code: str, exchange_code: str) -> str:
    """Map an exchange ticker (e.g. 'TATAPOWER') to the ISEC/Breeze short
    code (e.g. 'TATPOW') that the historical-data API expects. Codes that
    are already Breeze short codes resolve to themselves."""
    try:
        resp = breeze.get_names(exchange_code=exchange_code, stock_code=stock_code)
    except Exception:
        return stock_code
    if isinstance(resp, dict):
        isec_code = resp.get("isec_stock_code")
        if isec_code:
            return isec_code
    return stock_code


def run_symbol(stock_code: str, exchange_code: str) -> SymbolSummary:
    symbol_label = f"{exchange_code}:{stock_code}-EQ"
    stock_code = resolve_breeze_code(stock_code, exchange_code)
    try:
        gap_trend_service = GapTrendService(breeze)
        order_manager = OrderManager(breeze)

        strategy = GapStrategy(
            gap_trend_service=gap_trend_service,
            order_manager=order_manager,
            stock_code=stock_code,
            exchange_code=exchange_code,
            quantity=QUANTITY,
            gap_pct=GAP_PCT,
            max_gap_pct=MAX_GAP_PCT,
            target_pct=TARGET_PCT,
            stop_loss_pct=STOP_LOSS_PCT,
            start_date=START_DATE,
            end_date=END_DATE,
            interval=INTERVAL,
            behavior_lookback_days=BEHAVIOR_LOOKBACK_DAYS,
            min_gap_history=MIN_GAP_HISTORY,
            continuation_threshold=CONTINUATION_THRESHOLD,
            reversal_threshold=REVERSAL_THRESHOLD,
        )

        results = strategy.run_backtest()

        if not results:
            return SymbolSummary(
                symbol=symbol_label, stock_code=stock_code,
                total_trades=0, wins=0, losses=0,
                win_rate=0.0, total_pnl=0.0, avg_pnl=0.0,
                max_win=0.0, max_loss=0.0,
            )

        pnls = [r.pnl for r in results]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_pnl = round(sum(pnls), 2)
        win_rate = round(len(wins) / len(pnls) * 100, 1)

        return SymbolSummary(
            symbol=symbol_label,
            stock_code=stock_code,
            total_trades=len(results),
            wins=len(wins),
            losses=len(losses),
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl=round(total_pnl / len(results), 2),
            max_win=round(max(wins, default=0.0), 2),
            max_loss=round(min(losses, default=0.0), 2),
        )

    except Exception as exc:
        print(f"  [ERROR] {symbol_label}: {exc}")
        return SymbolSummary(
            symbol=symbol_label, stock_code=stock_code,
            total_trades=0, wins=0, losses=0,
            win_rate=0.0, total_pnl=0.0, avg_pnl=0.0,
            max_win=0.0, max_loss=0.0, error=str(exc),
        )


def print_report(summaries: list[SymbolSummary]) -> None:
    successful = [s for s in summaries if not s.error]
    failed = [s for s in summaries if s.error]

    col = {
        "symbol":  28,
        "trades":   7,
        "wins":     5,
        "losses":   7,
        "win%":     6,
        "total":   10,
        "avg":      8,
        "max_win":  9,
        "max_loss": 9,
    }

    header = (
        f"{'Symbol':<{col['symbol']}} "
        f"{'Trades':>{col['trades']}} "
        f"{'Wins':>{col['wins']}} "
        f"{'Losses':>{col['losses']}} "
        f"{'Win%':>{col['win%']}} "
        f"{'Total PnL':>{col['total']}} "
        f"{'Avg PnL':>{col['avg']}} "
        f"{'Max Win':>{col['max_win']}} "
        f"{'Max Loss':>{col['max_loss']}}"
    )
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("  CONSOLIDATED BACKTEST REPORT — SYMBOL-WISE SUMMARY")
    print(f"  Period : {START_DATE}  →  {END_DATE}")
    print(f"  Gap    : {GAP_PCT}%–{MAX_GAP_PCT}%  |  Target: {TARGET_PCT}%  |  SL: {STOP_LOSS_PCT}%")
    print(f"{'='*len(header)}\n")

    print(header)
    print(sep)

    for s in sorted(successful, key=lambda x: x.total_pnl, reverse=True):
        pnl_str = f"{'+' if s.total_pnl >= 0 else ''}{s.total_pnl:.2f}"
        avg_str = f"{'+' if s.avg_pnl >= 0 else ''}{s.avg_pnl:.2f}"
        print(
            f"{s.symbol:<{col['symbol']}} "
            f"{s.total_trades:>{col['trades']}} "
            f"{s.wins:>{col['wins']}} "
            f"{s.losses:>{col['losses']}} "
            f"{s.win_rate:>{col['win%']}.1f} "
            f"{pnl_str:>{col['total']}} "
            f"{avg_str:>{col['avg']}} "
            f"{s.max_win:>{col['max_win']}.2f} "
            f"{s.max_loss:>{col['max_loss']}.2f}"
        )

    print(sep)

    all_trades  = sum(s.total_trades for s in successful)
    all_wins    = sum(s.wins for s in successful)
    all_losses  = sum(s.losses for s in successful)
    all_pnl     = round(sum(s.total_pnl for s in successful), 2)
    overall_wr  = round(all_wins / all_trades * 100, 1) if all_trades else 0.0
    overall_avg = round(all_pnl / all_trades, 2) if all_trades else 0.0
    pnl_str = f"{'+' if all_pnl >= 0 else ''}{all_pnl:.2f}"
    avg_str = f"{'+' if overall_avg >= 0 else ''}{overall_avg:.2f}"

    print(
        f"{'TOTAL / OVERALL':<{col['symbol']}} "
        f"{all_trades:>{col['trades']}} "
        f"{all_wins:>{col['wins']}} "
        f"{all_losses:>{col['losses']}} "
        f"{overall_wr:>{col['win%']}.1f} "
        f"{pnl_str:>{col['total']}} "
        f"{avg_str:>{col['avg']}} "
        f"{'':>{col['max_win']}} "
        f"{'':>{col['max_loss']}}"
    )
    print(f"{'='*len(header)}\n")

    if failed:
        print(f"  Symbols with errors ({len(failed)}):")
        for s in failed:
            print(f"    {s.symbol:<30} {s.error}")
        print()


def save_csv(summaries: list[SymbolSummary], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Symbol", "Total Trades", "Wins", "Losses", "Win Rate %",
            "Total PnL", "Avg PnL", "Max Win", "Max Loss", "Error",
        ])
        for s in summaries:
            writer.writerow([
                s.symbol, s.total_trades, s.wins, s.losses,
                s.win_rate, s.total_pnl, s.avg_pnl,
                s.max_win, s.max_loss, s.error,
            ])
    print(f"  CSV report saved → {path}\n")


# ── Run ───────────────────────────────────────────────────────────────────────

if MULTI_SYMBOL:
    seen: set[str] = set()
    unique_symbols: list[str] = []
    for sym in SYMBOLS:
        if sym not in seen:
            seen.add(sym)
            unique_symbols.append(sym)

    print(f"Running backtest for {len(unique_symbols)} symbols...\n")

    summaries: list[SymbolSummary] = []
    for sym in unique_symbols:
        stock_code, exchange_code = parse_symbol(sym)
        summaries.append(run_symbol(stock_code, exchange_code))

    print_report(summaries)
    save_csv(summaries, REPORT_CSV)

else:
    gap_trend_service = GapTrendService(breeze)
    order_manager = OrderManager(breeze)

    strategy = GapStrategy(
        gap_trend_service=gap_trend_service,
        order_manager=order_manager,
        stock_code=resolve_breeze_code(STOCK_CODE, EXCHANGE_CODE),
        exchange_code=EXCHANGE_CODE,
        quantity=QUANTITY,
        gap_pct=GAP_PCT,
        max_gap_pct=MAX_GAP_PCT,
        target_pct=TARGET_PCT,
        stop_loss_pct=STOP_LOSS_PCT,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        behavior_lookback_days=BEHAVIOR_LOOKBACK_DAYS,
        min_gap_history=MIN_GAP_HISTORY,
        continuation_threshold=CONTINUATION_THRESHOLD,
        reversal_threshold=REVERSAL_THRESHOLD,
    )

    strategy.run_backtest()
