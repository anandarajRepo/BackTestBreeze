import csv
import os
from dataclasses import dataclass

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.orb_data_service import ORBDataService
from strategy.orb_strategy import ORBStrategy

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

STOCK_CODE    = "NIFTY"
EXCHANGE_CODE = "NSE"

# ── Multi-Symbol List (sourced from FyersORB universe) ────────────────────────

SYMBOLS = [
    # Oil & Gas — PSU Upstream
    "NSE:ONGC-EQ",
    "NSE:OILIND-EQ",
    "NSE:GAIL-EQ",

    # Renewables — Structural Beneficiaries
    "NSE:ADAGRE-EQ",
    "NSE:TATPOW-EQ",
    "NSE:CESC-EQ",

    # City Gas / LNG Distribution
    "NSE:INDGAS-EQ",
    "NSE:MAHGAS-EQ",
    "NSE:GUJGA-EQ",
    "NSE:PETLNG-EQ",

    # Defence
    "NSE:HINAER-EQ",
    "NSE:BHAELE-EQ",
    "NSE:MAZDOC-EQ",
    "NSE:DATPAT-EQ",

    # Sugar - Ethanol
    "NSE:EIDPAR-EQ",
    "NSE:BALCHI-EQ",
    "NSE:TRIENG-EQ",

    # Pharmaceuticals
    "NSE:SUNPHA-EQ",
    "NSE:DIVLAB-EQ",
    "NSE:CIPLA-EQ",

    # Petroleum (Oil Marketing Companies)
    "NSE:INDOIL-EQ",
    "NSE:BHAPET-EQ",
    "NSE:HINPET-EQ",

    # Airlines
    "NSE:INDPAI-EQ",

    # Paints
    "NSE:ASIPAI-EQ",
    "NSE:BERPAI-EQ",
    "NSE:KANNER-EQ",
    "NSE:INDPAI-EQ",

    # Tyres
    "NSE:CEAT-EQ",
    "NSE:MRFTYR-EQ",
    "NSE:APOTYR-EQ",
    "NSE:JKTYRE-EQ",
    "NSE:BALIND-EQ",

    # Autos (Nifty Auto)
    "NSE:MARUTI-EQ",
    "NSE:MAHMAH-EQ",
    "NSE:BAAUTO-EQ",
    "NSE:EICMOT-EQ",
    "NSE:TVSMOT-EQ",

    # Jewellery
    "NSE:TITIND-EQ",
    "NSE:KALJEW-EQ",
    "NSE:PCJEW-EQ",
    "NSE:PNGADG-EQ",
    "NSE:THAJEW-EQ",
    "NSE:SENGOL-EQ",
    "NSE:SKYGOL-EQ",
    "NSE:GOLINT-EQ",

    # IT
    "NSE:INFTEC-EQ",
    "NSE:TCS-EQ",
    "NSE:HCLTEC-EQ",
    "NSE:WIPRO-EQ",
    "NSE:TECMAH-EQ",

    # Banking
    "NSE:HDFBAN-EQ",
    "NSE:ICIBAN-EQ",
    "NSE:AXIBAN-EQ",
    "NSE:KOTMAH-EQ",
    "NSE:STABAN-EQ",

    # Favourite Stocks
    "NSE:STETEC-EQ",
    "NSE:AXIIT-EQ",
]

# ── Strategy Configuration (mirrors FyersORB defaults) ────────────────────────

QUANTITY          = 1
ORB_MINUTES       = 15       # Opening range period in minutes (9:15–9:30 AM)
STOP_LOSS_PCT     = 1.5      # Stop-loss percentage from entry
RISK_REWARD_RATIO = 2.0      # Target = risk * RR ratio
START_DATE        = "25-Jan-2026 9:15:00"
END_DATE          = "28-Apr-2026 15:29:59"
INTERVAL          = "1minute"

REPORT_CSV = "orb_backtest_report.csv"

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


def run_symbol(stock_code: str, exchange_code: str) -> SymbolSummary:
    symbol_label = f"{exchange_code}:{stock_code}-EQ"
    try:
        orb_data_service = ORBDataService(breeze)

        strategy = ORBStrategy(
            orb_data_service  = orb_data_service,
            stock_code        = stock_code,
            exchange_code     = exchange_code,
            quantity          = QUANTITY,
            orb_minutes       = ORB_MINUTES,
            stop_loss_pct     = STOP_LOSS_PCT,
            risk_reward_ratio = RISK_REWARD_RATIO,
            start_date        = START_DATE,
            end_date          = END_DATE,
            interval          = INTERVAL,
        )

        results = strategy.run_backtest()

        if not results:
            return SymbolSummary(
                symbol=symbol_label, stock_code=stock_code,
                total_trades=0, wins=0, losses=0,
                win_rate=0.0, total_pnl=0.0, avg_pnl=0.0,
                max_win=0.0, max_loss=0.0,
            )

        pnls   = [r.pnl for r in results]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_pnl = round(sum(pnls), 2)
        win_rate  = round(len(wins) / len(pnls) * 100, 1)

        return SymbolSummary(
            symbol       = symbol_label,
            stock_code   = stock_code,
            total_trades = len(results),
            wins         = len(wins),
            losses       = len(losses),
            win_rate     = win_rate,
            total_pnl    = total_pnl,
            avg_pnl      = round(total_pnl / len(results), 2),
            max_win      = round(max(wins,   default=0.0), 2),
            max_loss     = round(min(losses, default=0.0), 2),
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
    failed     = [s for s in summaries if s.error]

    col = {
        "symbol":   28,
        "trades":    7,
        "wins":      5,
        "losses":    7,
        "win%":      6,
        "total":    10,
        "avg":       8,
        "max_win":   9,
        "max_loss":  9,
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
    print("  CONSOLIDATED ORB BACKTEST REPORT — SYMBOL-WISE SUMMARY")
    print(f"  Period    : {START_DATE}  →  {END_DATE}")
    print(f"  ORB period: {ORB_MINUTES} min  |  SL: {STOP_LOSS_PCT}%  |  RR: 1:{RISK_REWARD_RATIO}")
    print(f"{'='*len(header)}\n")

    print(header)
    print(sep)

    for s in sorted(successful, key=lambda x: x.total_pnl, reverse=True):
        pnl_str = f"{'+' if s.total_pnl >= 0 else ''}{s.total_pnl:.2f}"
        avg_str = f"{'+' if s.avg_pnl   >= 0 else ''}{s.avg_pnl:.2f}"
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
    all_wins    = sum(s.wins         for s in successful)
    all_losses  = sum(s.losses       for s in successful)
    all_pnl     = round(sum(s.total_pnl for s in successful), 2)
    overall_wr  = round(all_wins / all_trades * 100, 1) if all_trades else 0.0
    overall_avg = round(all_pnl / all_trades, 2)        if all_trades else 0.0
    pnl_str = f"{'+' if all_pnl     >= 0 else ''}{all_pnl:.2f}"
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

    print(f"Running ORB backtest for {len(unique_symbols)} symbols...\n")

    summaries: list[SymbolSummary] = []
    for sym in unique_symbols:
        stock_code, exchange_code = parse_symbol(sym)
        summaries.append(run_symbol(stock_code, exchange_code))

    print_report(summaries)
    save_csv(summaries, REPORT_CSV)

else:
    orb_data_service = ORBDataService(breeze)

    strategy = ORBStrategy(
        orb_data_service  = orb_data_service,
        stock_code        = STOCK_CODE,
        exchange_code     = EXCHANGE_CODE,
        quantity          = QUANTITY,
        orb_minutes       = ORB_MINUTES,
        stop_loss_pct     = STOP_LOSS_PCT,
        risk_reward_ratio = RISK_REWARD_RATIO,
        start_date        = START_DATE,
        end_date          = END_DATE,
        interval          = INTERVAL,
    )

    strategy.run_backtest()
