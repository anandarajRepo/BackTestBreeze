"""
ORB Portfolio Backtest — Momentum-ranked Open Range Breakout

Flow per backtest run:
  1. Score ALL symbols at once using MomentumScoringService.
  2. Select top TOP_N_STOCKS by composite momentum score.
  3. Fetch 1-minute intraday data for each of those stocks.
  4. For every trading day:
       a. Compute the 15-minute opening range for each stock.
       b. Scan post-ORB candles for a breakout (high > ORB-high → BUY,
          low < ORB-low → SELL).
       c. Collect breakout candidates ordered by the time they broke out.
       d. Accept the first MAX_DAILY_TRADES unique stocks that broke out;
          ignore subsequent breakouts for stocks already traded that day.
  5. Simulate exits (target / stop-loss / market-close) and record PnL.
  6. Print consolidated and per-day reports; save to CSV.
"""

import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from models.orb_models import BreakoutDirection, OpenRange, ORBTradeResult
from services.momentum_service import MomentumScore, MomentumScoringService
from services.orb_data_service import ORBDataService
from services.trend_direction_service import TrendDirectionService
from strategy.orb_strategy import ORBStrategy

load_dotenv()

# ── Session ───────────────────────────────────────────────────────────────────

breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN"),
)
print("Session Generated Successfully\n")

# ── Universe ──────────────────────────────────────────────────────────────────

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

# ── Strategy Configuration ────────────────────────────────────────────────────

QUANTITY          = 1
ORB_MINUTES       = 15       # Opening range period: 9:15–9:30 AM
STOP_LOSS_PCT     = 1.5
RISK_REWARD_RATIO = 2.0
START_DATE        = "30-Apr-2026 9:15:00"
END_DATE          = "30-Apr-2026 15:29:59"
INTERVAL          = "1second"

# ── Portfolio Selection ───────────────────────────────────────────────────────

TOP_N_STOCKS     = 15   # Keep top N stocks by momentum score
MAX_DAILY_TRADES = 3    # Maximum trades to take per calendar day

# ── Momentum Scoring ──────────────────────────────────────────────────────────

MIN_MOMENTUM_SCORE     = 50.0
MOMENTUM_LOOKBACK_DAYS = 30

# ── Trend Filter ──────────────────────────────────────────────────────────────

ENABLE_TREND_FILTER = True
TREND_FILTER_MODE   = "STRICT"
TREND_LOOKBACK_DAYS = 10
HISTORICAL_WEIGHT   = 0.6
INTRADAY_WEIGHT     = 0.4

REPORT_CSV = "orb_portfolio_report.csv"

# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass
class PortfolioTradeResult:
    stock_code:      str
    trade_date:      date
    direction:       BreakoutDirection
    orb_high:        float
    orb_low:         float
    entry_price:     float
    target:          float
    stop_loss:       float
    exit_price:      float
    exit_reason:     str
    pnl:             float
    breakout_time:   str
    momentum_score:  Optional[float] = None
    momentum_rank:   Optional[int]   = None


@dataclass
class DailySummary:
    trade_date:  date
    trades_taken: int
    total_pnl:   float
    symbols:     list[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────


def parse_symbol(symbol: str) -> tuple[str, str]:
    """'NSE:ONGC-EQ' → ('ONGC', 'NSE'); 'ONGC' → ('ONGC', 'NSE')"""
    if ":" in symbol:
        exchange, rest = symbol.split(":", 1)
    else:
        exchange, rest = "NSE", symbol
    stock_code = rest.removesuffix("-EQ")
    return stock_code, exchange


def deduplicate(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _build_open_range(orb_candles: list[dict]) -> OpenRange:
    high = max(float(c["high"]) for c in orb_candles)
    low  = min(float(c["low"])  for c in orb_candles)
    return OpenRange(high=high, low=low)


def _detect_breakout(candle: dict, orb: OpenRange) -> BreakoutDirection:
    if float(candle["high"]) > orb.high:
        return BreakoutDirection.BUY
    if float(candle["low"]) < orb.low:
        return BreakoutDirection.SELL
    return BreakoutDirection.NONE


def _compute_levels(
    direction: BreakoutDirection,
    entry: float,
    stop_loss_pct: float,
    risk_reward_ratio: float,
) -> tuple[float, float]:
    if direction == BreakoutDirection.BUY:
        stop_loss = round(entry * (1 - stop_loss_pct / 100), 2)
        risk      = entry - stop_loss
        target    = round(entry + risk * risk_reward_ratio, 2)
    else:
        stop_loss = round(entry * (1 + stop_loss_pct / 100), 2)
        risk      = stop_loss - entry
        target    = round(entry - risk * risk_reward_ratio, 2)
    return target, stop_loss


def _simulate_exit(
    direction: BreakoutDirection,
    entry: float,
    target: float,
    stop_loss: float,
    post_breakout_candles: list[dict],
) -> tuple[float, str]:
    for candle in post_breakout_candles:
        high = float(candle["high"])
        low  = float(candle["low"])
        if direction == BreakoutDirection.BUY:
            if low <= stop_loss:
                return stop_loss, "stop_loss"
            if high >= target:
                return target, "target"
        else:
            if high >= stop_loss:
                return stop_loss, "stop_loss"
            if low <= target:
                return target, "target"
    return float(post_breakout_candles[-1]["close"]), "close"


# ── Phase 1: Score all symbols ────────────────────────────────────────────────


def score_all_and_select_top(
    momentum_svc: MomentumScoringService,
    symbols: list[str],
    as_of_date: datetime,
    top_n: int,
) -> list[tuple[str, str, MomentumScore]]:
    """
    Score every symbol, print a ranked table, and return the top *top_n*
    entries as (stock_code, exchange_code, MomentumScore).
    """
    parsed = [parse_symbol(s) for s in symbols]

    print(f"\n{'='*70}")
    print(f"  PHASE 1 — Momentum scoring {len(parsed)} symbols (as of {as_of_date.date()})")
    print(f"{'='*70}")

    scores = momentum_svc.score_all_symbols(
        symbols=parsed,
        as_of_date=as_of_date,
        lookback_days=MOMENTUM_LOOKBACK_DAYS,
    )

    # Print ranked table
    print(f"\n  {'Rank':<5} {'Symbol':<12} {'Score':>6}  {'Quality':<12}  "
          f"{'ROC5d':>7}  {'RSI':>5}  {'VolRatio':>8}")
    print(f"  {'-'*65}")
    for rank, ms in enumerate(scores, 1):
        marker = " ◀ TOP" if rank <= top_n else ""
        print(
            f"  {rank:<5} {ms.symbol:<12} {ms.composite_score:>6.1f}  "
            f"{ms.data_quality:<12}  {ms.roc_5d:>+7.1f}%  "
            f"{ms.rsi_14:>5.0f}  {ms.volume_ratio_5d:>8.2f}{marker}"
        )
    print()

    # Select top N that meet minimum score threshold
    top = [
        (stock_code, exchange_code, ms)
        for stock_code, exchange_code, ms in (
            (parse_symbol(ms.symbol)[0], parse_symbol(ms.symbol)[1], ms)
            for ms in scores[:top_n]
        )
        if ms.composite_score >= MIN_MOMENTUM_SCORE
    ]

    # Rebuild properly using the already-parsed tuples from `parsed`
    symbol_map = {sc: (sc, exc) for sc, exc in parsed}
    top_structured: list[tuple[str, str, MomentumScore]] = []
    for rank, ms in enumerate(scores[:top_n], 1):
        sc, exc = parse_symbol(ms.symbol)
        if ms.composite_score >= MIN_MOMENTUM_SCORE:
            top_structured.append((sc, exc, ms))

    filtered_out = top_n - len(top_structured)
    print(f"  Selected {len(top_structured)} stocks for ORB scanning "
          f"(filtered {filtered_out} below min score {MIN_MOMENTUM_SCORE})")
    print(f"{'='*70}\n")

    return top_structured


# ── Phase 2 & 3: Per-day ORB scan across top stocks ──────────────────────────


def run_portfolio_orb_backtest(
    orb_data_svc: ORBDataService,
    trend_svc: Optional[TrendDirectionService],
    top_stocks: list[tuple[str, str, MomentumScore]],  # (code, exchange, score)
) -> list[PortfolioTradeResult]:
    """
    For each trading day:
      1. Compute the opening range for all top_stocks.
      2. Detect breakouts per stock in chronological order.
      3. Accept at most MAX_DAILY_TRADES per day (earliest breakouts win).
    """
    print(f"\n{'='*70}")
    print(f"  PHASE 2 — Fetching intraday data for {len(top_stocks)} stocks")
    print(f"{'='*70}")

    # Fetch intraday candles for all top stocks
    stock_candles: dict[str, dict] = {}  # stock_code → {date: [candles]}
    for stock_code, exchange_code, ms in top_stocks:
        try:
            candles = orb_data_svc.get_intraday_candles(
                stock_code, exchange_code, START_DATE, END_DATE, INTERVAL
            )
            stock_candles[stock_code] = ORBDataService.group_by_date(candles)
            print(f"  {stock_code:<12} — {sum(len(v) for v in stock_candles[stock_code].values())} candles "
                  f"across {len(stock_candles[stock_code])} days")
        except Exception as exc:
            print(f"  {stock_code:<12} — ERROR: {exc}")

    if not stock_candles:
        print("  No data fetched. Aborting.")
        return []

    # Union of all trading dates
    all_dates = sorted(
        set(d for days in stock_candles.values() for d in days.keys())
    )

    print(f"\n{'='*70}")
    print(f"  PHASE 3 — ORB scan across {len(all_dates)} trading days "
          f"(max {MAX_DAILY_TRADES} trades/day)")
    print(f"{'='*70}\n")

    # Pre-compute per-stock trend analysis (once, using start date as cutoff)
    first_date = all_dates[0]
    as_of_date = datetime(first_date.year, first_date.month, first_date.day)
    hist_trends = {}
    if trend_svc and ENABLE_TREND_FILTER:
        print("  Computing historical trend for each stock…")
        for stock_code, exchange_code, _ in top_stocks:
            try:
                trend = trend_svc.analyze_trend(
                    stock_code=stock_code,
                    exchange_code=exchange_code,
                    as_of_date=as_of_date,
                    lookback_days=TREND_LOOKBACK_DAYS,
                )
                hist_trends[stock_code] = trend
            except Exception as exc:
                print(f"    {stock_code}: trend error — {exc}")
        print()

    momentum_rank_map = {sc: rank for rank, (sc, _, _) in enumerate(top_stocks, 1)}
    momentum_score_map = {sc: ms.composite_score for sc, _, ms in top_stocks}

    all_results: list[PortfolioTradeResult] = []
    daily_summaries: list[DailySummary] = []

    for trade_date in all_dates:
        # Collect breakout candidates for this day across all stocks
        @dataclass
        class _Candidate:
            stock_code:     str
            direction:      BreakoutDirection
            breakout_time:  str
            breakout_idx:   int
            orb:            OpenRange
            entry:          float
            post_orb_candles: list[dict]

        candidates: list[_Candidate] = []

        for stock_code, exchange_code, ms in top_stocks:
            day_candles = stock_candles.get(stock_code, {}).get(trade_date)
            if not day_candles or len(day_candles) <= ORB_MINUTES:
                continue

            orb_candles      = ORBDataService.get_orb_candles(day_candles, ORB_MINUTES)
            post_orb_candles = ORBDataService.get_post_orb_candles(day_candles, ORB_MINUTES)
            orb              = _build_open_range(orb_candles)

            # Scan post-ORB candles for the first breakout
            for idx, candle in enumerate(post_orb_candles):
                direction = _detect_breakout(candle, orb)
                if direction != BreakoutDirection.NONE:
                    entry = orb.high if direction == BreakoutDirection.BUY else orb.low

                    # Optional trend alignment check
                    if hist_trends and ENABLE_TREND_FILTER:
                        hist_trend = hist_trends.get(stock_code)
                        intraday_trend = trend_svc.analyze_intraday_trend(
                            stock_code, orb_candles
                        ) if trend_svc else None
                        if hist_trend:
                            aligned, _ = trend_svc.is_signal_aligned(
                                breakout_is_buy=direction == BreakoutDirection.BUY,
                                stock_trend=hist_trend,
                                intraday=intraday_trend,
                                filter_mode=TREND_FILTER_MODE,
                                historical_weight=HISTORICAL_WEIGHT,
                                intraday_weight=INTRADAY_WEIGHT,
                            )
                            if not aligned:
                                break  # Skip this stock for today

                    candidates.append(_Candidate(
                        stock_code=stock_code,
                        direction=direction,
                        breakout_time=candle["datetime"],
                        breakout_idx=idx,
                        orb=orb,
                        entry=entry,
                        post_orb_candles=post_orb_candles,
                    ))
                    break  # Only first breakout per stock per day

        # First-crossover priority: order strictly by when the breakout occurred,
        # ignoring momentum rank. The earliest MAX_DAILY_TRADES crossovers win.
        candidates.sort(key=lambda c: datetime.fromisoformat(c.breakout_time))
        selected = candidates[:MAX_DAILY_TRADES]

        day_pnl    = 0.0
        day_stocks = []

        print(f"  {trade_date}  — {len(candidates)} breakout(s) found, "
              f"taking {len(selected)} (max {MAX_DAILY_TRADES})")

        for cand in selected:
            target, stop_loss = _compute_levels(
                cand.direction, cand.entry, STOP_LOSS_PCT, RISK_REWARD_RATIO
            )

            remaining = cand.post_orb_candles[cand.breakout_idx + 1:]
            if remaining:
                exit_price, exit_reason = _simulate_exit(
                    cand.direction, cand.entry, target, stop_loss, remaining
                )
            else:
                exit_price  = float(cand.post_orb_candles[cand.breakout_idx]["close"])
                exit_reason = "close"

            if cand.direction == BreakoutDirection.BUY:
                pnl = round((exit_price - cand.entry) * QUANTITY, 2)
            else:
                pnl = round((cand.entry - exit_price) * QUANTITY, 2)

            day_pnl += pnl
            day_stocks.append(cand.stock_code)

            bt_time   = datetime.fromisoformat(cand.breakout_time).strftime("%H:%M")
            dir_label = "BUY " if cand.direction == BreakoutDirection.BUY else "SELL"
            pnl_sign  = "+" if pnl >= 0 else ""
            rank      = momentum_rank_map.get(cand.stock_code, "?")
            score     = momentum_score_map.get(cand.stock_code, 0.0)
            print(
                f"    [{rank}] {cand.stock_code:<12} {dir_label} @{bt_time}"
                f"  ORB[{cand.orb.low:.2f}–{cand.orb.high:.2f}]"
                f"  Entry {cand.entry:.2f}  T {target:.2f}  SL {stop_loss:.2f}"
                f"  Exit {exit_price:.2f} [{exit_reason:10s}]"
                f"  PnL {pnl_sign}{pnl:.2f}"
                f"  Mom:{score:.0f}"
            )

            all_results.append(PortfolioTradeResult(
                stock_code    = cand.stock_code,
                trade_date    = trade_date,
                direction     = cand.direction,
                orb_high      = cand.orb.high,
                orb_low       = cand.orb.low,
                entry_price   = cand.entry,
                target        = target,
                stop_loss     = stop_loss,
                exit_price    = exit_price,
                exit_reason   = exit_reason,
                pnl           = pnl,
                breakout_time = cand.breakout_time,
                momentum_score = momentum_score_map.get(cand.stock_code),
                momentum_rank  = momentum_rank_map.get(cand.stock_code),
            ))

        if selected:
            day_sign = "+" if day_pnl >= 0 else ""
            print(f"    Day PnL: {day_sign}{day_pnl:.2f}\n")
            daily_summaries.append(DailySummary(
                trade_date=trade_date,
                trades_taken=len(selected),
                total_pnl=day_pnl,
                symbols=day_stocks,
            ))

    return all_results


# ── Reports ───────────────────────────────────────────────────────────────────


def print_final_report(results: list[PortfolioTradeResult]) -> None:
    if not results:
        print("  No trades executed.")
        return

    pnls   = [r.pnl for r in results]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total  = round(sum(pnls), 2)
    wr     = round(len(wins) / len(pnls) * 100, 1)
    avg    = round(total / len(pnls), 2)

    # Per-stock breakdown
    by_stock: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_stock[r.stock_code].append(r.pnl)

    print(f"\n{'='*72}")
    print("  PORTFOLIO ORB BACKTEST — FINAL REPORT")
    print(f"  Period    : {START_DATE}  →  {END_DATE}")
    print(f"  Universe  : {len(SYMBOLS)} symbols  →  Top {TOP_N_STOCKS} by momentum")
    print(f"  ORB period: {ORB_MINUTES} min  |  SL: {STOP_LOSS_PCT}%  |  RR: 1:{RISK_REWARD_RATIO}")
    print(f"  Max trades/day: {MAX_DAILY_TRADES}")
    print(f"{'='*72}\n")

    print(f"  {'Stock':<14} {'Trades':>6} {'Wins':>5} {'Losses':>7} "
          f"{'Win%':>5} {'Total PnL':>10} {'Avg PnL':>8}")
    print(f"  {'-'*60}")
    for stock_code in sorted(by_stock, key=lambda s: sum(by_stock[s]), reverse=True):
        sp = by_stock[stock_code]
        sw = [p for p in sp if p > 0]
        sl = [p for p in sp if p <= 0]
        st = round(sum(sp), 2)
        swr = round(len(sw) / len(sp) * 100, 1) if sp else 0.0
        sign = "+" if st >= 0 else ""
        avg_s = round(st / len(sp), 2) if sp else 0.0
        avg_sign = "+" if avg_s >= 0 else ""
        print(
            f"  {stock_code:<14} {len(sp):>6} {len(sw):>5} {len(sl):>7} "
            f"{swr:>5.1f} {sign}{st:>9.2f} {avg_sign}{avg_s:>7.2f}"
        )

    print(f"  {'-'*60}")
    total_sign = "+" if total >= 0 else ""
    avg_sign   = "+" if avg    >= 0 else ""
    print(
        f"  {'TOTAL':<14} {len(pnls):>6} {len(wins):>5} {len(losses):>7} "
        f"{wr:>5.1f} {total_sign}{total:>9.2f} {avg_sign}{avg:>7.2f}"
    )
    print(f"  Max win : {max(wins, default=0.0):.2f}   "
          f"Max loss: {min(losses, default=0.0):.2f}")
    print(f"\n{'='*72}\n")


def save_csv(results: list[PortfolioTradeResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Date", "Stock", "Direction", "Momentum Rank", "Momentum Score",
            "ORB High", "ORB Low", "Entry", "Target", "Stop Loss",
            "Exit Price", "Exit Reason", "PnL", "Breakout Time",
        ])
        for r in results:
            writer.writerow([
                r.trade_date, r.stock_code, r.direction.value,
                r.momentum_rank, r.momentum_score,
                r.orb_high, r.orb_low, r.entry_price,
                r.target, r.stop_loss, r.exit_price, r.exit_reason,
                r.pnl, r.breakout_time,
            ])
    print(f"  CSV saved → {path}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__" or True:
    unique_symbols = deduplicate(SYMBOLS)

    momentum_svc = MomentumScoringService(breeze)
    orb_data_svc = ORBDataService(breeze)
    trend_svc    = TrendDirectionService(breeze) if ENABLE_TREND_FILTER else None

    # Determine as_of_date from START_DATE (day before test period)
    start_dt  = datetime.strptime(START_DATE, "%d-%b-%Y %H:%M:%S")
    as_of_date = start_dt  # momentum scored up to (not including) start of test

    # Phase 1: Score all, select top N
    top_stocks = score_all_and_select_top(
        momentum_svc=momentum_svc,
        symbols=unique_symbols,
        as_of_date=as_of_date,
        top_n=TOP_N_STOCKS,
    )

    if not top_stocks:
        print("No stocks passed the momentum filter. Exiting.")
    else:
        # Phase 2 & 3: ORB scan with daily cap
        results = run_portfolio_orb_backtest(
            orb_data_svc=orb_data_svc,
            trend_svc=trend_svc,
            top_stocks=top_stocks,
        )

        print_final_report(results)
        save_csv(results, REPORT_CSV)
