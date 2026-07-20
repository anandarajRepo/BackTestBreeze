"""
ORB Portfolio Backtest — Momentum-ranked Open Range Breakout

Flow — repeated for EACH backtest day (as of the previous day's close, so no
future data leaks into the pre-market filters):
  0. Pre-market: run trend direction analysis for Nifty 50 + all top stocks.
                 Only stocks whose trend aligns with the breakout direction are traded.
  1. Score ALL symbols using MomentumScoringService.
  2. Select top TOP_N_STOCKS by composite momentum score.
  3. Fetch 1-minute intraday data for each of those stocks.
  4. For that trading day:
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
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from models.orb_models import BreakoutDirection, OpenRange, ORBTradeResult
from services.momentum_service import MomentumScore, MomentumScoringService
from services.orb_data_service import ORBDataService
from services.trend_direction_service import TrendAnalysis, TrendDirectionService
from strategy.orb_strategy import ORBStrategy, run_premarket_trend_analysis

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

CAPITAL_PER_STOCK = 100_000  # Rs. allocated per trade; qty = floor(capital / entry)
ORB_MINUTES       = 15       # Opening range period: 9:15–9:30 AM
STOP_LOSS_PCT     = 1.5
RISK_REWARD_RATIO = 2.0
START_DATE        = "01-Jul-2026 9:15:00"
END_DATE          = "17-Jul-2026 15:29:59"
INTERVAL          = "1minute" #1minute, 1second

# ── Fair Value Gap (FVG) Entry Confirmation ───────────────────────────────────

ENABLE_FVG_ENTRY = True   # require a confirmed FVG after the ORB breakout before entering
# A bullish FVG forms at candle i when low[i] > high[i-2] (gap zone = high[i-2]…low[i]).
# A bearish FVG forms at candle i when high[i] < low[i-2] (gap zone = high[i]…low[i-2]).
# Confirmation: price retraces into the gap zone and closes back beyond it in the
# breakout direction. Entry is taken at the close of that confirming candle.

# ── Partial Profit Booking / Trailing Stop ────────────────────────────────────

ENABLE_PARTIAL_BOOKING   = True
PARTIAL_BOOK_TRIGGER_PCT = 1.0   # book partial once price moves 1% in favour of entry
PARTIAL_BOOK_FRACTION    = 0.5   # book 50% of the position at the trigger
TRAILING_STOP_PCT        = 1.0   # trail the remaining 50% by this % off the peak
# After the partial is booked the stop on the remaining position is moved to the
# entry price (breakeven), so a full retrace can no longer turn the trade red.

# ── Portfolio Selection ───────────────────────────────────────────────────────

TOP_N_STOCKS     = 15   # Keep top N stocks by momentum score
MAX_DAILY_TRADES = 3    # Maximum trades to take per calendar day

# ── Momentum Scoring ──────────────────────────────────────────────────────────

MIN_MOMENTUM_SCORE     = 50.0
MOMENTUM_LOOKBACK_DAYS = 200  # needs ~157 trading days for Wilder RSI warmup to converge

# ── Trend Filter ──────────────────────────────────────────────────────────────

ENABLE_TREND_FILTER = True
TREND_FILTER_MODE   = "STRICT"
TREND_LOOKBACK_DAYS = 10
HISTORICAL_WEIGHT   = 0.6
INTRADAY_WEIGHT     = 0.4

# ── Nifty 50 benchmark (pre-market context) ───────────────────────────────────

NIFTY50_STOCK_CODE   = "NIFTY"
NIFTY50_EXCHANGE     = "NSE"
ANALYZE_NIFTY50      = True          # set False to skip index analysis

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
    quantity:        int
    capital_used:    float
    pnl:             float
    return_pct:      float
    breakout_time:   str
    momentum_score:  Optional[float] = None
    momentum_rank:   Optional[int]   = None
    partial_exit_price: Optional[float] = None
    partial_quantity:   int             = 0
    fvg_low:            Optional[float] = None
    fvg_high:           Optional[float] = None


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


# ── Fair Value Gap detection & confirmation ───────────────────────────────────


@dataclass
class FVGEntry:
    entry_idx:   int      # index (within the scanned candle list) of the confirming candle
    entry_price: float    # close of the confirming candle
    fvg_low:     float    # bottom of the gap zone
    fvg_high:    float    # top of the gap zone
    fvg_time:    str      # datetime of the candle that completed the gap


def _find_fvg_confirmed_entry(
    direction: BreakoutDirection,
    candles: list[dict],
) -> Optional[FVGEntry]:
    """
    Scan *candles* (starting at the breakout candle) for a fair value gap in the
    breakout direction, then wait for confirmation before returning an entry.

    Bullish FVG at candle i: low[i] > high[i-2]  → zone (high[i-2], low[i]).
    Bearish FVG at candle i: high[i] < low[i-2]  → zone (high[i], low[i-2]).

    Confirmation: a later candle retraces into the gap zone and closes back
    beyond the zone in the breakout direction. If a candle instead closes
    through the far side of the zone the gap is invalidated and the scan
    continues looking for a fresh gap.
    """
    gap: Optional[tuple[float, float, str]] = None  # (fvg_low, fvg_high, fvg_time)

    for i in range(2, len(candles)):
        high_i = float(candles[i]["high"])
        low_i  = float(candles[i]["low"])
        close_i = float(candles[i]["close"])

        if gap is not None:
            fvg_low, fvg_high, fvg_time = gap
            if direction == BreakoutDirection.BUY:
                if close_i < fvg_low:               # gap filled through — invalidated
                    gap = None
                elif low_i <= fvg_high and close_i > fvg_high:
                    return FVGEntry(i, close_i, fvg_low, fvg_high, fvg_time)
            else:
                if close_i > fvg_high:              # invalidated
                    gap = None
                elif high_i >= fvg_low and close_i < fvg_low:
                    return FVGEntry(i, close_i, fvg_low, fvg_high, fvg_time)
            if gap is not None:
                continue

        # No active gap — look for a new one completing at candle i
        high_prev2 = float(candles[i - 2]["high"])
        low_prev2  = float(candles[i - 2]["low"])
        if direction == BreakoutDirection.BUY and low_i > high_prev2:
            gap = (high_prev2, low_i, candles[i]["datetime"])
        elif direction == BreakoutDirection.SELL and high_i < low_prev2:
            gap = (high_i, low_prev2, candles[i]["datetime"])

    return None


# ── Partial booking / breakeven / trailing-stop exit simulation ───────────────


@dataclass
class ExitSimulation:
    pnl_per_share_x_qty: float           # total PnL in Rs. for the given quantity
    final_exit_price:    float           # exit price of the remaining position
    exit_reason:         str
    partial_exit_price:  Optional[float] # price at which the partial was booked
    partial_quantity:    int             # shares booked at the partial


def _simulate_exit_with_partials(
    direction: BreakoutDirection,
    entry: float,
    stop_loss: float,
    quantity: int,
    post_entry_candles: list[dict],
    partial_trigger_pct: float = PARTIAL_BOOK_TRIGGER_PCT,
    partial_fraction: float = PARTIAL_BOOK_FRACTION,
    trailing_stop_pct: float = TRAILING_STOP_PCT,
) -> ExitSimulation:
    """
    Exit model:
      1. Initial stop at *stop_loss* on the full position.
      2. When price moves *partial_trigger_pct* % in favour, book
         *partial_fraction* of the position at that level and move the stop on
         the remainder to the entry price (breakeven).
      3. The remainder then trails: stop = best price ∓ *trailing_stop_pct* %,
         never below breakeven. Exits on trailing stop or at market close.
    Stops are checked before profit triggers within a candle (conservative).
    """
    is_buy = direction == BreakoutDirection.BUY
    sign   = 1.0 if is_buy else -1.0

    partial_qty   = int(quantity * partial_fraction) if quantity >= 2 else 0
    remaining_qty = quantity - partial_qty
    partial_price = round(entry * (1 + sign * partial_trigger_pct / 100), 2)

    partial_done: bool = False
    partial_fill: Optional[float] = None
    stop  = stop_loss
    best  = entry   # best favourable price seen (peak for BUY, trough for SELL)
    pnl   = 0.0

    for candle in post_entry_candles:
        high = float(candle["high"])
        low  = float(candle["low"])

        # 1) Stop check first (conservative intra-candle ordering)
        stopped = (low <= stop) if is_buy else (high >= stop)
        if stopped:
            live_qty = quantity if not partial_done else remaining_qty
            pnl += sign * (stop - entry) * live_qty
            reason = ("trailing_stop" if partial_done else "stop_loss")
            if partial_done and abs(stop - entry) < 1e-9:
                reason = "breakeven"
            return ExitSimulation(round(pnl, 2), stop, reason, partial_fill, partial_qty if partial_done else 0)

        # 2) Partial profit booking at +partial_trigger_pct from entry
        if not partial_done and partial_qty > 0:
            hit = (high >= partial_price) if is_buy else (low <= partial_price)
            if hit:
                partial_done = True
                partial_fill = partial_price
                pnl += sign * (partial_price - entry) * partial_qty
                stop = entry  # move stop to breakeven on the remaining position
                best = partial_price

        # 3) Trail the stop on the remaining position (only after partial)
        if partial_done:
            best = max(best, high) if is_buy else min(best, low)
            trail = best * (1 - sign * trailing_stop_pct / 100)
            stop = max(stop, round(trail, 2)) if is_buy else min(stop, round(trail, 2))

    # Market close — exit whatever is still open at the last close
    last_close = float(post_entry_candles[-1]["close"])
    live_qty = quantity if not partial_done else remaining_qty
    pnl += sign * (last_close - entry) * live_qty
    return ExitSimulation(round(pnl, 2), last_close, "close", partial_fill, partial_qty if partial_done else 0)


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
        min_score=MIN_MOMENTUM_SCORE,
        top_n=top_n,
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

    # score_all_symbols already filtered by min_score and sliced to top_n
    top_structured: list[tuple[str, str, MomentumScore]] = []
    for ms in scores:
        sc, exc = parse_symbol(ms.symbol)
        top_structured.append((sc, exc, ms))

    logger.info(f"Momentum screening selected {len(top_structured)} stocks:")
    for rank, (sc, exc, ms) in enumerate(top_structured, 1):
        logger.info(
            f"  #{rank} {ms.symbol}: Score={ms.composite_score:.1f}/100 "
            f"ROC5d={ms.roc_5d:+.1f}% RSI={ms.rsi_14:.0f} "
            f"Close=Rs.{ms.last_close:.2f}"
        )

    print(f"  Selected {len(top_structured)} stocks for ORB scanning "
          f"(min score >= {MIN_MOMENTUM_SCORE}, top {top_n})")
    print(f"{'='*70}\n")

    return top_structured



# ── Phase 2 & 3: Per-day ORB scan across top stocks ──────────────────────────


def run_portfolio_orb_backtest(
    orb_data_svc: ORBDataService,
    trend_svc: Optional[TrendDirectionService],
    top_stocks: list[tuple[str, str, MomentumScore]],
    hist_trends: Optional[dict[str, TrendAnalysis]] = None,  # pre-computed from run_premarket_trend_analysis
    start_date: str = START_DATE,
    end_date: str = END_DATE,
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
                stock_code, exchange_code, start_date, end_date, INTERVAL
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

    # Use pre-computed trends (from run_premarket_trend_analysis) or fall back to
    # computing them inline when called without pre-market analysis.
    first_date = all_dates[0]
    as_of_date = datetime(first_date.year, first_date.month, first_date.day)
    if hist_trends is None:
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
            if not day_candles:
                continue

            orb_candles      = ORBDataService.get_orb_candles(day_candles, ORB_MINUTES)
            post_orb_candles = ORBDataService.get_post_orb_candles(day_candles, ORB_MINUTES)

            if not orb_candles or not post_orb_candles:
                continue
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

        # Momentum-rank priority: among all breakout candidates, prefer the
        # highest-ranked stocks (rank 1 = best). This matches live FyersORB
        # behaviour where momentum rank determines which trades are placed.
        candidates.sort(key=lambda c: momentum_rank_map.get(c.stock_code, 9999))
        selected = candidates[:MAX_DAILY_TRADES]

        day_pnl    = 0.0
        day_stocks = []

        print(f"  {trade_date}  — {len(candidates)} breakout(s) found, "
              f"taking {len(selected)} (max {MAX_DAILY_TRADES})")

        for cand in selected:
            entry_price = cand.entry
            entry_idx   = cand.breakout_idx
            entry_time  = cand.breakout_time
            fvg_low: Optional[float]  = None
            fvg_high: Optional[float] = None

            # FVG confirmation: only enter once a fair value gap in the breakout
            # direction forms after the breakout and price confirms it.
            if ENABLE_FVG_ENTRY:
                fvg_scan = cand.post_orb_candles[cand.breakout_idx:]
                fvg = _find_fvg_confirmed_entry(cand.direction, fvg_scan)
                if fvg is None:
                    print(f"    [-] {cand.stock_code:<12} skipped — no confirmed FVG "
                          f"after breakout @{cand.breakout_time}")
                    continue
                entry_idx   = cand.breakout_idx + fvg.entry_idx
                entry_price = fvg.entry_price
                entry_time  = cand.post_orb_candles[entry_idx]["datetime"]
                fvg_low, fvg_high = fvg.fvg_low, fvg.fvg_high

            target, stop_loss = _compute_levels(
                cand.direction, entry_price, STOP_LOSS_PCT, RISK_REWARD_RATIO
            )

            # Fixed-capital position sizing: Rs. CAPITAL_PER_STOCK per trade
            quantity = int(CAPITAL_PER_STOCK // entry_price)
            if quantity < 1:
                print(f"    [-] {cand.stock_code:<12} skipped — entry {entry_price:.2f} "
                      f"exceeds capital Rs.{CAPITAL_PER_STOCK:,}")
                continue
            capital_used = round(quantity * entry_price, 2)

            remaining = cand.post_orb_candles[entry_idx + 1:]
            partial_exit_price: Optional[float] = None
            partial_quantity = 0

            if not remaining:
                exit_price  = float(cand.post_orb_candles[entry_idx]["close"])
                exit_reason = "close"
                if cand.direction == BreakoutDirection.BUY:
                    pnl = round((exit_price - entry_price) * quantity, 2)
                else:
                    pnl = round((entry_price - exit_price) * quantity, 2)
            elif ENABLE_PARTIAL_BOOKING:
                sim = _simulate_exit_with_partials(
                    cand.direction, entry_price, stop_loss, quantity, remaining
                )
                exit_price         = sim.final_exit_price
                exit_reason        = sim.exit_reason
                pnl                = sim.pnl_per_share_x_qty
                partial_exit_price = sim.partial_exit_price
                partial_quantity   = sim.partial_quantity
            else:
                exit_price, exit_reason = _simulate_exit(
                    cand.direction, entry_price, target, stop_loss, remaining
                )
                if cand.direction == BreakoutDirection.BUY:
                    pnl = round((exit_price - entry_price) * quantity, 2)
                else:
                    pnl = round((entry_price - exit_price) * quantity, 2)

            return_pct = round(pnl / capital_used * 100, 2)

            day_pnl += pnl
            day_stocks.append(cand.stock_code)

            bt_time   = datetime.fromisoformat(entry_time).strftime("%H:%M")
            dir_label = "BUY " if cand.direction == BreakoutDirection.BUY else "SELL"
            pnl_sign  = "+" if pnl >= 0 else ""
            rank      = momentum_rank_map.get(cand.stock_code, "?")
            score     = momentum_score_map.get(cand.stock_code, 0.0)
            fvg_label = (f"  FVG[{fvg_low:.2f}–{fvg_high:.2f}]"
                         if fvg_low is not None else "")
            partial_label = (f"  Partial {partial_quantity}@{partial_exit_price:.2f}"
                             if partial_exit_price is not None else "")
            print(
                f"    [{rank}] {cand.stock_code:<12} {dir_label} @{bt_time}"
                f"  ORB[{cand.orb.low:.2f}–{cand.orb.high:.2f}]{fvg_label}"
                f"  Entry {entry_price:.2f} x{quantity}  SL {stop_loss:.2f}"
                f"{partial_label}"
                f"  Exit {exit_price:.2f} [{exit_reason:10s}]"
                f"  PnL {pnl_sign}{pnl:.2f} ({pnl_sign}{return_pct:.2f}%)"
                f"  Mom:{score:.0f}"
            )

            all_results.append(PortfolioTradeResult(
                stock_code    = cand.stock_code,
                trade_date    = trade_date,
                direction     = cand.direction,
                orb_high      = cand.orb.high,
                orb_low       = cand.orb.low,
                entry_price   = entry_price,
                target        = target,
                stop_loss     = stop_loss,
                exit_price    = exit_price,
                exit_reason   = exit_reason,
                quantity      = quantity,
                capital_used  = capital_used,
                pnl           = pnl,
                return_pct    = return_pct,
                breakout_time = cand.breakout_time,
                momentum_score = momentum_score_map.get(cand.stock_code),
                momentum_rank  = momentum_rank_map.get(cand.stock_code),
                partial_exit_price = partial_exit_price,
                partial_quantity   = partial_quantity,
                fvg_low            = fvg_low,
                fvg_high           = fvg_high,
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
    print(f"  Max trades/day: {MAX_DAILY_TRADES}  |  Capital/trade: Rs.{CAPITAL_PER_STOCK:,}")
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

    total_capital = round(sum(r.capital_used for r in results), 2)
    overall_ret   = round(total / total_capital * 100, 2) if total_capital else 0.0
    ret_sign      = "+" if overall_ret >= 0 else ""
    print(f"  Capital deployed (sum of trades): Rs.{total_capital:,.2f}")
    print(f"  Overall return on deployed capital: {ret_sign}{overall_ret:.2f}%")
    print(f"\n{'='*72}\n")


def save_csv(results: list[PortfolioTradeResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Date", "Stock", "Direction", "Momentum Rank", "Momentum Score",
            "ORB High", "ORB Low", "Entry", "Quantity", "Capital Used",
            "Target", "Stop Loss",
            "FVG Low", "FVG High",
            "Partial Exit Price", "Partial Qty",
            "Exit Price", "Exit Reason", "PnL", "Return %", "Breakout Time",
        ])
        for r in results:
            writer.writerow([
                r.trade_date, r.stock_code, r.direction.value,
                r.momentum_rank, r.momentum_score,
                r.orb_high, r.orb_low, r.entry_price,
                r.quantity, r.capital_used,
                r.target, r.stop_loss,
                r.fvg_low, r.fvg_high,
                r.partial_exit_price, r.partial_quantity,
                r.exit_price, r.exit_reason,
                r.pnl, r.return_pct, r.breakout_time,
            ])
    print(f"  CSV saved → {path}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__" or True:
    unique_symbols = deduplicate(SYMBOLS)

    momentum_svc = MomentumScoringService(breeze)
    orb_data_svc = ORBDataService(breeze)
    trend_svc    = TrendDirectionService(breeze) if ENABLE_TREND_FILTER else None

    start_dt = datetime.strptime(START_DATE, "%d-%b-%Y %H:%M:%S")
    end_dt   = datetime.strptime(END_DATE,   "%d-%b-%Y %H:%M:%S")

    all_results: list[PortfolioTradeResult] = []

    # Run the full pre-market pipeline (momentum scoring + trend direction
    # analysis) fresh for EVERY backtest day, exactly as FyersORB does live
    # each morning — not just once for the whole window.
    current = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    while current <= end_dt:
        trade_day = current
        current  += timedelta(days=1)

        if trade_day.weekday() >= 5:  # skip Sat/Sun
            continue

        # Use end of the previous calendar day (23:59:59) so the Breeze API does
        # not return a same-day daily candle whose timestamp is midnight of the
        # trade day — that candle has a stale/incorrect close and shifts ROC/RSI.
        # FyersORB runs at ~09:10 so its to_date naturally falls before any
        # same-day candle; this mirrors that behaviour for each backtest day.
        as_of_date = trade_day - timedelta(seconds=1)

        day_start = trade_day.strftime("%d-%b-%Y") + " 9:15:00"
        day_end   = trade_day.strftime("%d-%b-%Y") + " 15:29:59"

        print(f"\n{'#'*70}")
        print(f"#  BACKTEST DAY: {trade_day.date()}")
        print(f"{'#'*70}")

        # Phase 1: Score all, select top N (as of the previous day's close)
        top_stocks = score_all_and_select_top(
            momentum_svc=momentum_svc,
            symbols=unique_symbols,
            as_of_date=as_of_date,
            top_n=TOP_N_STOCKS,
        )

        if not top_stocks:
            print(f"  {trade_day.date()}: no stocks passed the momentum filter — skipping day.")
            continue

        # Phase 1b: Pre-market trend direction analysis (mirrors FyersORB)
        hist_trends: Optional[dict] = None
        if trend_svc and ENABLE_TREND_FILTER:
            hist_trends = run_premarket_trend_analysis(
                trend_svc         = trend_svc,
                top_stocks        = top_stocks,
                as_of_date        = as_of_date,
                trend_lookback_days = TREND_LOOKBACK_DAYS,
                analyze_nifty     = ANALYZE_NIFTY50,
                nifty_stock_code  = NIFTY50_STOCK_CODE,
                nifty_exchange    = NIFTY50_EXCHANGE,
            )

        # Phase 2 & 3: ORB scan with daily cap for this day only
        day_results = run_portfolio_orb_backtest(
            orb_data_svc = orb_data_svc,
            trend_svc    = trend_svc,
            top_stocks   = top_stocks,
            hist_trends  = hist_trends,
            start_date   = day_start,
            end_date     = day_end,
        )
        all_results.extend(day_results)

    if not all_results:
        print("No trades executed across the backtest window.")
    print_final_report(all_results)
    save_csv(all_results, REPORT_CSV)
