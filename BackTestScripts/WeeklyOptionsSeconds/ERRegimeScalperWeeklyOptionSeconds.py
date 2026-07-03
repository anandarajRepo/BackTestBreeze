"""
Nifty 50 Weekly Options — Xiznit ER Regime Scalper Backtest  (1-second data)
============================================================================

Strategy:
  An options adaptation of the "Xiznit ER Regime Scalper" — a momentum scalping
  system originally built for the 2-minute timeframe on futures. The trading
  signal is computed on the Nifty 50 spot series (resampled to RESAMPLE_SECONDS
  candles) and the resulting bias is expressed by buying the ATM option:

    * bullish / green (Uptrend) regime  → buy ATM CE  (long)
    * bearish / red   (Downtrend) regime → buy ATM PE  (short)

Regime filter (Kaufman Efficiency Ratio):
  Every bar is classed into Uptrend (green), Downtrend (red), Chop (orange) or
  Consolidation (grey). Entries are taken only on the FIRST qualifying candle
  after the market transitions into a trending state from a non-trending state;
  a full regime reset is required before the next entry.

Entry:
  On a fresh trend transition, the configured ENTRY_MODE alignment (VWAP / dual
  EMA / fresh-cross / pullback / regime-only / full-filter) plus any enabled
  entry filters must pass. CE is bought on green, PE on red.

Exit:
  - Take-profit / stop-loss in ticks (TP_TICKS / SL_TICKS × TICK_SIZE)
  - Optional move-to-breakeven once BREAKEVEN_TICKS in favour
  - Immediate flatten when the regime shifts away from the trade direction
  - EOD flatten at EOD_FLATTEN_TIME (IST)

Usage:
  1. Set START_DATE / END_DATE (DD-Mon-YYYY) for the backtest window.
  2. Set RESAMPLE_SECONDS to the strategy candle size (120 ≈ the original
     2-minute timeframe). Data is fetched as 1-second bars and resampled
     locally before the regime logic runs.
  3. Pick ENTRY_MODE and toggle the entry filters below.
"""

import os
from datetime import date, time

import pandas as pd
from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.er_regime_scalper_option_seconds_strategy import (
    ERRegimeScalperOptionSecondsStrategy,
)

load_dotenv()

# ── Session ─────────────────────────────────────────────────────────────────

breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN"),
)
print("Session Generated Successfully\n")

# ── Configuration ───────────────────────────────────────────────────────────

START_DATE = "01-Jun-2026"   # format: DD-Mon-YYYY
END_DATE   = "30-Jun-2026"   # format: DD-Mon-YYYY

CAPITAL = 100_000.0          # capital per contract (used for position sizing)

# ── Efficiency Ratio regime filter ──
ER_PERIOD               = 10     # Kaufman ER lookback (in candles)
ER_TREND_THRESHOLD      = 0.40   # ER at/above this = trending (green/red)
ER_STRONG_THRESHOLD     = 0.60   # used when STRONGER_ER filter is enabled
CONSOLIDATION_RANGE_PCT = 0.15   # non-trend split: range% ≥ this = chop, else consolidation

# ── Dual moving averages ──
EMA_FAST = 9
EMA_SLOW = 21

# ── Entry mode ──
# One of: full_filter | vwap_only | ema_only | regime_only | fresh_cross | pullback
ENTRY_MODE       = "full_filter"
FRESH_CROSS_BARS = 5     # for fresh_cross mode: crossover must be within N bars
PULLBACK_BARS    = 5     # for pullback mode: price must touch fast MA within N bars

# ── Entry filters (toggle independently) ──
BLOCK_FIRST_MINUTES     = True   # block the first FIRST_MINUTES of the session
FIRST_MINUTES           = 20
REQUIRE_PRIOR_ALIGNMENT = False  # mode alignment must also hold on the prior bar
MIN_BODY_ENABLED        = False  # skip doji/indecision signal candles
MIN_BODY_POINTS         = 2.0    # minimum signal-candle body (index points)
REQUIRE_MA_SLOPE        = False  # both MAs rising (long) / falling (short)
REQUIRE_PRIOR_BAR_BREAK = False  # close above prior high (long) / below prior low (short)
BLOCK_LUNCH             = False  # block entries inside the lunch window
LUNCH_START             = time(12, 0)
LUNCH_END               = time(13, 0)
STRONGER_ER             = False  # require ER ≥ ER_STRONG_THRESHOLD for entry

# ── Trade management ──
TICK_SIZE         = 0.05   # option-premium tick size
TP_TICKS          = 100    # take-profit distance in ticks
SL_TICKS          = 100    # stop-loss distance in ticks (1:1 default)
BREAKEVEN_ENABLED = True    # move SL to entry once BREAKEVEN_TICKS in favour
BREAKEVEN_TICKS   = 50
EOD_FLATTEN_TIME  = time(15, 20)   # flatten all open positions (IST)
MAX_TRADES_PER_DAY = 20

# Always fetch raw 1-second bars from Breeze; resampling is done locally.
INTERVAL         = "1second"
# Candle size (in seconds) used by the strategy. 120 ≈ the original 2-minute TF.
RESAMPLE_SECONDS = 5

PRINT_TRADES = False

# When True, candle data is served ONLY from the local cache — no Breeze API
# calls are made. Expiries whose data is not cached are skipped.
CACHE_ONLY = True

# When True, a fresh ATM strike is chosen for EACH trading day; when False a
# single ATM strike anchored to the week's Monday open is traded all week.
PER_DAY_ATM = True

# NSE market holidays for 2026. A Tuesday weekly expiry on one of these dates is
# rolled back to the previous trading day.
MARKET_HOLIDAYS = {
    date(2026, 1, 26),   # Republic Day            (Monday)
    date(2026, 3, 3),    # Holi                    (Tuesday)
    date(2026, 3, 26),   # Shri Ram Navami         (Thursday)
    date(2026, 3, 31),   # Shri Mahavir Jayanti    (Tuesday)
    date(2026, 4, 3),    # Good Friday             (Friday)
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti    (Tuesday)
    date(2026, 5, 1),    # Maharashtra Day         (Friday)
    date(2026, 5, 28),   # Bakri Id (Eid-al-Adha)  (Thursday)
    date(2026, 6, 26),   # Moharram                (Friday)
    date(2026, 8, 15),   # Independence Day        (Saturday)
    date(2026, 9, 14),   # Ganesh Chaturthi        (Monday)
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti  (Friday)
    date(2026, 11, 8),   # Diwali Laxmi Pujan      (Sunday)
    date(2026, 11, 10),  # Diwali Balipratipada    (Tuesday)
    date(2026, 11, 24),  # Guru Nanak Jayanti      (Tuesday)
    date(2026, 12, 25),  # Christmas               (Friday)
}

# ── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nifty_service = NiftyOptionService(breeze)

    strategy = ERRegimeScalperOptionSecondsStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        er_period=ER_PERIOD,
        er_trend_threshold=ER_TREND_THRESHOLD,
        er_strong_threshold=ER_STRONG_THRESHOLD,
        consolidation_range_pct=CONSOLIDATION_RANGE_PCT,
        ema_fast=EMA_FAST,
        ema_slow=EMA_SLOW,
        entry_mode=ENTRY_MODE,
        fresh_cross_bars=FRESH_CROSS_BARS,
        pullback_bars=PULLBACK_BARS,
        block_first_minutes=BLOCK_FIRST_MINUTES,
        first_minutes=FIRST_MINUTES,
        require_prior_alignment=REQUIRE_PRIOR_ALIGNMENT,
        min_body_enabled=MIN_BODY_ENABLED,
        min_body_points=MIN_BODY_POINTS,
        require_ma_slope=REQUIRE_MA_SLOPE,
        require_prior_bar_break=REQUIRE_PRIOR_BAR_BREAK,
        block_lunch=BLOCK_LUNCH,
        lunch_start=LUNCH_START,
        lunch_end=LUNCH_END,
        stronger_er=STRONGER_ER,
        tick_size=TICK_SIZE,
        tp_ticks=TP_TICKS,
        sl_ticks=SL_TICKS,
        breakeven_enabled=BREAKEVEN_ENABLED,
        breakeven_ticks=BREAKEVEN_TICKS,
        eod_flatten_time=EOD_FLATTEN_TIME,
        max_trades_per_day=MAX_TRADES_PER_DAY,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        resample_seconds=RESAMPLE_SECONDS,
        print_trades=PRINT_TRADES,
        cache_only=CACHE_ONLY,
        market_holidays=MARKET_HOLIDAYS,
        per_day_atm=PER_DAY_ATM,
    )

    expiry_results = strategy.run_weekly_backtest()
    ERRegimeScalperOptionSecondsStrategy.print_report(expiry_results)

    # ── Trades DataFrame ──────────────────────────────────────────────────
    all_trades = [t for er in expiry_results for t in er.all_trades]

    if all_trades:
        rows = []
        for t in sorted(all_trades, key=lambda x: x.entry_time):
            rows.append({
                "symbol":           t.symbol,
                "option_type":      t.option_type,
                "direction":        t.direction,
                "strike":           t.strike,
                "expiry_date":      t.expiry_date,
                "entry_time":       t.entry_time,
                "exit_time":        t.exit_time,
                "entry_price":      t.entry_price,
                "exit_price":       t.exit_price,
                "shares":           t.shares,
                "pnl":              t.pnl,
                "exit_reason":      t.exit_reason,
                "entry_mode":       t.entry_mode,
                "regime_at_entry":  t.regime_at_entry,
                "er_at_entry":      t.er_at_entry,
                "duration_minutes": t.duration_minutes,
            })

        trades_df = pd.DataFrame(rows)

        print(f"\n{'='*96}")
        print("  TRADES")
        print(f"{'='*96}")
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", None,
        ):
            print(trades_df.to_string(index=False))
    else:
        print("\n  No trades to display in DataFrame.")
