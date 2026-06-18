"""
Nifty 50 Weekly Options — Fibonacci Retracement Backtest  (1-second data)
=========================================================================

Strategy:
  An algorithmic Fibonacci-retracement system run directly on each option
  contract's own price series. Buying an option is an inherently long bet, so
  only bullish pullback entries are traded (a CE long = bullish index, a PE
  long = bearish index).

  For every candle the algorithm:
    1. Detects the trend with a moving average (uptrend = price above the MA).
    2. Identifies the recent Swing High / Swing Low over a rolling window.
    3. Computes the uptrend retracement levels (38.2% / 50% / 61.8% / 78.6%).

Entry:
  A LONG is taken when the price corrects into the Golden Zone (pulls back to at
  least the 50% level but holds above the 78.6% level) AND confirmation fires:
    - Trend  : price above the moving average.
    - Momentum: RSI bounces back above the oversold threshold.
    - Candle : optional bullish (close > open) candle confirmation.
    - Volume : confluence — volume above VOLUME_FACTOR × recent average volume.

Exit:
  - Stop-loss   : just beyond the 78.6% level (fallback: STOP_LOSS_PCT below entry)
  - Profit target: a Fibonacci extension (EXTENSION_RATIO, e.g. 127.2% / 161.8%)
  - Trailing stop-loss (optional; see TRAILING_STOP_* config)
  - Break-even stop with optional 50% partial booking (see BREAKEVEN_* config)
  - Square-off at 15:20 IST
  - No new entries before the warm-up completes or after 14:45
  - Max MAX_TRADES_PER_DAY trades per day per symbol

Usage:
  1. Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  2. Set RESAMPLE_SECONDS to the desired candle size (1 = raw 1-second bars).
  Data is always fetched as 1-second bars from Breeze; resampling is local.
"""

import os
from datetime import date

import pandas as pd
from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.fibonacci_option_seconds_strategy import FibonacciOptionSecondsStrategy

load_dotenv()

# ── Session ───────────────────────────────────────────────────────────────────

breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN"),
)
print("Session Generated Successfully\n")

# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE        = "01-Jan-2026"   # format: DD-Mon-YYYY
END_DATE          = "16-Jun-2026"   # format: DD-Mon-YYYY

CAPITAL           = 100_000.0       # capital per contract (used for position sizing)

# ── Trend detection ─────────────────────────────────────────────────────────
# Moving-average period (in candles). An entry is only taken while the option
# price trades above this moving average (an established uptrend).
MA_PERIOD         = 50

# ── Swing identification ────────────────────────────────────────────────────
# Rolling window (in candles) over which the recent Swing High / Swing Low of
# the move are identified.
SWING_LOOKBACK    = 60

# ── Fibonacci levels ────────────────────────────────────────────────────────
# Retracement ratio the entry is taken at (deep end of the Golden Zone), the
# shallow end of the Golden Zone, the level the stop sits just beyond (78.6%),
# and the profit-target extension ratio (127.2% / 161.8%).
ENTRY_RATIO       = 0.618
GOLDEN_ZONE_START = 0.5
STOP_RATIO        = 0.786
EXTENSION_RATIO   = 1.272

# ── Momentum confirmation (RSI) ─────────────────────────────────────────────
RSI_PERIOD        = 14
RSI_OVERSOLD      = 30.0
# When True, RSI must CROSS back above the oversold level on the entry candle;
# when False, RSI simply needs to be above it.
RSI_CROSS_REQUIRED = True
# When True, additionally require a bullish (close > open) entry candle.
REQUIRE_BULLISH_CANDLE = True

# ── Volume confluence ───────────────────────────────────────────────────────
# The entry candle's volume must be at least VOLUME_FACTOR times the trailing
# average volume over VOLUME_AVG_PERIOD candles. Set VOLUME_FACTOR to 0 to
# disable the volume filter.
VOLUME_FACTOR     = 1.5
VOLUME_AVG_PERIOD = 60

# Hard stop-loss fallback (percent below entry) used when the 78.6% Fibonacci
# stop is not strictly below the entry price.
STOP_LOSS_PCT     = 25.0

# Maximum number of entries per day per option contract.
MAX_TRADES_PER_DAY = 5

# ── Trailing / break-even stops ─────────────────────────────────────────────
TRAILING_STOP_ENABLED = True
TRAILING_STOP_PCT     = 5.0

BREAKEVEN_ENABLED     = True
BREAKEVEN_TRIGGER_PCT = 5.0
BREAKEVEN_PARTIAL_BOOK_ENABLED = True

# Always fetch raw 1-second bars from Breeze; resampling is done locally.
INTERVAL          = "1second"

# Candle size (in seconds) used for the strategy. Set to 1 for raw 1-second bars.
RESAMPLE_SECONDS  = 30

# Print the final resampled DataFrame alongside the trades for each contract.
PRINT_RESAMPLED   = False

# When True, candle data is served ONLY from the local cache — no Breeze API
# calls are made. Expiries whose data is not cached are skipped.
CACHE_ONLY        = False

# When True, a fresh ATM strike is chosen for EACH trading day from that day's
# Nifty 9:15 open. When False, a single ATM strike anchored to the week's Monday
# open is traded across the whole expiry window.
PER_DAY_ATM       = True

# NSE market holidays for 2026. When a Tuesday weekly expiry falls on one of
# these dates, the expiry is rolled back to the previous trading day.
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

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nifty_service = NiftyOptionService(breeze)

    strategy = FibonacciOptionSecondsStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        ma_period=MA_PERIOD,
        swing_lookback=SWING_LOOKBACK,
        rsi_period=RSI_PERIOD,
        rsi_oversold=RSI_OVERSOLD,
        rsi_cross_required=RSI_CROSS_REQUIRED,
        require_bullish_candle=REQUIRE_BULLISH_CANDLE,
        entry_ratio=ENTRY_RATIO,
        golden_zone_start=GOLDEN_ZONE_START,
        stop_ratio=STOP_RATIO,
        extension_ratio=EXTENSION_RATIO,
        volume_factor=VOLUME_FACTOR,
        volume_avg_period=VOLUME_AVG_PERIOD,
        stop_loss_pct=STOP_LOSS_PCT,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        resample_seconds=RESAMPLE_SECONDS,
        max_trades_per_day=MAX_TRADES_PER_DAY,
        print_resampled=PRINT_RESAMPLED,
        cache_only=CACHE_ONLY,
        market_holidays=MARKET_HOLIDAYS,
        per_day_atm=PER_DAY_ATM,
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_pct=TRAILING_STOP_PCT,
        breakeven_enabled=BREAKEVEN_ENABLED,
        breakeven_trigger_pct=BREAKEVEN_TRIGGER_PCT,
        breakeven_partial_book_enabled=BREAKEVEN_PARTIAL_BOOK_ENABLED,
    )

    expiry_results = strategy.run_weekly_backtest()
    FibonacciOptionSecondsStrategy.print_report(expiry_results)

    # ── Trades DataFrame with swing range and Fibonacci-entry details ──────────
    # The swing range is the width of the identified swing (swing high − swing
    # low) at the time the trade was taken. The swing-range percentage expresses
    # that width relative to the swing low.
    all_trades = [t for er in expiry_results for t in er.all_trades]

    if all_trades:
        rows = []
        for t in sorted(all_trades, key=lambda x: x.entry_time):
            swing_range = round(t.swing_high - t.swing_low, 2)
            swing_range_pct = (
                round(swing_range / t.swing_low * 100, 2) if t.swing_low else 0.0
            )
            rows.append({
                "symbol":          t.symbol,
                "option_type":     t.option_type,
                "strike":          t.strike,
                "expiry_date":     t.expiry_date,
                "entry_time":      t.entry_time,
                "exit_time":       t.exit_time,
                "entry_price":     t.entry_price,
                "exit_price":      t.exit_price,
                "shares":          t.shares,
                "pnl":             t.pnl,
                "exit_reason":     t.exit_reason,
                "swing_high":      t.swing_high,
                "swing_low":       t.swing_low,
                "swing_range":     swing_range,
                "swing_range_pct": swing_range_pct,
                "fib_entry_level": t.fib_entry_level,
                "fib_ratio":       t.fib_ratio,
                "rsi_at_entry":    t.rsi_at_entry,
                "volume_ratio":    t.volume_ratio,
                "duration_minutes": t.duration_minutes,
            })

        trades_df = pd.DataFrame(rows)

        print(f"\n{'='*90}")
        print("  TRADES WITH SWING RANGE & FIBONACCI-ENTRY DETAILS")
        print(f"{'='*90}")
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", None,
        ):
            print(trades_df.to_string(index=False))
    else:
        print("\n  No trades to display in DataFrame.")
