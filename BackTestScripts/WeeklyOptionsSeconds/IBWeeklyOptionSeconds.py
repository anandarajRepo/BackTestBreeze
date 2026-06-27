"""
Nifty 50 Weekly Options — Initial Balance (IB) Backtest  (1-second data)
========================================================================

The Initial Balance (IB) is the price range established during the very first
part of the trading session — by default the first hour (9:15 → 10:15 for the
NSE). The opening hour carries the heaviest volume and sets the tone for the day.
Once the IB window closes, its high and low are locked and used as the key
intraday boundaries. Historically the price only breaks ONE side of the IB in
~70-85% of sessions, which is the statistical edge behind both approaches.

This runs the IB system on each option contract's OWN price series (resampled to
N-second candles). Buying an option is an inherently long/directional bet, so the
strategy is always LONG the premium — a CE trade is bullish on the index, a PE
trade is bearish on the index. The two classic IB approaches are mapped onto the
option's own price as follows:

Initial Balance:
  For each trading day the IB high/low and the average IB-candle volume are built
  from the first IB_MINUTES of trading (from 9:15).

1. BREAKOUT approach (momentum continuation):
     Buy when an option candle CLOSES firmly above its IB high — clearing it by at
     least BREAKOUT_BUFFER_PCT percent — with good volume (breakout candle volume
     >= VOLUME_FACTOR × average IB volume). Optionally (RETEST_ENABLED) wait for a
     pullback that retests the IB high before entering.

2. REVERSAL approach (failed breakdown / bear trap):
     Buy when the premium dips below its IB low but fails to hold and CLOSES back
     above the IB low, rotating back up into the range.

  Use STRATEGY_MODE to trade "breakout", "reversal" or "both".

Exit:
  - Target  : entry + risk × RISK_REWARD_RATIO  (risk = entry − stop)
  - Stop-loss: STOP_LOSS_PCT percent below entry
  - Trailing stop-loss (optional; see TRAILING_STOP_* config)
  - Break-even stop: once price moves BREAKEVEN_TRIGGER_PCT percent above entry,
    the stop-loss is moved up to the entry price. When
    BREAKEVEN_PARTIAL_BOOK_ENABLED is True, 50% of the position is also booked
    while the remaining 50% continues to run (see BREAKEVEN_* config)
  - Square-off at 15:20 IST
  - No new entries before the IB window completes or after 14:45
  - Max MAX_TRADES_PER_DAY trades per day per symbol

Usage:
  1. Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  2. Set RESAMPLE_SECONDS to the desired candle size for the strategy:
       1  → raw 1-second bars (no resampling)
       5  → 5-second bars, 10/15/30/45/60 → larger bars (any positive integer).
  Data is always fetched as 1-second bars from Breeze; resampling is done
  locally before the IB logic runs.
"""

import os
from datetime import date

import pandas as pd
from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.ib_option_seconds_strategy import IBOptionSecondsStrategy

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

# Initial Balance length, in minutes from 9:15. The default of 60 captures the
# first trading hour (9:15 → 10:15), the classic IB definition.
IB_MINUTES        = 60

# Which IB approach(es) to trade:
#   "breakout" → only the firm break above the IB high (momentum continuation)
#   "reversal" → only the failed-breakdown trap below the IB low (rotation back up)
#   "both"     → take whichever signal fires first (governed by MAX_TRADES_PER_DAY)
STRATEGY_MODE     = "both"

# Volume confirmation for entries. The entry candle's volume must be at least
# VOLUME_FACTOR times the average Initial Balance volume for the signal to be
# taken. 1.0 = at least average volume; >1.0 demands a "good"/surging volume.
# Set to 0 to disable the volume filter.
VOLUME_FACTOR     = 1.5

# A BREAKOUT candle must clear the IB high by at least this percent to count as a
# *firm* break ("breaks firmly above the IB high"). 0 = any close above the IB
# high qualifies. e.g. 0.5 requires the close to be at least 0.5% above the IB high.
BREAKOUT_BUFFER_PCT = 0.0

# Retest confirmation for breakouts. When True, the strategy does not enter on the
# breakout candle itself; instead it waits for a slight pullback that retests the
# IB high (price comes back to within RETEST_TOLERANCE_PCT percent of the IB high)
# and then resumes upward before entering — mirroring the common practice of
# waiting for a pullback to the broken boundary. When False (default), the entry is
# taken on the firm breakout candle.
RETEST_ENABLED       = False
RETEST_TOLERANCE_PCT = 2.0

# Hard stop-loss as a percentage below the entry price, and target as a multiple
# of the stop distance (risk:reward). Per the IB playbook, stops sit just inside
# the breakout range and targets run 1.5–2× the initial risk.
STOP_LOSS_PCT     = 25.0
RISK_REWARD_RATIO = 2.0

# Maximum number of entries per day per option contract.
MAX_TRADES_PER_DAY = 5

# Trailing stop-loss. When enabled, the position is closed if the option price
# falls TRAILING_STOP_PCT percent below the highest price reached since entry
# (the stop ratchets up with the peak, never down).
TRAILING_STOP_ENABLED = True
TRAILING_STOP_PCT     = 5.0

# Break-even stop. When enabled, once the option price moves BREAKEVEN_TRIGGER_PCT
# percent above the entry price, the stop-loss is moved up to the entry price so
# the trade can no longer turn into a loss.
BREAKEVEN_ENABLED     = True
BREAKEVEN_TRIGGER_PCT = 5.0

# Partial profit booking at break-even. When True, 50% of the position is booked
# at the moment the stop-loss is moved up to the entry price, while the remaining
# 50% continues to run. When False, the stop-loss is still moved to break-even but
# the full position is held on. Only applies when BREAKEVEN_ENABLED is True.
BREAKEVEN_PARTIAL_BOOK_ENABLED = True

# Always fetch raw 1-second bars from Breeze; resampling is done locally.
INTERVAL          = "1second"

# Candle size (in seconds) used for the strategy. Set to 1 for raw 1-second bars.
RESAMPLE_SECONDS  = 5

# Print the final resampled DataFrame alongside the trades for each contract.
PRINT_RESAMPLED   = False

# When True, candle data is served ONLY from the local cache — no Breeze API
# calls are made. Expiries whose data is not cached are skipped.
CACHE_ONLY        = True

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

    strategy = IBOptionSecondsStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        ib_minutes=IB_MINUTES,
        strategy_mode=STRATEGY_MODE,
        volume_factor=VOLUME_FACTOR,
        breakout_buffer_pct=BREAKOUT_BUFFER_PCT,
        retest_enabled=RETEST_ENABLED,
        retest_tolerance_pct=RETEST_TOLERANCE_PCT,
        stop_loss_pct=STOP_LOSS_PCT,
        risk_reward_ratio=RISK_REWARD_RATIO,
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
    IBOptionSecondsStrategy.print_report(expiry_results)

    # ── Trades DataFrame with Initial Balance range and IB-range percentage ────
    # The IB range is the width of the Initial Balance (IB high − IB low) for the
    # day on which the trade was taken. The IB-range percentage expresses that
    # width relative to the IB low, i.e. (IB high − IB low) / IB low × 100.
    all_trades = [t for er in expiry_results for t in er.all_trades]

    if all_trades:
        rows = []
        for t in sorted(all_trades, key=lambda x: x.entry_time):
            ib_range = round(t.ib_high - t.ib_low, 2)
            ib_range_pct = (
                round(ib_range / t.ib_low * 100, 2) if t.ib_low else 0.0
            )
            rows.append({
                "symbol":         t.symbol,
                "option_type":    t.option_type,
                "strike":         t.strike,
                "expiry_date":    t.expiry_date,
                "entry_time":     t.entry_time,
                "exit_time":      t.exit_time,
                "entry_mode":     t.entry_mode,
                "entry_price":    t.entry_price,
                "exit_price":     t.exit_price,
                "shares":         t.shares,
                "pnl":            t.pnl,
                "exit_reason":    t.exit_reason,
                "ib_high":        t.ib_high,
                "ib_low":         t.ib_low,
                "ib_range":       ib_range,
                "ib_range_pct":   ib_range_pct,
                "volume_ratio":   t.volume_ratio,
                "duration_minutes": t.duration_minutes,
            })

        trades_df = pd.DataFrame(rows)

        print(f"\n{'='*96}")
        print("  TRADES WITH INITIAL BALANCE RANGE & IB-RANGE PERCENTAGE")
        print(f"{'='*96}")
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", None,
        ):
            print(trades_df.to_string(index=False))
    else:
        print("\n  No trades to display in DataFrame.")
