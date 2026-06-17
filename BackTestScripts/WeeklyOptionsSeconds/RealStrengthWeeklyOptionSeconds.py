"""
Nifty 50 Weekly Options — Real Strength Histogram Backtest  (1-second data)
============================================================================

A momentum-continuation scalping strategy. Signals are generated from Nifty 50
spot candles (resampled to N-second bars) using a single composite oscillator —
the Real Strength histogram — combined with a dual-SMA trend filter. Trades are
executed on the ATM CE / PE option contract for the expiry week.

Real Strength histogram:
  EMA( ROC × max(vol/vol_MA, floor) × max(ADX/20, floor) )
  - ROC (price momentum) supplies the sign / direction
  - volume ratio and ADX act as amplifiers (magnitude = "real" conviction)
  A high reading requires price, participation AND trend strength to align.

Entry (Long → CE) — every condition true on the same bar:
  - Histogram > STRENGTH_THRESHOLD
  - Histogram rising vs previous bar
  - ADX >= MIN_ADX
  - DI+ > DI-
  - Volume ratio >= VOL_RATIO_MIN
  - Optional: SMA(fast) > SMA(slow)        (USE_SMA_FILTER)
Short entries (→ PE) mirror these on the bearish side.

Exit — regime-dependent:
  - Static stop loss (STOP_LOSS_PCT) is always active and overrides both exit
    modes if hit first
  - Min hold of MIN_BARS_HOLD bars before peak/flip exits can fire
  - While SMA still confirms the trade → exit only when the histogram crosses
    into the opposite zone past ±FLIP_THRESHOLD (lets winners ride pullbacks)
  - After SMA reverses against the trade → exit when the histogram falls
    PEAK_DROP_PCT% from its peak value reached during the trade
  - Square-off at 15:20 IST

Re-entry lock: after a stop loss, no re-entry in the same direction until the
histogram returns to (or crosses) zero.

Usage:
  1. Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  2. Set RESAMPLE_SECONDS to the desired candle size for the strategy:
       1  → raw 1-second bars (no resampling)
       5  → 5-second bars, 10/15/30/60/300 → larger bars (any positive integer).
  Data is always fetched as 1-second bars from Breeze; resampling is done
  locally before the indicators and strategy logic run.
"""

import os
from datetime import date

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.real_strength_option_seconds_strategy import (
    RealStrengthOptionSecondsStrategy,
)

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
END_DATE          = "09-Jun-2026"   # format: DD-Mon-YYYY

CAPITAL           = 100_000.0       # capital per contract (used for position sizing)

# ── Indicator periods ──────────────────────────────────────────────────────────
ADX_PERIOD        = 14              # lookback for ADX / DI± calculation
ROC_PERIOD        = 10              # rate-of-change lookback (price momentum, sign)
VOL_MA_PERIOD     = 20              # moving-average window for the volume ratio
SMOOTH_PERIOD     = 3               # EMA smoothing applied to the raw histogram

# ── Entry filters ──────────────────────────────────────────────────────────────
# Minimum histogram magnitude required for entry. Lower → more trades; higher →
# more selective.
STRENGTH_THRESHOLD = 1.0
# Filters out low-trending environments.
MIN_ADX            = 14.0
# Requires above-average participation on the entry bar (1.2 = 1.2× average vol).
VOL_RATIO_MIN      = 1.2
# Dual SMA trend regime filter. Can be disabled to compare baseline performance.
SMA_FAST           = 30
SMA_SLOW           = 60
USE_SMA_FILTER     = True

# ── Exit controls ──────────────────────────────────────────────────────────────
# Static stop loss as a percentage of the option entry price. Always active and
# overrides both regime exits if hit first.
STOP_LOSS_PCT      = 1.0
# How much the histogram must fall from its in-trade peak to trigger the
# peak-drop exit (active only after the SMA filter reverses against the trade).
PEAK_DROP_PCT      = 25.0
# How far the histogram must travel into the opposite zone to trigger the flip
# exit (active while the SMA filter still confirms the trade).
FLIP_THRESHOLD     = 0.8
# Minimum bars held before peak/flip exits can fire (protects against same-bar
# noise immediately after entry). The static stop loss is exempt.
MIN_BARS_HOLD      = 3

# Always fetch raw 1-second bars from Breeze; resampling is done locally.
INTERVAL          = "1second"

# Candle size (in seconds) used for the strategy. Set to 1 for raw 1-second bars.
# The strategy is designed around 5-minute bars; 300 reproduces that timeframe.
RESAMPLE_SECONDS  = 300

# Print the resampled spot signal DataFrame alongside the trades.
PRINT_RESAMPLED   = False

# When True, candle data is served ONLY from the local cache — no Breeze API
# calls are made. Expiries (or days, with per-day ATM) whose option data is not
# cached are skipped.
CACHE_ONLY        = True

# When True, a fresh ATM strike is chosen for EACH trading day from that day's
# Nifty open. When False, a single ATM strike anchored to the week's Monday open
# is traded across the whole expiry window.
PER_DAY_ATM       = False

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

    strategy = RealStrengthOptionSecondsStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        adx_period=ADX_PERIOD,
        roc_period=ROC_PERIOD,
        vol_ma_period=VOL_MA_PERIOD,
        smooth_period=SMOOTH_PERIOD,
        strength_threshold=STRENGTH_THRESHOLD,
        min_adx=MIN_ADX,
        vol_ratio_min=VOL_RATIO_MIN,
        sma_fast=SMA_FAST,
        sma_slow=SMA_SLOW,
        use_sma_filter=USE_SMA_FILTER,
        stop_loss_pct=STOP_LOSS_PCT,
        peak_drop_pct=PEAK_DROP_PCT,
        flip_threshold=FLIP_THRESHOLD,
        min_bars_hold=MIN_BARS_HOLD,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        resample_seconds=RESAMPLE_SECONDS,
        print_resampled=PRINT_RESAMPLED,
        cache_only=CACHE_ONLY,
        market_holidays=MARKET_HOLIDAYS,
        per_day_atm=PER_DAY_ATM,
    )

    expiry_results = strategy.run_weekly_backtest()
    RealStrengthOptionSecondsStrategy.print_report(expiry_results)
