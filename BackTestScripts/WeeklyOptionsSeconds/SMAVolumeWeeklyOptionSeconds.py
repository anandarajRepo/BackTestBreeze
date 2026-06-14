"""
Nifty 50 Weekly Options — SMA Fast/Slow Crossover + Volume Backtest  (1-second data)
====================================================================================

Entry:
  CE — buy when the fast SMA crosses above the slow SMA and volume is good
  PE — buy when the fast SMA crosses above the slow SMA and volume is good
       ("good volume" = entry bar volume >= VOLUME_FACTOR × rolling-avg volume)

Exit:
  - SMA crossover reversal (fast SMA crosses back below the slow SMA)
  - Trailing stop-loss (optional, percentage-based; see TRAILING_STOP_* config)
  - Break-even stop: once price moves BREAKEVEN_TRIGGER_PCT percent above entry,
    the stop-loss is moved up to the entry price (see BREAKEVEN_* config)
  - Square-off at 15:20 IST
  - No new entries before 9:30 or after 14:45
  - Max 5 trades per day per symbol

Usage:
  1. Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  2. Set RESAMPLE_SECONDS to the desired candle size for the strategy:
       1  → raw 1-second bars (no resampling)
       5  → 5-second bars
       10 → 10-second bars
       15 → 15-second bars
       30 → 30-second bars
       45 → 45-second bars
       60 → 60-second (1-minute) bars
     Any positive integer is accepted.
  Data is always fetched as 1-second bars from Breeze; resampling is done
  locally before the SMA indicators and strategy logic run.
"""

import os
from datetime import date

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.sma_option_strategy import SMAOptionStrategy

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
END_DATE          = "02-Jun-2026"   # format: DD-Mon-YYYY

CAPITAL           = 100_000.0       # capital per contract (used for position sizing)

# SMA crossover lookback periods (in resampled bars). The fast SMA crossing
# above the slow SMA is the entry trigger; crossing back below is the exit.
SMA_FAST          = 9               # fast simple moving average period
SMA_SLOW          = 21              # slow simple moving average period

# Volume confirmation for entries. The entry bar's volume must be at least
# VOLUME_FACTOR times the rolling-average volume (over SMA_SLOW bars) for the
# SMA crossover signal to be taken. 1.0 = at least average volume; >1.0 demands
# an above-average ("good") volume surge. Set to 0 to disable the volume filter.
VOLUME_FACTOR     = 1.0

# Trailing stop-loss. When TRAILING_STOP_ENABLED is True, the position is
# closed if the option price falls TRAILING_STOP_PCT percent below the highest
# price reached since entry (the stop ratchets up with the peak, never down).
TRAILING_STOP_ENABLED = True
TRAILING_STOP_PCT     = 20.0

# Break-even stop. When enabled, once the option price moves
# BREAKEVEN_TRIGGER_PCT percent above the entry price, the stop-loss is moved up
# to the entry price so the trade can no longer turn into a loss.
BREAKEVEN_ENABLED     = True
BREAKEVEN_TRIGGER_PCT = 5.0

# Always fetch raw 1-second bars from Breeze; resampling is done locally.
INTERVAL          = "1second"

# Candle size (in seconds) used for the strategy.
# Supported examples: 1, 5, 10, 15, 30, 45, 60, 120, 300, …
# Set to 1 to use raw 1-second bars without any resampling.
RESAMPLE_SECONDS  = 5

# Print the final resampled DataFrame (with SMA indicators) alongside the
# trades for each option contract before the summary report.
PRINT_RESAMPLED   = False

# When True, candle data is served ONLY from the local cache — no Breeze API
# calls are made. Any expiry whose data is not already cached is skipped.
CACHE_ONLY        = True

# When True, a fresh ATM strike is chosen for EACH trading day from that day's
# Nifty 9:15 open. When False (default), a single ATM strike is anchored to the
# week's Monday open and traded across the whole expiry window.
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

    strategy = SMAOptionStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        sma_fast=SMA_FAST,
        sma_slow=SMA_SLOW,
        volume_factor=VOLUME_FACTOR,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        resample_seconds=RESAMPLE_SECONDS,
        print_resampled=PRINT_RESAMPLED,
        cache_only=CACHE_ONLY,
        market_holidays=MARKET_HOLIDAYS,
        per_day_atm=PER_DAY_ATM,
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_pct=TRAILING_STOP_PCT,
        breakeven_enabled=BREAKEVEN_ENABLED,
        breakeven_trigger_pct=BREAKEVEN_TRIGGER_PCT,
    )

    expiry_results = strategy.run_weekly_backtest()
    SMAOptionStrategy.print_report(expiry_results)
