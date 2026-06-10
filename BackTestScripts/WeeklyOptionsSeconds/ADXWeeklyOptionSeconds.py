"""
Nifty 50 Weekly Options — ADX DI+/DI- Crossover Backtest  (1-second data)
==========================================================================

Entry:
  CE — buy when DI+ crosses above DI-, ADX >= ADX_THRESHOLD, and volume is good
  PE — buy when DI- crosses above DI+, ADX >= ADX_THRESHOLD, and volume is good
       ("good volume" = entry bar volume >= VOLUME_FACTOR × rolling-avg volume)

Exit:
  - DI direction reversal (crossover flips)
  - Square-off at 15:20 IST
  - No new entries before 9:30 or after 14:45
  - Max 5 trades per day per symbol

Note: no stop-loss in this strategy.

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
  locally before the ADX indicator and strategy logic run.
"""

import os

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.adx_option_strategy import ADXOptionStrategy

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
END_DATE          = "26-May-2026"   # format: DD-Mon-YYYY

CAPITAL           = 100_000.0       # capital per contract (used for position sizing)
ADX_PERIOD        = 60              # lookback period for ADX / DI calculation
ADX_THRESHOLD     = 24              # minimum ADX value required to enter a trade

# Volume confirmation for entries. The entry bar's volume must be at least
# VOLUME_FACTOR times the rolling-average volume (over ADX_PERIOD bars) for the
# DI crossover signal to be taken. 1.0 = at least average volume; >1.0 demands
# an above-average ("good") volume surge. Set to 0 to disable the volume filter.
VOLUME_FACTOR     = 1.0

# Always fetch raw 1-second bars from Breeze; resampling is done locally.
INTERVAL          = "1second"

# Candle size (in seconds) used for the strategy.
# Supported examples: 1, 5, 10, 15, 30, 45, 60, 120, 300, …
# Set to 1 to use raw 1-second bars without any resampling.
RESAMPLE_SECONDS  = 5

# Print the final resampled DataFrame (with ADX/DI indicators) alongside the
# trades for each option contract before the summary report.
PRINT_RESAMPLED   = False

# When True, candle data is served ONLY from the local cache — no Breeze API
# calls are made. Any expiry whose data is not already cached is skipped and the
# backtest moves on to the next expiry.
# When False, data is read from cache as usual, and any data not present in the
# cache is fetched from the Breeze API via a historical-data request.
CACHE_ONLY        = True

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nifty_service = NiftyOptionService(breeze)

    strategy = ADXOptionStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        adx_period=ADX_PERIOD,
        adx_threshold=ADX_THRESHOLD,
        volume_factor=VOLUME_FACTOR,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        resample_seconds=RESAMPLE_SECONDS,
        print_resampled=PRINT_RESAMPLED,
        cache_only=CACHE_ONLY,
    )

    expiry_results = strategy.run_weekly_backtest()
    ADXOptionStrategy.print_report(expiry_results)
