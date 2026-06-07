"""
Nifty 50 Weekly Options — ADX DI+/DI- Crossover Backtest  (1-second data)
==========================================================================

Entry:
  CE — buy when DI+ crosses above DI- and ADX >= ADX_THRESHOLD
  PE — buy when DI- crosses above DI+ and ADX >= ADX_THRESHOLD

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
ADX_PERIOD        = 16              # lookback period for ADX / DI calculation
ADX_THRESHOLD     = 30.0            # minimum ADX value required to enter a trade

# Always fetch raw 1-second bars from Breeze; resampling is done locally.
INTERVAL          = "1second"

# Candle size (in seconds) used for the strategy.
# Supported examples: 1, 5, 10, 15, 30, 45, 60, 120, 300, …
# Set to 1 to use raw 1-second bars without any resampling.
RESAMPLE_SECONDS  = 5

# Print the final resampled DataFrame (with ADX/DI indicators) alongside the
# trades for each option contract before the summary report.
PRINT_RESAMPLED   = True

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nifty_service = NiftyOptionService(breeze)

    strategy = ADXOptionStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        adx_period=ADX_PERIOD,
        adx_threshold=ADX_THRESHOLD,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        resample_seconds=RESAMPLE_SECONDS,
        print_resampled=PRINT_RESAMPLED,
    )

    expiry_results = strategy.run_weekly_backtest()
    ADXOptionStrategy.print_report(expiry_results)
