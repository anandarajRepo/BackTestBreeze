"""
Nifty 50 Weekly Options — ADX DI+/DI- Crossover Backtest
=========================================================

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
  Set START_DATE / END_DATE to the desired backtest window (YYYY-MM-DD).
  Each Thursday in that range is treated as a weekly expiry.
  The ATM strike is computed from Monday's Nifty opening price.
  The trade window for each expiry is Wednesday (prior week) → Thursday.
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

START_DATE     = "01-Jan-2025"   # format: DD-Mon-YYYY
END_DATE       = "31-Mar-2025"   # format: DD-Mon-YYYY

CAPITAL        = 100_000.0       # capital per contract (used for position sizing)
ADX_PERIOD     = 14              # lookback period for ADX / DI calculation
ADX_THRESHOLD  = 20.0            # minimum ADX value required to enter a trade

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
    )

    expiry_results = strategy.run_weekly_backtest()
    ADXOptionStrategy.print_report(expiry_results)
