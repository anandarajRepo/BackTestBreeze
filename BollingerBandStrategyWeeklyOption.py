"""
Nifty 50 Weekly Options — Bollinger Band Mean-Reversion Backtest
================================================================

Entry:
  CE — buy when price crosses above the lower Bollinger Band (oversold bounce)
  PE — buy when price crosses below the upper Bollinger Band (overbought reversal)

Exit:
  - Price reaches the middle band (mean reversion target)
  - Square-off at 15:20 IST
  - No new entries before 9:30 or after 14:45
  - Max 5 trades per day per symbol

Note: no stop-loss in this strategy.

Usage:
  Set START_DATE / END_DATE to the desired backtest window (YYYY-MM-DD).
  Each Tuesday in that range is treated as a weekly expiry.
  The ATM strike is computed from Monday's Nifty opening price.
  The trade window for each expiry is Wednesday (prior week) → Tuesday.
"""

import os

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.bollinger_option_strategy import BollingerOptionStrategy

load_dotenv()

# ── Session ───────────────────────────────────────────────────────────────────

breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN"),
)
print("Session Generated Successfully\n")

# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE  = "01-Apr-2026"   # format: DD-Mon-YYYY
END_DATE    = "28-Apr-2026"   # format: DD-Mon-YYYY

CAPITAL     = 100000.0        # capital per contract (used for position sizing)
BB_PERIOD   = 20              # lookback period for Bollinger Bands
BB_STD_DEV  = 2.0             # number of standard deviations for the bands
INTERVAL    = "1minute"       # candle interval: "1second", "1minute", "5minute", "30minute", or "1day"

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nifty_service = NiftyOptionService(breeze)

    strategy = BollingerOptionStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        bb_period=BB_PERIOD,
        bb_std_dev=BB_STD_DEV,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
    )

    expiry_results = strategy.run_weekly_backtest()
    BollingerOptionStrategy.print_report(expiry_results)
