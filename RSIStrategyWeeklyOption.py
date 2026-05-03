"""
Nifty 50 Weekly Options — RSI Convergence & Divergence Backtest
===============================================================

Signals
-------
  Bullish divergence  → buy CE
    Price makes a lower low while RSI makes a higher low (hidden strength).

  Bearish divergence  → buy PE
    Price makes a higher high while RSI makes a lower high (hidden weakness).

  Bullish convergence → buy CE
    RSI crosses back above the oversold level (momentum confirming upswing).

  Bearish convergence → buy PE
    RSI crosses back below the overbought level (momentum confirming downswing).

Exit
----
  - RSI crosses the 50 neutral level
  - Square-off at 15:20 IST
  - No new entries before 9:30 or after 14:45
  - Max 5 trades per day per symbol

Usage
-----
  Set START_DATE / END_DATE to the desired backtest window (YYYY-MM-DD).
  Each Tuesday in that range is treated as a weekly expiry.
  The ATM strike is computed from Monday's Nifty opening price.
"""

import os

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.rsi_option_strategy import RSIOptionStrategy

load_dotenv()

# ── Session ───────────────────────────────────────────────────────────────────

breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN"),
)
print("Session Generated Successfully\n")

# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE           = "01-Apr-2026"   # format: DD-Mon-YYYY
END_DATE             = "28-Apr-2026"   # format: DD-Mon-YYYY

CAPITAL              = 100000.0        # capital per contract (used for position sizing)
RSI_PERIOD           = 14              # lookback period for RSI calculation
OVERSOLD             = 30.0            # RSI level considered oversold
OVERBOUGHT           = 70.0            # RSI level considered overbought
DIVERGENCE_LOOKBACK  = 5               # bars to scan for price/RSI divergence
INTERVAL             = "1minute"       # candle interval

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nifty_service = NiftyOptionService(breeze)

    strategy = RSIOptionStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        rsi_period=RSI_PERIOD,
        oversold=OVERSOLD,
        overbought=OVERBOUGHT,
        divergence_lookback=DIVERGENCE_LOOKBACK,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
    )

    expiry_results = strategy.run_weekly_backtest()
    RSIOptionStrategy.print_report(expiry_results)
