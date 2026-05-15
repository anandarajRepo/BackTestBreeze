"""
Nifty 50 Weekly Options — RSI Convergence & Divergence Backtest
===============================================================

Signal types:

DIVERGENCE
  CE — bullish divergence: price prints a lower low but RSI prints a higher low
       (classic oversold divergence signalling upward reversal)
  PE — bearish divergence: price prints a higher high but RSI prints a lower high
       (classic overbought divergence signalling downward reversal)

CONVERGENCE  (RSI extreme mean-reversion)
  CE — RSI crosses back above the oversold threshold (was below, now above)
  PE — RSI crosses back below the overbought threshold (was above, now below)

Exit:
  - RSI crosses the neutral exit level (default 50)
  - Square-off at 15:20 IST
  - No new entries before 9:30 or after 14:45
  - Max 5 trades per day per symbol

Usage:
  Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  Each Tuesday in that range is treated as a weekly expiry.
  The ATM strike is computed from Monday's Nifty opening price.
  The trade window for each expiry is Wednesday (prior week) → Tuesday.
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

START_DATE          = "01-Apr-2026"   # format: DD-Mon-YYYY
END_DATE            = "05-May-2026"   # format: DD-Mon-YYYY

CAPITAL             = 100000.0        # capital per contract (used for position sizing)
RSI_PERIOD          = 14              # RSI lookback period
RSI_OVERSOLD        = 30.0            # entry threshold for CE convergence/divergence
RSI_OVERBOUGHT      = 70.0            # entry threshold for PE convergence/divergence
RSI_EXIT_LEVEL      = 50.0            # RSI level that triggers exit (neutral zone)
DIVERGENCE_LOOKBACK = 20              # number of bars to scan for divergence pivots
INTERVAL            = "1minute"       # candle interval

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nifty_service = NiftyOptionService(breeze)

    strategy = RSIOptionStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        rsi_period=RSI_PERIOD,
        rsi_oversold=RSI_OVERSOLD,
        rsi_overbought=RSI_OVERBOUGHT,
        rsi_exit_level=RSI_EXIT_LEVEL,
        divergence_lookback=DIVERGENCE_LOOKBACK,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
    )

    expiry_results = strategy.run_weekly_backtest()
    RSIOptionStrategy.print_report(expiry_results)
