"""
Nifty 50 Weekly Options — 3-EMA + ATR Take Profit / Stop Loss Backtest
=======================================================================

Entry:
  CE — buy when Middle EMA crosses above Slow EMA
  PE — buy when Middle EMA crosses below Slow EMA

Exit:
  - Take Profit : entry ± (ATR_MULTIPLIER_TP × ATR)
  - Stop Loss   : entry ∓ (ATR_MULTIPLIER_SL × ATR)
  - Fast EMA crosses below Middle EMA (CE) / above Middle EMA (PE)
  - Square-off at 15:20 IST
  - No new entries before 9:30 or after 14:45
  - Max 5 trades per day per symbol

Usage:
  Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  Each Wednesday in that range is treated as a weekly expiry.
  The ATM strike is computed from Monday's Nifty opening price.
  The trade window for each expiry is Wednesday (prior week) → Wednesday.
"""

import os

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.ema_atr_option_strategy import EMAATROptionStrategy

load_dotenv()

# ── Session ───────────────────────────────────────────────────────────────────

breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN"),
)
print("Session Generated Successfully\n")

# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE          = "01-Jan-2026"   # format: DD-Mon-YYYY
END_DATE            = "05-May-2026"   # format: DD-Mon-YYYY

CAPITAL             = 100000.0        # capital per contract (used for position sizing)

# EMA periods
FAST_EMA_PERIOD     = 9               # fast EMA  — used for exit signal
MID_EMA_PERIOD      = 21              # middle EMA — used for entry signal
SLOW_EMA_PERIOD     = 50              # slow EMA  — used for entry signal

# ATR settings
ATR_PERIOD          = 14              # lookback period for ATR
ATR_MULTIPLIER_TP   = 2.0             # take-profit = entry ± (ATR_MULTIPLIER_TP × ATR)
ATR_MULTIPLIER_SL   = 1.0             # stop-loss   = entry ∓ (ATR_MULTIPLIER_SL × ATR)

INTERVAL            = "1minute"       # candle interval: "1minute", "5minute", "30minute"

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nifty_service = NiftyOptionService(breeze)

    strategy = EMAATROptionStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        fast_period=FAST_EMA_PERIOD,
        mid_period=MID_EMA_PERIOD,
        slow_period=SLOW_EMA_PERIOD,
        atr_period=ATR_PERIOD,
        atr_multiplier_tp=ATR_MULTIPLIER_TP,
        atr_multiplier_sl=ATR_MULTIPLIER_SL,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
    )

    expiry_results = strategy.run_weekly_backtest()
    EMAATROptionStrategy.print_report(expiry_results)
