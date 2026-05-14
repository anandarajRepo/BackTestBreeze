"""
Nifty 50 Weekly Options — RSI + Bollinger Bands Strategy Backtest
=================================================================

Entry (Long / CE):
  Both indicators must be oversold simultaneously:
    • Price is below the lower Bollinger Band
    • RSI is below RSI_OVERSOLD threshold

Entry (Short / PE):
  Both indicators must be overbought simultaneously:
    • Price is above the upper Bollinger Band
    • RSI is above RSI_OVERBOUGHT threshold

Exit:
  Long  (CE) — price crosses above the upper Bollinger Band
  Short (PE) — price crosses below the lower Bollinger Band
  Square-off at 15:20 IST
  No new entries before 9:30 or after 14:45
  Max 5 trades per day per symbol

Mode:
  LONG_ONLY  = True  (default) — only CE legs are traded
  SHORT_ONLY = True             — only PE legs are traded
  Both False                    — CE and PE legs are both traded

Note: LONG_ONLY and SHORT_ONLY cannot both be True.

MINIMUM_ENTRY_PRICE: set to a positive value (e.g. 30) to skip entries where
  the option price is at or below that threshold. Default 0 disables the filter.

Usage:
  Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  Each Wednesday in that range is treated as a weekly expiry.
  The ATM strike is computed from Monday's Nifty opening price.
"""

import os

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.rsi_bb_option_strategy import RSIBBOptionStrategy

load_dotenv()

# ── Session ───────────────────────────────────────────────────────────────────

breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN"),
)
print("Session Generated Successfully\n")

# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE     = "01-Jan-2026"   # format: DD-Mon-YYYY
END_DATE       = "05-May-2026"   # format: DD-Mon-YYYY

CAPITAL        = 100_000.0       # capital per contract (used for position sizing)
RSI_PERIOD     = 14              # lookback period for RSI
RSI_OVERSOLD   = 30.0            # RSI threshold for oversold condition
RSI_OVERBOUGHT = 70.0            # RSI threshold for overbought condition
BB_PERIOD      = 20              # lookback period for Bollinger Bands
BB_STD_DEV     = 2.0             # standard deviation multiplier for Bollinger Bands
INTERVAL       = "1minute"       # candle interval: "1minute", "5minute", "30minute", "1day"

# Mode toggle — exactly one should be True, or both False for dual-leg trading
LONG_ONLY      = True            # default: only trade CE (long side)
SHORT_ONLY     = False           # set True to trade only PE (short side)

MINIMUM_ENTRY_PRICE = 0.0        # minimum option price to allow entry; 0 means no filter

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nifty_service = NiftyOptionService(breeze)

    strategy = RSIBBOptionStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        rsi_period=RSI_PERIOD,
        rsi_oversold=RSI_OVERSOLD,
        rsi_overbought=RSI_OVERBOUGHT,
        bb_period=BB_PERIOD,
        bb_std_dev=BB_STD_DEV,
        long_only=LONG_ONLY,
        short_only=SHORT_ONLY,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        minimum_entry_price=MINIMUM_ENTRY_PRICE,
    )

    expiry_results = strategy.run_weekly_backtest()
    RSIBBOptionStrategy.print_report(expiry_results)
