"""
Nifty 50 Weekly Options — Lucky13EMA 1-Min Momentum Scalper Backtest
=====================================================================

Entry (signal from Nifty spot 1-min candles):
  CE — first green candle closing above 13 EMA after being below
  PE — first red  candle closing below 13 EMA after being above

Triple Filter (USE_FILTERS toggle):
  1. Volume > 20-period SMA × VOL_MULTIPLIER
  2. VWAP alignment  (CE: close > VWAP, PE: close < VWAP)
       Set USE_VWAP_CROSS=True for stricter "actual VWAP cross" condition
  3. 5-min EMA bias  (CE: spot above 5-min EMA, PE: spot below 5-min EMA)

Exit (on option price):
  - Profit target : PROFIT_PCT %  above option entry price
  - Stop loss     : STOP_PCT   %  below option entry price
  - Trailing stop : TRAILING_STOP_PTS points below option peak (0 = disabled)
  - EMA reversal  : spot crosses back through 13 EMA against the trade
  - Square-off    : 15:20 IST
  - No new entries : before 9:30 or after 14:45 | max 5 trades/day/symbol

Note: ATM strike is computed from Monday's Nifty opening price.
      Trade window per expiry: prior Wednesday → Tuesday expiry.

Usage:
  Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  Tune PROFIT_PCT, STOP_PCT, VOL_MULTIPLIER, USE_FILTERS as needed.
"""

import os

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.lucky13_ema_strategy import Lucky13EmaStrategy

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
END_DATE            = "19-May-2026"   # format: DD-Mon-YYYY

CAPITAL             = 100_000.0       # capital per contract (for position sizing)

# Indicator parameters
EMA_PERIOD          = 13              # 13 EMA on 1-min spot candles
VOL_PERIOD          = 20              # volume SMA lookback

# Filter parameters
USE_FILTERS         = True            # master toggle — disable to see raw EMA signals
VOL_MULTIPLIER      = 1.2             # volume must be > SMA × this value
USE_VWAP_CROSS      = False           # False: close vs VWAP  |  True: require VWAP cross

# Exit parameters
PROFIT_PCT          = 1.5             # take profit at +1.5% on option price
STOP_PCT            = 0.75            # stop loss  at −0.75% on option price
TRAILING_STOP_PTS   = 0.0             # trailing stop in option price points (0 = off)

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nifty_service = NiftyOptionService(breeze)

    strategy = Lucky13EmaStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        ema_period=EMA_PERIOD,
        vol_period=VOL_PERIOD,
        vol_multiplier=VOL_MULTIPLIER,
        profit_pct=PROFIT_PCT,
        stop_pct=STOP_PCT,
        trailing_stop_pts=TRAILING_STOP_PTS,
        use_filters=USE_FILTERS,
        use_vwap_cross=USE_VWAP_CROSS,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    expiry_results = strategy.run_weekly_backtest()
    Lucky13EmaStrategy.print_report(expiry_results)
