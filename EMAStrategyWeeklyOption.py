"""
Nifty 50 Weekly Options — 5-Minute EMA Crossover Backtest
==========================================================

Entry:
  CE — buy when the fast EMA crosses above the slow EMA (bullish crossover)
  PE — buy when the fast EMA crosses below the slow EMA (bearish crossover)

Exit:
  - Opposite EMA crossover (trend reversal signal)
  - Square-off at 15:20 IST
  - No new entries before 9:30 or after 14:45
  - Max 5 trades per day per symbol

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
from strategy.ema_crossover_strategy import EMACrossoverStrategy

load_dotenv()

# ── Session ───────────────────────────────────────────────────────────────────

breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN"),
)
print("Session Generated Successfully\n")

# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE   = "01-Jan-2026"   # format: DD-Mon-YYYY
END_DATE     = "05-May-2026"   # format: DD-Mon-YYYY

CAPITAL             = 100000     # capital per contract (used for position sizing)
FAST_PERIOD         = 45        # fast EMA period
SLOW_PERIOD         = 105       # slow EMA period
INTERVAL            = "1minute" # candle interval — keep as 5minute for this strategy
TRAILING_STOP_PCT   = 50.0      # trailing stop loss percentage (e.g. 10.0 = 10%)

# When True: use full CAPITAL if option price >= 100, else use 10% of CAPITAL.
# When False: always use full CAPITAL regardless of price.
PRICE_BASED_CAPITAL = False

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nifty_service = NiftyOptionService(breeze)

    strategy = EMACrossoverStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        fast_period=FAST_PERIOD,
        slow_period=SLOW_PERIOD,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        trailing_stop_pct=TRAILING_STOP_PCT,
        price_based_capital=PRICE_BASED_CAPITAL,
    )

    expiry_results = strategy.run_weekly_backtest()
    EMACrossoverStrategy.print_report(expiry_results)
