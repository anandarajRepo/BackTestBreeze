"""
Nifty 50 Weekly Options — Opening Range Breakout Vertical Spread Backtest (1-second data)
=========================================================================================

The strategy defines the opening range from a user-defined initial time window
after market open (built from the Nifty spot index). It then trades a directional,
defined-risk vertical debit spread on the breakout:

Entry trigger (one position per day, no re-entry within the session):
  • Spot crosses ABOVE the opening-range high → Bull Call Spread
        Buy 1 ATM Call + Sell 1 higher-strike Call (SPREAD_DISTANCE away).
  • Spot crosses BELOW the opening-range low  → Bear Put Spread
        Buy 1 ATM Put  + Sell 1 lower-strike  Put (SPREAD_DISTANCE away).

Exit trigger (combined two-leg P&L monitored continuously):
  • Combined P&L ≥ +PROFIT_TARGET (default ₹3,000)  → PROFIT_TARGET
  • Combined P&L ≤ −STOP_LOSS     (default ₹3,000)  → STOP_LOSS
  • Time = 15:25 IST (square off all open positions) → SQUARE_OFF

Re-entry logic:
  No re-entry after exit within the same session. Strategy resets next trading day.

Usage:
  1. Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  2. Set RESAMPLE_SECONDS to the desired candle size for the strategy:
       1  → raw 1-second bars (no resampling)
       5  → 5-second bars
       … any positive integer is accepted.
  Data is always fetched as 1-second bars from Breeze; resampling is done
  locally before the strategy logic runs.
"""

import os
from datetime import date

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.orb_spread_option_seconds_strategy import ORBSpreadOptionSecondsStrategy

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
END_DATE          = "16-Jun-2026"   # format: DD-Mon-YYYY

# Length of the opening range, in minutes from 9:15 (the user-defined window).
OR_MINUTES        = 15

# Distance (in index points) between the long (ATM) strike and the short strike
# of the spread. 100 = two 50-point strikes away (e.g. buy 25000, sell 25100).
SPREAD_DISTANCE   = 100

# Combined-position profit target / stop-loss in rupees.
PROFIT_TARGET     = 3000.0
STOP_LOSS         = 3000.0

# Nifty lot size and number of lots traded per leg.
LOT_SIZE          = 75
LOTS              = 1

# Always fetch raw 1-second bars from Breeze; resampling is done locally.
INTERVAL          = "1second"

# Candle size (in seconds) used for the strategy.
# Set to 1 to use raw 1-second bars without any resampling.
RESAMPLE_SECONDS  = 5

# Print the resampled DataFrame alongside trades (verbose).
PRINT_RESAMPLED   = False

# When True, candle data is served ONLY from the local cache — no Breeze API
# calls are made. Any day whose data is not already cached is skipped.
CACHE_ONLY        = True

# NSE market holidays for 2026. When a Tuesday weekly expiry falls on one of
# these dates, the expiry is rolled back to the previous trading day.
MARKET_HOLIDAYS = {
    date(2026, 1, 26),   # Republic Day            (Monday)
    date(2026, 3, 3),    # Holi                    (Tuesday)
    date(2026, 3, 26),   # Shri Ram Navami         (Thursday)
    date(2026, 3, 31),   # Shri Mahavir Jayanti    (Tuesday)
    date(2026, 4, 3),    # Good Friday             (Friday)
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti    (Tuesday)
    date(2026, 5, 1),    # Maharashtra Day         (Friday)
    date(2026, 5, 28),   # Bakri Id (Eid-al-Adha)  (Thursday)
    date(2026, 6, 26),   # Moharram                (Friday)
    date(2026, 8, 15),   # Independence Day        (Saturday)
    date(2026, 9, 14),   # Ganesh Chaturthi        (Monday)
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti  (Friday)
    date(2026, 11, 8),   # Diwali Laxmi Pujan      (Sunday)
    date(2026, 11, 10),  # Diwali Balipratipada    (Tuesday)
    date(2026, 11, 24),  # Guru Nanak Jayanti      (Tuesday)
    date(2026, 12, 25),  # Christmas               (Friday)
}

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nifty_service = NiftyOptionService(breeze)

    strategy = ORBSpreadOptionSecondsStrategy(
        nifty_service=nifty_service,
        or_minutes=OR_MINUTES,
        spread_distance=SPREAD_DISTANCE,
        profit_target=PROFIT_TARGET,
        stop_loss=STOP_LOSS,
        lot_size=LOT_SIZE,
        lots=LOTS,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        resample_seconds=RESAMPLE_SECONDS,
        print_resampled=PRINT_RESAMPLED,
        cache_only=CACHE_ONLY,
        market_holidays=MARKET_HOLIDAYS,
    )

    day_results = strategy.run_backtest()
    ORBSpreadOptionSecondsStrategy.print_report(day_results)
