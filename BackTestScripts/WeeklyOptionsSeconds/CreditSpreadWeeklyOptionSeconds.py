"""
Nifty 50 Weekly Options — Credit Spread Backtest (1-second data)
================================================================

A credit spread simultaneously SELLS a higher-premium option and BUYS a
lower-premium option of the same type and expiry, collecting a net credit. It is
a defined-risk, theta-positive (time-decay) play:

  • Bull Put Spread  — sell an OTM Put + buy a further-OTM Put (below it).
        Deployed on a bullish / sideways bias: profits while the Nifty stays
        above the short put strike as both puts decay.
  • Bear Call Spread — sell an OTM Call + buy a further-OTM Call (above it).
        Deployed on a bearish / sideways bias: profits while the Nifty stays
        below the short call strike as both calls decay.

Key mechanics:
  • Max profit = net credit collected (both legs expire worthless).
  • Max risk   = spread width − net credit (spot beyond the long strike).
  • Breakeven  = short strike − net credit (bull put)
               = short strike + net credit (bear call).

Direction selection (DIRECTION_MODE):
  • "TREND"     — build an opening range from the first OR_MINUTES of Nifty spot;
                  a break ABOVE the range → Bull Put Spread, a break BELOW →
                  Bear Call Spread. One position per day on the first breakout.
  • "BULL_PUT"  — always deploy a Bull Put Spread at the opening-range end.
  • "BEAR_CALL" — always deploy a Bear Call Spread at the opening-range end.

Exit (combined two-leg P&L monitored continuously):
  • Combined P&L ≥ PROFIT_TARGET_PCT % of the collected credit → PROFIT_TARGET
  • Combined P&L ≤ −STOP_LOSS_MULT × the collected credit      → STOP_LOSS
  • Spot breaches the breakeven (short strike ∓ net credit)    → BREAKEVEN_BREACH
  • Time = 15:25 IST (square off whatever is still open)       → SQUARE_OFF

Usage:
  1. Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  2. Set SHORT_OTM_DISTANCE (300-500 points is typical for a high-probability
     OTM short strike) and SPREAD_WIDTH (the defined-risk width).
  3. Set RESAMPLE_SECONDS to the desired candle size for the strategy:
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
from strategy.credit_spread_option_seconds_strategy import (
    CreditSpreadOptionSecondsStrategy,
)

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

# Length of the opening range, in minutes from 9:15. In "TREND" mode this window
# selects the spread side (breakout above → bull put, breakdown below → bear
# call); in the forced-side modes it is simply the entry timestamp.
OR_MINUTES        = 15

# How the spread side is chosen each day:
#   "TREND"     → opening-range breakout picks the side (default)
#   "BULL_PUT"  → always a Bull Put Spread
#   "BEAR_CALL" → always a Bear Call Spread
DIRECTION_MODE    = "TREND"

# Distance (index points) from the ATM to the SHORT strike. The short leg is
# placed this far OTM (puts below spot / calls above spot). The execution
# guidance targets 300-500 points for a high-probability OTM short strike.
SHORT_OTM_DISTANCE = 400

# Distance (index points) between the short strike and the further-OTM long
# (protective) strike — the spread width and the max-risk reference.
SPREAD_WIDTH       = 100

# Profit target as a percentage of the collected net credit. Booking ~50% of the
# credit is a common theta-play target.
PROFIT_TARGET_PCT  = 50.0

# Stop-loss as a multiple of the collected net credit. e.g. 2.0 exits once the
# combined loss reaches twice the credit received.
STOP_LOSS_MULT     = 2.0

# When True, also exit if the Nifty spot breaches the breakeven
# (short strike ∓ net credit), per the risk-management execution tip.
BREAKEVEN_EXIT_ENABLED = True

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

    strategy = CreditSpreadOptionSecondsStrategy(
        nifty_service=nifty_service,
        or_minutes=OR_MINUTES,
        direction_mode=DIRECTION_MODE,
        short_otm_distance=SHORT_OTM_DISTANCE,
        spread_width=SPREAD_WIDTH,
        profit_target_pct=PROFIT_TARGET_PCT,
        stop_loss_mult=STOP_LOSS_MULT,
        breakeven_exit_enabled=BREAKEVEN_EXIT_ENABLED,
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
    CreditSpreadOptionSecondsStrategy.print_report(day_results)
