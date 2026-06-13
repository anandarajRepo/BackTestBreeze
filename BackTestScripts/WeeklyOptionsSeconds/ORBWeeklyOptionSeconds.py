"""
Nifty 50 Weekly Options — Open Range Breakout Backtest  (1-second data)
========================================================================

Strategy:
  For each trading day, an opening range (ORB) is built from the first
  ORB_MINUTES of trading (from 9:15). A LONG option entry is taken when a candle
  CLOSES above the ORB high *with good volume* — the breakout candle's volume
  must be at least VOLUME_FACTOR times the average opening-range volume. Buying
  an option is a directional long bet, so only upside breakouts of the option's
  own price are traded (a CE breakout = bullish index, a PE breakout = bearish
  index).

Entry:
  CE / PE — buy when an option candle closes above its opening-range high with
            volume >= VOLUME_FACTOR × average ORB volume ("good volume breakout")

Exit:
  - Target  : entry + risk × RISK_REWARD_RATIO  (risk = entry − stop)
  - Stop-loss: STOP_LOSS_PCT percent below entry
  - Trailing stop-loss (optional; see TRAILING_STOP_* config)
  - Square-off at 15:20 IST
  - No new entries before the opening range completes or after 14:45
  - Max MAX_TRADES_PER_DAY trades per day per symbol

Usage:
  1. Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  2. Set RESAMPLE_SECONDS to the desired candle size for the strategy:
       1  → raw 1-second bars (no resampling)
       5  → 5-second bars, 10/15/30/45/60 → larger bars (any positive integer).
  Data is always fetched as 1-second bars from Breeze; resampling is done
  locally before the ORB logic runs.
"""

import os
from datetime import date

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.orb_option_seconds_strategy import ORBOptionSecondsStrategy

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
END_DATE          = "02-Jun-2026"   # format: DD-Mon-YYYY

CAPITAL           = 100_000.0       # capital per contract (used for position sizing)

# Opening range length, in minutes from 9:15. The first ORB_MINUTES of each day
# define the breakout high/low and the average volume benchmark.
ORB_MINUTES       = 15

# Volume confirmation for breakouts. The breakout candle's volume must be at
# least VOLUME_FACTOR times the average opening-range volume for the breakout to
# be taken. 1.0 = at least average volume; >1.0 demands a "good"/surging volume
# breakout. Set to 0 to disable the volume filter.
VOLUME_FACTOR     = 1.5

# Hard stop-loss as a percentage below the entry price, and target as a multiple
# of the stop distance (risk:reward).
STOP_LOSS_PCT     = 25.0
RISK_REWARD_RATIO = 2.0

# Maximum number of breakout entries per day per option contract.
MAX_TRADES_PER_DAY = 5

# Trailing stop-loss. When enabled, the position is closed if the option price
# falls TRAILING_STOP_PCT percent below the highest price reached since entry
# (the stop ratchets up with the peak, never down).
TRAILING_STOP_ENABLED = True
TRAILING_STOP_PCT     = 20.0

# Always fetch raw 1-second bars from Breeze; resampling is done locally.
INTERVAL          = "1second"

# Candle size (in seconds) used for the strategy. Set to 1 for raw 1-second bars.
RESAMPLE_SECONDS  = 5

# Print the final resampled DataFrame alongside the trades for each contract.
PRINT_RESAMPLED   = False

# When True, candle data is served ONLY from the local cache — no Breeze API
# calls are made. Expiries whose data is not cached are skipped.
CACHE_ONLY        = True

# When True, a fresh ATM strike is chosen for EACH trading day from that day's
# Nifty 9:15 open. When False, a single ATM strike anchored to the week's Monday
# open is traded across the whole expiry window.
PER_DAY_ATM       = False

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

    strategy = ORBOptionSecondsStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        orb_minutes=ORB_MINUTES,
        volume_factor=VOLUME_FACTOR,
        stop_loss_pct=STOP_LOSS_PCT,
        risk_reward_ratio=RISK_REWARD_RATIO,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        resample_seconds=RESAMPLE_SECONDS,
        max_trades_per_day=MAX_TRADES_PER_DAY,
        print_resampled=PRINT_RESAMPLED,
        cache_only=CACHE_ONLY,
        market_holidays=MARKET_HOLIDAYS,
        per_day_atm=PER_DAY_ATM,
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_pct=TRAILING_STOP_PCT,
    )

    expiry_results = strategy.run_weekly_backtest()
    ORBOptionSecondsStrategy.print_report(expiry_results)
