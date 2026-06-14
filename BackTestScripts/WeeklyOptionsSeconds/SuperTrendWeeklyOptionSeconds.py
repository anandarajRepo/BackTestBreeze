"""
Nifty 50 Weekly Options — Supertrend Backtest  (1-second data)
==============================================================

Entry:
  CE — buy when the CE contract's Supertrend flips from bearish to bullish
  PE — buy when the PE contract's Supertrend flips from bearish to bullish
       (each leg is traded long on its OWN price's Supertrend signal)

Position management / Exit:
  - Scaled take-profit against a percentage TARGET (TARGET_PCT):
        • sell 25% of the position once price reaches 25% of the target
        • sell a further 25% once price reaches 50% of the target
        • sell ALL remaining shares once the full target is hit
  - BREAKEVEN_TRIGGER — once price gains BREAKEVEN_TRIGGER_PCT, the stop is
    raised to the entry price (trade can no longer become a loss).
  - TRAILING_STOP — exit remaining shares if price falls TRAILING_STOP_PCT
    below the highest price reached since entry.
  - Supertrend flip back to bearish closes the remaining position.
  - Square-off at 15:20 IST.
  - No new entries before 9:30 or after 14:45.
  - Max 5 trades per day per symbol.

Usage:
  1. Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  2. Set RESAMPLE_SECONDS to the desired candle size (1, 5, 10, 15, 30, 45, 60…).
  Data is always fetched as 1-second bars from Breeze; resampling is done
  locally before the Supertrend indicator and strategy logic run.
"""

import os
from datetime import date

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.supertrend_option_strategy import SuperTrendOptionStrategy

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
END_DATE          = "09-Jun-2026"   # format: DD-Mon-YYYY

CAPITAL           = 100_000.0       # capital per contract (used for position sizing)

# Supertrend parameters.
ST_PERIOD         = 10              # ATR lookback period for the Supertrend
ST_MULTIPLIER     = 3.0             # ATR multiplier for the Supertrend bands

# Full take-profit target, expressed as a percentage of the entry price.
# The scaled exits fire at 25%, 50% and 100% of this target move:
#   • 25% of position sold at 0.25 × TARGET_PCT gain
#   • 25% of position sold at 0.50 × TARGET_PCT gain
#   • remaining 50% sold at the full TARGET_PCT gain
TARGET_PCT        = 15.0

# Trailing stop-loss. When enabled, the remaining position is closed if the
# option price falls TRAILING_STOP_PCT percent below the highest price reached
# since entry (the stop ratchets up with the peak, never down).
TRAILING_STOP_ENABLED = True
TRAILING_STOP_PCT     = 3.0

# Breakeven trigger. Once the option price gains BREAKEVEN_TRIGGER_PCT percent,
# the stop is raised to the entry price so the trade can no longer turn into a
# loss. Set BREAKEVEN_TRIGGER_ENABLED to False to disable.
BREAKEVEN_TRIGGER_ENABLED = True
BREAKEVEN_TRIGGER_PCT     = 3.0

# Always fetch raw 1-second bars from Breeze; resampling is done locally.
INTERVAL          = "1second"

# Candle size (in seconds) used for the strategy. Set to 1 to use raw 1-second
# bars without any resampling.
RESAMPLE_SECONDS  = 60

# Print the final resampled DataFrame (with Supertrend) alongside the trades.
PRINT_RESAMPLED   = False

# When True, candle data is served ONLY from the local cache — no Breeze API
# calls are made. Expiries whose data is not already cached are skipped.
CACHE_ONLY        = True

# When True, a fresh ATM strike is chosen for EACH trading day from that day's
# Nifty 9:15 open. When False, a single ATM strike is anchored to the week's
# Monday open and traded across the whole expiry window.
PER_DAY_ATM       = True

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

    strategy = SuperTrendOptionStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        st_period=ST_PERIOD,
        st_multiplier=ST_MULTIPLIER,
        target_pct=TARGET_PCT,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        resample_seconds=RESAMPLE_SECONDS,
        print_resampled=PRINT_RESAMPLED,
        cache_only=CACHE_ONLY,
        market_holidays=MARKET_HOLIDAYS,
        per_day_atm=PER_DAY_ATM,
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_pct=TRAILING_STOP_PCT,
        breakeven_trigger_enabled=BREAKEVEN_TRIGGER_ENABLED,
        breakeven_trigger_pct=BREAKEVEN_TRIGGER_PCT,
    )

    expiry_results = strategy.run_weekly_backtest()
    SuperTrendOptionStrategy.print_report(expiry_results)
