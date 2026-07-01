"""
Nifty 50 Weekly Options — Heikin Ashi Exhaustion + Supertrend Backtest  (1-second data)
========================================================================================

Ported from a "Shinobi HA" style Pine Script strategy: it looks for Heikin Ashi
exhaustion/pullback patterns (two clean same-direction HA candles followed by a
doji-like indecision candle with rising range) as reversal entries within a
Supertrend trend, filtered by ADX trend strength, rising swing lows (structure)
and an optional volume surge.

Entry (long the option premium, CE & PE legs — each on its OWN price series):
  A bullish HA exhaustion pattern completes, the contract's own Supertrend is
  bullish, ADX >= adx_min, the swing-low structure is rising, and (optionally)
  the entry bar has a volume surge.

Exit:
  - Stop-loss at the Heikin Ashi low of the entry (signal) bar.
  - Target at entry + (entry - stop) * RR_RATIO  (1:1 by default).
  - Square-off at 15:20 IST.
  - No new entries before 9:30 or after 14:45.
  - Max 5 trades per day per symbol.

Usage:
  1. Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  2. Set RESAMPLE_SECONDS to the desired candle size for the strategy:
       1  → raw 1-second bars (no resampling)
       5  → 5-second bars
       … any positive integer is accepted.
  Data is always fetched as 1-second bars from Breeze; resampling is done
  locally before the indicators and strategy logic run.
"""

import os
from datetime import date

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.heikin_ashi_supertrend_option_strategy import HeikinAshiSupertrendOptionStrategy

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

CAPITAL           = 100_000.0       # capital per contract (used for position sizing)

# Supertrend trend filter, computed on the option's own OHLC.
ST_PERIOD         = 15
ST_MULTIPLIER     = 3.44

# ADX trend-strength filter.
ADX_PERIOD        = 14
ADX_MIN           = 22.0

# Heikin Ashi doji / exhaustion candle detection.
DOJI_MAX_PCT        = 20.0   # doji body must be <= this % of its total range
WICK_TOLERANCE_PCT  = 3.0    # "clean" wick tolerance % for the two pullback candles
DOJI_WICK_MULT      = 0.3    # both doji wicks must be >= body * this multiple

# Risk:reward ratio for the fixed target. 1.0 = 1:1, 1.5 = 1:1.5, 2.0 = 1:2, etc.
RR_RATIO          = 1.0

# Volume confirmation for entries. The entry bar's volume must be at least
# VOLUME_MULT times its EMA(VOLUME_MA_LEN). Set USE_VOLUME_FILTER to False to
# disable the volume filter entirely.
USE_VOLUME_FILTER = True
VOLUME_MA_LEN     = 20
VOLUME_MULT       = 1.0

# Swing high/low structure filter sensitivity (bars on each side of a pivot).
SWING_LEN         = 5

# Always fetch raw 1-second bars from Breeze; resampling is done locally.
INTERVAL          = "1second"

# Candle size (in seconds) used for the strategy.
# Set to 1 to use raw 1-second bars without any resampling.
RESAMPLE_SECONDS  = 120

# Max entries per day per contract.
MAX_TRADES_PER_DAY = 5

# Print the final resampled DataFrame (with indicators) alongside the trades for
# each option contract before the summary report.
PRINT_RESAMPLED   = False

# When True, candle data is served ONLY from the local cache — no Breeze API
# calls are made. Any expiry whose data is not already cached is skipped.
CACHE_ONLY        = True

# When True, a fresh ATM strike is chosen for EACH trading day from that day's
# Nifty 9:15 open. When False (default), a single ATM strike is anchored to the
# week's Monday open and traded across the whole expiry window.
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

    strategy = HeikinAshiSupertrendOptionStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        st_period=ST_PERIOD,
        st_multiplier=ST_MULTIPLIER,
        adx_period=ADX_PERIOD,
        adx_min=ADX_MIN,
        doji_max_pct=DOJI_MAX_PCT,
        wick_tolerance_pct=WICK_TOLERANCE_PCT,
        doji_wick_mult=DOJI_WICK_MULT,
        rr_ratio=RR_RATIO,
        use_volume_filter=USE_VOLUME_FILTER,
        volume_ma_len=VOLUME_MA_LEN,
        volume_mult=VOLUME_MULT,
        swing_len=SWING_LEN,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        resample_seconds=RESAMPLE_SECONDS,
        max_trades_per_day=MAX_TRADES_PER_DAY,
        print_resampled=PRINT_RESAMPLED,
        cache_only=CACHE_ONLY,
        market_holidays=MARKET_HOLIDAYS,
        per_day_atm=PER_DAY_ATM,
    )

    expiry_results = strategy.run_weekly_backtest()
    HeikinAshiSupertrendOptionStrategy.print_report(expiry_results)
