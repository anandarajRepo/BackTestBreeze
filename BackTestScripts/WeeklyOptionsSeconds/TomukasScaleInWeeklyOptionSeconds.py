"""
Nifty 50 Weekly Options — Tomukas Scale-In v2 Backtest  (1-second data)
=======================================================================

Port of the TradingView Pine strategy "Tomukas Scale-In V2" to the option
premium (long CE & PE legs, traded independently):

Trend filter:
  EMA100 > EMA200 on the option premium (bullish premium trend). Only the
  bullish side is traded per leg since option positions are long-only; the
  Pine script's short side is expressed by the opposite leg's own premium
  turning bullish.

Entry (liquidity sweep):
  the bar's low sweeps below the lowest low of the previous SWEEP_LOOKBACK
  bars, but the bar closes back above that level and closes green
  (close > open), while the trend filter holds.

Scale-in engine (pyramiding, up to 5 entries):
  the first sweep opens the position; each subsequent sweep while in a
  position adds another leg with quantity weights ENTRY_WEIGHTS
  (default 10/10/20/40/80 — later adds are progressively larger).

Exit (take-profit only, as in the Pine script):
  - Take-profit at (weighted-average entry price + ATR × TP_ATR_MULT),
    recomputed every bar with the current ATR
  - Square-off at 15:20 IST
  - No new positions before 9:30 or after 14:45
  - Max 5 positions per day per symbol

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
from strategy.tomukas_scale_in_option_strategy import TomukasScaleInOptionStrategy

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
END_DATE          = "30-Jun-2026"   # format: DD-Mon-YYYY

CAPITAL           = 100_000.0       # capital per contract (used for position sizing)

# Trend filter EMAs on the option premium (Pine: ema100 / ema200).
EMA_FAST_PERIOD   = 100
EMA_SLOW_PERIOD   = 200

# Liquidity-sweep lookback (Pine: "Sweep Lookback"). The entry bar must sweep
# below the lowest low of the previous SWEEP_LOOKBACK bars and close back
# above it.
SWEEP_LOOKBACK    = 20

# Scale-in quantity weights (Pine: Entry 1 … Entry 5). Leg i is allocated
# CAPITAL × weight / sum(weights); at most len(ENTRY_WEIGHTS) legs are
# pyramided into one position.
ENTRY_WEIGHTS     = (10, 10, 20, 40, 80)

# ATR take-profit (Pine: "ATR Length" / "TP ATR Multiplier"). The whole
# position is closed once price reaches
#   weighted-average entry price + ATR × TP_ATR_MULT.
ATR_PERIOD        = 14
TP_ATR_MULT       = 1.5

# Always fetch raw 1-second bars from Breeze; resampling is done locally.
INTERVAL          = "1second"

# Candle size (in seconds) used for the strategy.
# Set to 1 to use raw 1-second bars without any resampling.
RESAMPLE_SECONDS  = 120

# Print the final resampled DataFrame (with indicators) alongside the trades
# for each option contract before the summary report.
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

    strategy = TomukasScaleInOptionStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        ema_fast_period=EMA_FAST_PERIOD,
        ema_slow_period=EMA_SLOW_PERIOD,
        sweep_lookback=SWEEP_LOOKBACK,
        entry_weights=ENTRY_WEIGHTS,
        atr_period=ATR_PERIOD,
        tp_atr_mult=TP_ATR_MULT,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        resample_seconds=RESAMPLE_SECONDS,
        print_resampled=PRINT_RESAMPLED,
        cache_only=CACHE_ONLY,
        market_holidays=MARKET_HOLIDAYS,
        per_day_atm=PER_DAY_ATM,
    )

    expiry_results = strategy.run_weekly_backtest()
    TomukasScaleInOptionStrategy.print_report(expiry_results)
