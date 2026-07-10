"""
Nifty 50 Weekly Options — Bowman RSI Signals Backtest  (1-second data)
======================================================================

Python adaptation of the TradingView Pine script "Bowman RSI Signals
Backtester v3", run on the option premium (CE & PE legs).

Entry (long the option premium):
  buy when the fast RSI crosses UP through the oversold level (30), while the
  higher-timeframe RSI is still depressed (below HTF_RSI_CUTOFF) and,
  optionally, the price is above a higher-timeframe trend EMA.

Exit — one of four modes (EXIT_MODE):
  - "ATR_TRAILING_STOP":   trailing stop ATR_MULT × ATR(ATR_LENGTH) below the
                           peak price since entry (activates once price has
                           moved that distance above entry).
  - "BEAR_DIV_PIVOT_HIGH": close when a bearish RSI divergence (price higher,
                           RSI lower over DIV_LOOKBACK bars, RSI > 70)
                           coincides with a confirmed pivot high.
  - "PIVOT_HIGH_ONLY":     close on any confirmed pivot high.
  - "FIXED_PERCENTAGES":   fixed take-profit / stop-loss percentages
                           (FIXED_PROFIT_PCT / FIXED_LOSS_PCT).

Common intraday guards:
  - Square-off at 15:20 IST
  - No new entries before 9:30 or after 14:45
  - Max 5 trades per day per symbol

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
from strategy.bowman_rsi_option_strategy import BowmanRSIOptionStrategy

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

# Fast RSI on the option premium. The entry trigger is a fresh cross UP
# through RSI_OVERSOLD (Pine: ta.crossover(rsi, 30) with rsiLength=7).
RSI_LENGTH        = 7
RSI_OVERSOLD      = 30.0

# Bearish-divergence / pivot detection lookback (Pine: divLookback = 2, used
# for both the divergence comparison and the pivot-high left/right width).
DIV_LOOKBACK      = 2

# Higher-timeframe RSI filter (Pine: RSI(5) on the 1D timeframe, cutoff 40).
# Intraday adaptation: the HTF series is built by resampling the option's own
# candles to HTF_SECONDS buckets; only completed HTF bars are used.
HTF_SECONDS       = 3600            # 1-hour higher-timeframe bars
HTF_RSI_LENGTH    = 5
HTF_RSI_CUTOFF    = 40.0

# Optional EMA trend filter (Pine: EMA(200) on the 240-minute timeframe).
# Intraday adaptation: EMA_LENGTH-period EMA on EMA_SECONDS resampled bars;
# entries require price above the EMA.
USE_EMA_FILTER    = True
EMA_SECONDS       = 1800            # 30-minute bars for the trend EMA
EMA_LENGTH        = 200

# Exit strategy — one of:
#   "ATR_TRAILING_STOP", "BEAR_DIV_PIVOT_HIGH",
#   "PIVOT_HIGH_ONLY",   "FIXED_PERCENTAGES"
EXIT_MODE         = "ATR_TRAILING_STOP"

# ATR trailing stop parameters (used only in ATR_TRAILING_STOP mode).
ATR_LENGTH        = 14
ATR_MULT          = 3.0

# Fixed-percentage exit levels (used only in FIXED_PERCENTAGES mode).
FIXED_PROFIT_PCT  = 2.0
FIXED_LOSS_PCT    = 1.0

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

    strategy = BowmanRSIOptionStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        rsi_length=RSI_LENGTH,
        rsi_oversold=RSI_OVERSOLD,
        div_lookback=DIV_LOOKBACK,
        htf_seconds=HTF_SECONDS,
        htf_rsi_length=HTF_RSI_LENGTH,
        htf_rsi_cutoff=HTF_RSI_CUTOFF,
        use_ema_filter=USE_EMA_FILTER,
        ema_seconds=EMA_SECONDS,
        ema_length=EMA_LENGTH,
        exit_mode=EXIT_MODE,
        atr_length=ATR_LENGTH,
        atr_mult=ATR_MULT,
        fixed_profit_pct=FIXED_PROFIT_PCT,
        fixed_loss_pct=FIXED_LOSS_PCT,
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
    BowmanRSIOptionStrategy.print_report(expiry_results)
