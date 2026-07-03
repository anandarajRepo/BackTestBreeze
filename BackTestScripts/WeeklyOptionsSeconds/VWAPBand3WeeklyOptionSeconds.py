"""
Nifty 50 Weekly Options — VWAP Band 3 Mean-Reversion Backtest  (1-second data)
==============================================================================

Port of the TradingView Pine v6 "VWAP Band 3 Mean-Reversion Strategy" applied
to the option premium series (CE & PE legs):

Entry (resting limit orders at the bands, filled intrabar on a touch):
  LONG  — option price touches the LOWER Band 3 (VWAP − 3 stdev)
  SHORT — option price touches the UPPER Band 3 (VWAP + 3 stdev)
  Only one position open at a time per symbol; no new entries while a trade
  is active.

Exit:
  - Take-profit at the VWAP value captured at entry (fixed for the trade)
  - Stop-loss the same distance on the other side of the entry (1:1 R:R)
  - Square-off at 15:20 IST
  - No new entries after 14:45, or before TRADING_DELAY_HOURS have passed
    since the session start (lets the bands develop)
  - Max 5 trades per day per symbol

Position sizing:
  qty = equity × RISK_PERCENT ÷ (band-to-VWAP distance), so a stop-loss hit
  loses exactly RISK_PERCENT of current equity (compounds with realized PnL).

Usage:
  1. Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  2. Set RESAMPLE_SECONDS to the desired candle size for the strategy:
       1  → raw 1-second bars (no resampling)
       5  → 5-second bars
       … any positive integer is accepted.
  Data is always fetched as 1-second bars from Breeze; resampling is done
  locally before the bands and strategy logic run.
"""

import os
from datetime import date

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.vwap_band3_option_strategy import VWAPBand3OptionStrategy

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

CAPITAL           = 100_000.0       # starting equity per contract

# Band 3 settings. The bands sit BAND_MULT away from the session VWAP:
#   "Standard Deviation" → BAND_MULT volume-weighted stdevs of hlc3
#   "Percentage"         → BAND_MULT percent of the VWAP (1.0 = 1%)
BAND_MULT         = 3.0
CALC_MODE         = "Standard Deviation"

# Risk % of equity per trade. Position size is calculated so that a stop-loss
# hit loses exactly this % of current equity.
RISK_PERCENT      = 5.0

# No new entries until this many hours have passed since the session start
# (9:15 IST), giving the VWAP bands time to develop. The Pine default of 9h
# suits 24h markets; NSE's session is only 6h15m, so a smaller value is used.
TRADING_DELAY_HOURS = 2.0

# When False, upper-band touches (short the option premium) are ignored and
# only lower-band LONG entries are taken.
ALLOW_SHORT       = False

# Always fetch raw 1-second bars from Breeze; resampling is done locally.
INTERVAL          = "1second"

# Candle size (in seconds) used for the strategy.
# Set to 1 to use raw 1-second bars without any resampling.
RESAMPLE_SECONDS  = 1

# Print the final resampled DataFrame (with VWAP bands) alongside the trades
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

    strategy = VWAPBand3OptionStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        band_mult=BAND_MULT,
        calc_mode=CALC_MODE,
        risk_percent=RISK_PERCENT,
        trading_delay_hours=TRADING_DELAY_HOURS,
        allow_short=ALLOW_SHORT,
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
    VWAPBand3OptionStrategy.print_report(expiry_results)
