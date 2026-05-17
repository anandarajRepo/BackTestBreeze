"""
MCX Monthly Commodity Options — ADX DI+/DI- Crossover Backtest
===============================================================

Instruments:
  Gold (GOLD), Silver (SILVER), Crude Oil (CRUDEOIL), Natural Gas (NATURALGAS)
  — all traded on MCX.

Entry:
  CE — buy when DI+ crosses above DI- and ADX >= ADX_THRESHOLD
  PE — buy when DI- crosses above DI+ and ADX >= ADX_THRESHOLD

Exit:
  - DI direction reversal (crossover flips)
  - Square-off at 23:25 IST (MCX evening session close)
  - No new entries before 09:00 or after 22:45
  - Max 5 trades per day per symbol

GOLD contract structure (MCX):
  Futures expiry : 5th of even months (Feb/Apr/Jun/Aug/Oct/Dec), adjusted for
                   weekends/holidays — e.g. Apr 2 and Dec 4 in 2026.
  Option expiry  : official MCX dates per GOLD_OPTION_EXPIRY_DATES lookup
                   (e.g. Dec 31, Mar 24, Apr 30, Jun 30, Aug 31, Sep 23,
                   Oct 30, Nov 25 in 2025-2026); falls back to 27th adjusted
                   for weekends/holidays for months not in the lookup table.
  ATM strike     : recomputed every trading day from the active futures price
                   (rounded to nearest ₹100 interval)
  Trade window   : 28th of previous month → option expiry of current month

SILVER contract structure (MCX):
  Futures expiry : 5th of every month, adjusted for weekends/holidays per
                   SILVER_FUTURES_EXPIRY_DATES lookup.
  Option expiry  : official MCX dates per SILVER_OPTION_EXPIRY_DATES lookup;
                   falls back to 27th adjusted for weekends/holidays.
  ATM strike     : recomputed every trading day from the active futures price
                   (rounded to nearest ₹500 interval)
  Trade window   : 28th of previous month → option expiry of current month

Other commodities still use the last-Thursday expiry calendar with a
fixed ATM derived from the first day of the expiry month.

Usage:
  Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  Add or remove symbols from COMMODITIES as needed.
  Set GOLD_DAILY_ATM = True (default) to use the new daily-ATM logic for GOLD.
  Set SILVER_DAILY_ATM = True (default) to use the daily-ATM logic for SILVER.
"""

import os

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.commodity_option_service import CommodityOptionService
from strategy.adx_commodity_strategy import ADXCommodityStrategy

load_dotenv()

# ── Session ───────────────────────────────────────────────────────────────────

breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN"),
)
print("Session Generated Successfully\n")

# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE    = "01-Jan-2026"   # format: DD-Mon-YYYY
END_DATE      = "30-May-2026"   # format: DD-Mon-YYYY

COMMODITIES   = ["GOLD", "SILVER"]

CAPITAL       = 100_000.0       # capital per contract (used for position sizing)
ADX_PERIOD    = 16              # lookback period for ADX / DI calculation
ADX_THRESHOLD = 25.0            # minimum ADX value required to enter a trade
INTERVAL      = "1minute"       # candle interval for option data

# When True and COMMODITIES contains "GOLD", use the daily-ATM logic:
#   futures expire 5th of even months, options expire 27th, ATM recomputed each trading day.
GOLD_DAILY_ATM = True

# When True and COMMODITIES contains "SILVER", use the daily-ATM logic:
#   futures expire 5th of every month, options expire 27th, ATM recomputed each trading day.
SILVER_DAILY_ATM = True

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    commodity_service = CommodityOptionService(breeze)

    strategy = ADXCommodityStrategy(
        commodity_service=commodity_service,
        commodities=COMMODITIES,
        capital=CAPITAL,
        adx_period=ADX_PERIOD,
        adx_threshold=ADX_THRESHOLD,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
    )

    all_expiry_results = []

    if GOLD_DAILY_ATM and "GOLD" in COMMODITIES:
        # GOLD: daily ATM from active futures (even months), options expire 27th
        gold_results = strategy.run_gold_daily_atm_backtest()
        all_expiry_results.extend(gold_results)

    if SILVER_DAILY_ATM and "SILVER" in COMMODITIES:
        # SILVER: daily ATM from active futures (every month), options expire 27th
        silver_results = strategy.run_silver_daily_atm_backtest()
        all_expiry_results.extend(silver_results)

    other_commodities = [
        c for c in COMMODITIES
        if not (c == "GOLD" and GOLD_DAILY_ATM)
        and not (c == "SILVER" and SILVER_DAILY_ATM)
    ]
    if other_commodities:
        # Other commodities: fixed monthly ATM, last-Thursday expiry
        other_strategy = ADXCommodityStrategy(
            commodity_service=commodity_service,
            commodities=other_commodities,
            capital=CAPITAL,
            adx_period=ADX_PERIOD,
            adx_threshold=ADX_THRESHOLD,
            start_date=START_DATE,
            end_date=END_DATE,
            interval=INTERVAL,
        )
        other_results = other_strategy.run_monthly_backtest()
        all_expiry_results.extend(other_results)

    ADXCommodityStrategy.print_report(all_expiry_results)
