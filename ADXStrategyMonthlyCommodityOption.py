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

Expiry calendar:
  MCX monthly options expire on the last Thursday of each calendar month.
  The trade window for each expiry spans the 1st of that month through expiry day.

ATM strike:
  Computed from the commodity's opening price on the first day of each
  expiry month, rounded to the commodity-specific strike interval:
    Gold        → ₹100
    Silver      → ₹500
    Crude Oil   → ₹50
    Natural Gas → ₹10

Usage:
  Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  Add or remove symbols from COMMODITIES as needed.
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
END_DATE      = "30-Apr-2026"   # format: DD-Mon-YYYY

COMMODITIES   = ["GOLD", "SILVER", "CRUDEOIL", "NATURALGAS"]

CAPITAL       = 100_000.0       # capital per contract (used for position sizing)
ADX_PERIOD    = 16              # lookback period for ADX / DI calculation
ADX_THRESHOLD = 25.0            # minimum ADX value required to enter a trade
INTERVAL      = "1minute"       # candle interval for option data

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

    expiry_results = strategy.run_monthly_backtest()
    ADXCommodityStrategy.print_report(expiry_results)
