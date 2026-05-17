"""
MCX GOLD Futures — ADX DI+/DI- Crossover Backtest
===================================================

Instruments:
  GOLD futures on MCX.
  Futures contracts exist for even months only (Feb/Apr/Jun/Aug/Oct/Dec).

Entry:
  LONG  — buy when DI+ crosses above DI- and ADX >= ADX_THRESHOLD
  SHORT — sell when DI- crosses above DI+ and ADX >= ADX_THRESHOLD

Exit:
  - DI direction reversal (crossover flips)
  - Square-off at 23:25 IST (MCX evening session close)
  - No new entries before 09:00 or after 22:45
  - Max 5 trades per day

GOLD futures contract structure (MCX):
  Expiry   : 5th of even months (Feb/Apr/Jun/Aug/Oct/Dec), adjusted for
             weekends/holidays per GOLD_FUTURES_EXPIRY_DATES lookup.
  Lot size : 1 kg (configurable via LOT_SIZE)
  Window   : 1st of contract month → futures expiry date

Usage:
  Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  Adjust LOT_SIZE to match the lot size used by your broker/exchange.
"""

import os

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.commodity_option_service import CommodityOptionService
from strategy.adx_gold_futures_strategy import ADXGoldFuturesStrategy

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

CAPITAL       = 100_000.0       # capital per position (used for lot sizing)
ADX_PERIOD    = 16              # lookback period for ADX / DI calculation
ADX_THRESHOLD = 25.0            # minimum ADX value required to enter a trade
INTERVAL      = "1minute"       # candle interval for futures data
LOT_SIZE      = 1               # MCX GOLD standard lot = 1 kg

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    commodity_service = CommodityOptionService(breeze)

    strategy = ADXGoldFuturesStrategy(
        commodity_service=commodity_service,
        capital=CAPITAL,
        adx_period=ADX_PERIOD,
        adx_threshold=ADX_THRESHOLD,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        lot_size=LOT_SIZE,
    )

    results = strategy.run_backtest()
    ADXGoldFuturesStrategy.print_report(results)
