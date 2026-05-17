"""
MCX Commodity Monthly Futures — HTF Candle Direction Strategy Backtest
======================================================================

Based on: HTF Candle Direction Strategy V1 (TradingView)
Ref: https://www.tradingview.com/script/3kv64PEa-HTF-Candle-Direction-Strategy-V1/

Entry:
  LONG  — when HTF candle is bullish (close > open) and all filters pass
  SHORT — when HTF candle is bearish (close < open) and all filters pass

Filters (optional):
  EMA    — LONG only when price > EMA; SHORT only when price < EMA
  Volume — trade only when volume > avg_volume * VOLUME_MULTIPLIER

Exit:
  - Square-off at 23:25 IST (MCX evening session close)
  - No new entries before 09:00 or after 22:45
  - One trade per day

Supported commodities: GOLD, SILVER, CRUDEOIL, NATURALGAS

GOLD futures contract structure (MCX):
  Expiry : 5th of even months (Feb/Apr/Jun/Aug/Oct/Dec), adjusted for
           weekends/holidays.
  Lot size: 1 kg

SILVER futures contract structure (MCX):
  Expiry : 5th of every month, adjusted for weekends/holidays.
  Lot size: 30 kg (mini) or 1 kg (micro) — set via LOT_SIZE.

Usage:
  Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  Adjust HTF_INTERVAL to set the higher timeframe candle (e.g. "1day", "30minute").
  Set EMA_PERIOD to 0 to disable the EMA filter.
  Set USE_VOLUME_FILTER = False to disable the volume filter.
  TRADE_DIRECTION controls which side to trade: "BOTH", "LONG_ONLY", or "SHORT_ONLY".
"""

import os

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.commodity_option_service import CommodityOptionService
from strategy.htf_candle_futures_strategy import HTFCandleFuturesStrategy

load_dotenv()

# ── Session ───────────────────────────────────────────────────────────────────

breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN"),
)
print("Session Generated Successfully\n")

# ── Configuration ─────────────────────────────────────────────────────────────

COMMODITY          = "GOLD"          # "GOLD" | "SILVER" | "CRUDEOIL" | "NATURALGAS"
LOT_SIZE           = 1               # MCX GOLD standard lot = 1 kg

START_DATE         = "01-Jan-2026"   # format: DD-Mon-YYYY
END_DATE           = "05-May-2026"   # format: DD-Mon-YYYY

CAPITAL            = 100_000.0       # capital per position (used for lot sizing)

HTF_INTERVAL       = "1day"          # higher timeframe: "1day", "30minute", "1hour"
LF_INTERVAL        = "5minute"       # lower timeframe for execution and filters

EMA_PERIOD         = 21              # EMA period for trend filter (0 = disabled)
USE_VOLUME_FILTER  = False           # enable/disable volume filter
VOLUME_AVG_PERIOD  = 20              # lookback period for average volume
VOLUME_MULTIPLIER  = 1.5             # current volume must exceed avg * this value

TRADE_DIRECTION    = "BOTH"          # "BOTH" | "LONG_ONLY" | "SHORT_ONLY"

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    commodity_service = CommodityOptionService(breeze)

    strategy = HTFCandleFuturesStrategy(
        commodity_service=commodity_service,
        commodity=COMMODITY,
        lot_size=LOT_SIZE,
        capital=CAPITAL,
        htf_interval=HTF_INTERVAL,
        lf_interval=LF_INTERVAL,
        ema_period=EMA_PERIOD,
        use_volume_filter=USE_VOLUME_FILTER,
        volume_avg_period=VOLUME_AVG_PERIOD,
        volume_multiplier=VOLUME_MULTIPLIER,
        trade_direction=TRADE_DIRECTION,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    expiry_results = strategy.run_monthly_backtest()
    HTFCandleFuturesStrategy.print_report(expiry_results)
