from breeze_connect import BreezeConnect
from dotenv import load_dotenv
import os

from services.gap_trend_service import GapTrendService
from strategy.gap_strategy import GapStrategy
from strategy.order_manager import OrderManager

load_dotenv()

# ── Session ───────────────────────────────────────────────────────────────────

breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN"),
)
print("Session Generated Successfully")

# ── Configuration ─────────────────────────────────────────────────────────────

STOCK_CODE    = "OLAELE"
EXCHANGE_CODE = "NSE"
QUANTITY      = 1

# Gap filter: only trade gaps between GAP_PCT% and MAX_GAP_PCT%
GAP_PCT       = 0.5
MAX_GAP_PCT   = 5.0

# Exit levels
TARGET_PCT    = 5.0
STOP_LOSS_PCT = 5.0

# Date range and candle interval
START_DATE    = "01-Jan-2026 9:15:00"
END_DATE      = "28-Apr-2026 15:29:59"
INTERVAL      = "1day"  # Options: 1minute, 5minute, 30minute, 1day

# Historical behaviour analysis
BEHAVIOR_LOOKBACK_DAYS  = 30   # calendar days to look back per trade day
MIN_GAP_HISTORY         = 5    # minimum gap events required before trading
CONTINUATION_THRESHOLD  = 60.0 # % continuation rate needed to follow the gap
REVERSAL_THRESHOLD      = 60.0 # % reversal rate needed to fade the gap

# ── Run Strategy ──────────────────────────────────────────────────────────────

gap_trend_service = GapTrendService(breeze)
order_manager = OrderManager(breeze)

strategy = GapStrategy(
    gap_trend_service=gap_trend_service,
    order_manager=order_manager,
    stock_code=STOCK_CODE,
    exchange_code=EXCHANGE_CODE,
    quantity=QUANTITY,
    gap_pct=GAP_PCT,
    max_gap_pct=MAX_GAP_PCT,
    target_pct=TARGET_PCT,
    stop_loss_pct=STOP_LOSS_PCT,
    start_date=START_DATE,
    end_date=END_DATE,
    interval=INTERVAL,
    behavior_lookback_days=BEHAVIOR_LOOKBACK_DAYS,
    min_gap_history=MIN_GAP_HISTORY,
    continuation_threshold=CONTINUATION_THRESHOLD,
    reversal_threshold=REVERSAL_THRESHOLD,
)

strategy.run_backtest()
