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

STOCK_CODE    = "ITC"
EXCHANGE_CODE = "NSE"
QUANTITY      = 1
GAP_PCT       = 0.5
TARGET_PCT    = 1.0
STOP_LOSS_PCT = 0.5
START_DATE    = "01-Apr-2026 9:15:00"
END_DATE      = "28-Apr-2026 15:29:59"
INTERVAL      = "1minute"  # Options: 1minute, 5minute, 30minute, 1day

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
    target_pct=TARGET_PCT,
    stop_loss_pct=STOP_LOSS_PCT,
    start_date=START_DATE,
    end_date=END_DATE,
    interval=INTERVAL,
)

strategy.run_backtest()
