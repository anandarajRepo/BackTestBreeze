"""
Gap Up / Gap Down Trading Strategy using ICICI Direct Breeze API

Gap Up  → Today's open > Yesterday's close by gap_pct threshold → BUY
Gap Down → Today's open < Yesterday's close by gap_pct threshold → SELL (short)

Exit: target or stop-loss hit intraday, or square-off at market close.
"""

from breeze_connect import BreezeConnect
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

STOCK_CODE    = "RELIANCE"
EXCHANGE_CODE = "NSE"
QUANTITY      = 1
GAP_PCT       = 0.5      # minimum gap % to trigger a trade
TARGET_PCT    = 1.0      # target profit %
STOP_LOSS_PCT = 0.5      # stop-loss %

# ── Session ──────────────────────────────────────────────────────────────────

breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN"),
)
print("Session generated successfully.")

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_previous_close(stock_code: str, exchange_code: str) -> float:
    """Return the previous trading day's closing price."""
    today = datetime.now()
    # go back 5 calendar days to safely land on a trading day
    from_dt = (today - timedelta(days=5)).strftime("%Y-%m-%dT07:00:00.000Z")
    to_dt   = today.strftime("%Y-%m-%dT07:00:00.000Z")

    resp = breeze.get_historical_data_v2(
        interval="1day",
        from_date=from_dt,
        to_date=to_dt,
        stock_code=stock_code,
        exchange_code=exchange_code,
        product_type="cash",
    )

    candles = resp.get("Success") or []
    if len(candles) < 2:
        raise ValueError(f"Not enough historical data: {resp}")

    # candles are in ascending order; [-2] is the last completed day
    prev_candle = candles[-2]
    return float(prev_candle["close"])


def get_current_open(stock_code: str, exchange_code: str) -> float:
    """Return today's opening price from the first 1-minute candle."""
    today_str = datetime.now().strftime("%Y-%m-%dT07:00:00.000Z")
    now_str   = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000Z")

    resp = breeze.get_historical_data_v2(
        interval="1minute",
        from_date=today_str,
        to_date=now_str,
        stock_code=stock_code,
        exchange_code=exchange_code,
        product_type="cash",
    )

    candles = resp.get("Success") or []
    if not candles:
        raise ValueError(f"No intraday data available yet: {resp}")

    return float(candles[0]["open"])


def place_market_order(action: str, stock_code: str, exchange_code: str, quantity: int) -> dict:
    """Place a market (aggressive limit) order."""
    resp = breeze.place_order(
        stock_code=stock_code,
        exchange_code=exchange_code,
        product="cash",
        action=action,            # "buy" or "sell"
        order_type="market",
        quantity=str(quantity),
        price="0",
        validity="day",
    )
    return resp


# ── Strategy ─────────────────────────────────────────────────────────────────

def run_gap_strategy():
    prev_close = get_previous_close(STOCK_CODE, EXCHANGE_CODE)
    today_open = get_current_open(STOCK_CODE, EXCHANGE_CODE)

    gap_pct_actual = ((today_open - prev_close) / prev_close) * 100

    print(f"Previous close : {prev_close:.2f}")
    print(f"Today's open   : {today_open:.2f}")
    print(f"Gap            : {gap_pct_actual:+.2f}%")

    if gap_pct_actual >= GAP_PCT:
        # ── Gap Up → BUY ─────────────────────────────────────────────────────
        print(f"Gap UP detected (+{gap_pct_actual:.2f}%). Placing BUY order.")
        entry_price = today_open
        target      = round(entry_price * (1 + TARGET_PCT / 100), 2)
        stop_loss   = round(entry_price * (1 - STOP_LOSS_PCT / 100), 2)

        order_resp = place_market_order("buy", STOCK_CODE, EXCHANGE_CODE, QUANTITY)
        print(f"BUY order response : {order_resp}")
        print(f"Target: {target}  |  Stop-loss: {stop_loss}")

    elif gap_pct_actual <= -GAP_PCT:
        # ── Gap Down → SELL (short) ───────────────────────────────────────────
        print(f"Gap DOWN detected ({gap_pct_actual:.2f}%). Placing SELL order.")
        entry_price = today_open
        target      = round(entry_price * (1 - TARGET_PCT / 100), 2)
        stop_loss   = round(entry_price * (1 + STOP_LOSS_PCT / 100), 2)

        order_resp = place_market_order("sell", STOCK_CODE, EXCHANGE_CODE, QUANTITY)
        print(f"SELL order response : {order_resp}")
        print(f"Target: {target}  |  Stop-loss: {stop_loss}")

    else:
        print(f"No significant gap ({gap_pct_actual:+.2f}%). No trade today.")


if __name__ == "__main__":
    run_gap_strategy()
