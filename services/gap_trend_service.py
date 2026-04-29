"""
Data-fetching service: retrieves previous close and today's open via Breeze API.
"""

from breeze_connect import BreezeConnect
from datetime import datetime, timedelta

from models.trading_models import GapSignal, TradeDirection


class GapTrendService:
    def __init__(self, breeze: BreezeConnect):
        self.breeze = breeze

    def get_previous_close(self, stock_code: str, exchange_code: str) -> float:
        today = datetime.now()
        from_dt = (today - timedelta(days=12)).strftime("%d-%B-%Y %H:%M:%S")
        # Exclude today so the last candle is always the most recent completed session
        to_dt = (today - timedelta(days=1)).strftime("%d-%B-%Y %H:%M:%S")

        resp = self.breeze.get_historical_data_v2(
            interval="1minute",
            from_date=from_dt,
            to_date=to_dt,
            stock_code=stock_code,
            exchange_code=exchange_code,
            product_type="cash",
        )

        candles = resp.get("Success") or []
        if not candles:
            raise ValueError(f"Not enough historical data: {resp}")

        return float(candles[-1]["close"])

    def get_current_open(self, stock_code: str, exchange_code: str) -> float:
        today_str = datetime.now().strftime("%Y-%m-%dT07:00:00.000Z")
        now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000Z")

        resp = self.breeze.get_historical_data_v2(
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

    def build_gap_signal(
        self,
        stock_code: str,
        exchange_code: str,
        gap_pct_threshold: float,
    ) -> GapSignal:
        prev_close = self.get_previous_close(stock_code, exchange_code)
        today_open = self.get_current_open(stock_code, exchange_code)
        gap_pct = ((today_open - prev_close) / prev_close) * 100

        if gap_pct >= gap_pct_threshold:
            direction = TradeDirection.BUY
        elif gap_pct <= -gap_pct_threshold:
            direction = TradeDirection.SELL
        else:
            direction = TradeDirection.NONE

        return GapSignal(
            stock_code=stock_code,
            exchange_code=exchange_code,
            prev_close=prev_close,
            today_open=today_open,
            gap_pct=gap_pct,
            direction=direction,
        )
