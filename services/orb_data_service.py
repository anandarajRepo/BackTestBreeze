"""
Intraday data service for Open Range Breakout backtest via Breeze API.
"""

from collections import defaultdict
from datetime import datetime, timedelta

from breeze_connect import BreezeConnect


class ORBDataService:
    def __init__(self, breeze: BreezeConnect):
        self.breeze = breeze

    def get_intraday_candles(
        self,
        stock_code: str,
        exchange_code: str,
        start_date: str,
        end_date: str,
        interval: str = "1second",
    ) -> list[dict]:
        """Fetch minute-level candles for the given date range."""
        from_dt = datetime.strptime(start_date, "%d-%b-%Y %H:%M:%S")
        to_dt   = datetime.strptime(end_date,   "%d-%b-%Y %H:%M:%S")

        resp = self.breeze.get_historical_data_v2(
            interval=interval,
            from_date=from_dt,
            to_date=to_dt,
            stock_code=stock_code,
            exchange_code=exchange_code,
            product_type="cash",
        )

        candles = resp.get("Success") or []
        if not candles:
            raise ValueError(f"No intraday data returned for {stock_code}: {resp}")
        return candles

    @staticmethod
    def group_by_date(candles: list[dict]) -> dict:
        """Group candle list into {date: [candles]} ordered by time."""
        days: dict = defaultdict(list)
        for candle in candles:
            dt = datetime.fromisoformat(candle["datetime"])
            days[dt.date()].append(candle)
        # Ensure each day's candles are in chronological order
        for d in days:
            days[d].sort(key=lambda c: c["datetime"])
        return days

    @staticmethod
    def get_orb_candles(day_candles: list[dict], orb_minutes: int) -> list[dict]:
        """Return candles within the first orb_minutes of the trading day (time-based, not count-based)."""
        if not day_candles:
            return []
        first_dt = datetime.fromisoformat(day_candles[0]["datetime"])
        cutoff = first_dt + timedelta(minutes=orb_minutes)
        return [c for c in day_candles if datetime.fromisoformat(c["datetime"]) < cutoff]

    @staticmethod
    def get_post_orb_candles(day_candles: list[dict], orb_minutes: int) -> list[dict]:
        """Return candles after the ORB formation period (time-based, not count-based)."""
        if not day_candles:
            return []
        first_dt = datetime.fromisoformat(day_candles[0]["datetime"])
        cutoff = first_dt + timedelta(minutes=orb_minutes)
        return [c for c in day_candles if datetime.fromisoformat(c["datetime"]) >= cutoff]
