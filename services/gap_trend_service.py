"""
Data-fetching service: retrieves previous close and today's open via Breeze API.
"""

from breeze_connect import BreezeConnect
from datetime import datetime, timedelta

from models.trading_models import GapBehaviourStats, GapSignal, GapType, TradeDirection


class GapTrendService:
    def __init__(self, breeze: BreezeConnect):
        self.breeze = breeze

    def get_all_candles(
        self,
        stock_code: str,
        exchange_code: str,
        start_date: str,
        end_date: str,
        interval: str = "1minute",
    ) -> list[dict]:
        """Fetch candles for the given date range at the specified interval."""
        from_dt = datetime.strptime(start_date, "%d-%b-%Y %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S.000Z")
        to_dt = datetime.strptime(end_date, "%d-%b-%Y %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S.000Z")

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
            raise ValueError(f"No historical data returned: {resp}")
        return candles

    def get_previous_close(self, stock_code: str, exchange_code: str, start_date: str, end_date: str, interval: str = "1minute") -> float:
        from_dt = datetime.strptime(start_date, "%d-%b-%Y %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S.000Z")
        to_dt = datetime.strptime(end_date, "%d-%b-%Y %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S.000Z")

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

    @staticmethod
    def classify_gap(today_open: float, prev_close: float, prev_high: float, prev_low: float) -> GapType:
        if today_open > prev_high:
            return GapType.FULL_GAP_UP
        if today_open > prev_close:
            return GapType.PARTIAL_GAP_UP
        if today_open < prev_low:
            return GapType.FULL_GAP_DOWN
        if today_open < prev_close:
            return GapType.PARTIAL_GAP_DOWN
        return GapType.NONE

    @staticmethod
    def analyse_historical_gap_behaviour(
        days: dict,
        sorted_dates: list,
        current_idx: int,
        trade_date,
        lookback_days: int,
        gap_threshold_pct: float,
        max_gap_pct: float,
    ) -> tuple[GapBehaviourStats, GapBehaviourStats]:
        """
        Look back `lookback_days` calendar days before `trade_date` and compute
        continuation/reversal rates for gap-up and gap-down events separately.
        """
        cutoff = trade_date - timedelta(days=lookback_days)

        gap_up_cont = gap_up_rev = 0
        gap_down_cont = gap_down_rev = 0

        for i in range(1, current_idx):
            d = sorted_dates[i]
            if d >= trade_date or d < cutoff:
                continue

            prev_d = sorted_dates[i - 1]
            prev_candles = days[prev_d]
            curr_candles = days[d]

            prev_close = float(prev_candles[-1]["close"])
            prev_high = max(float(c["high"]) for c in prev_candles)
            prev_low = min(float(c["low"]) for c in prev_candles)

            today_open = float(curr_candles[0]["open"])
            today_close = float(curr_candles[-1]["close"])

            gap_pct = ((today_open - prev_close) / prev_close) * 100
            abs_gap = abs(gap_pct)

            if abs_gap < gap_threshold_pct or abs_gap > max_gap_pct:
                continue

            gap_type = GapTrendService.classify_gap(today_open, prev_close, prev_high, prev_low)

            if gap_type in (GapType.FULL_GAP_UP, GapType.PARTIAL_GAP_UP):
                if today_close >= today_open:
                    gap_up_cont += 1
                else:
                    gap_up_rev += 1
            elif gap_type in (GapType.FULL_GAP_DOWN, GapType.PARTIAL_GAP_DOWN):
                if today_close <= today_open:
                    gap_down_cont += 1
                else:
                    gap_down_rev += 1

        def make_stats(cont: int, rev: int) -> GapBehaviourStats:
            total = cont + rev
            if total == 0:
                return GapBehaviourStats(0, 0, 0, 0.0, 0.0)
            return GapBehaviourStats(
                sample_count=total,
                continuation_count=cont,
                reversal_count=rev,
                continuation_rate=round(cont / total * 100, 1),
                reversal_rate=round(rev / total * 100, 1),
            )

        return make_stats(gap_up_cont, gap_up_rev), make_stats(gap_down_cont, gap_down_rev)

    def build_gap_signal(
        self,
        stock_code: str,
        exchange_code: str,
        gap_pct_threshold: float,
        start_date: str,
        end_date: str,
        interval: str = "1minute",
    ) -> GapSignal:
        prev_close = self.get_previous_close(stock_code, exchange_code, start_date, end_date, interval)
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
