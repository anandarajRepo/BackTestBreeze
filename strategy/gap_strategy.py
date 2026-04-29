"""
Gap Up / Gap Down Trading Strategy — modular implementation for Breeze API.

Gap Up  (gap_pct >= threshold) → BUY
Gap Down (gap_pct <= -threshold) → SELL (short)

Exit: target or stop-loss hit intraday, or square-off at market close.
"""

from collections import defaultdict
from datetime import datetime

from models.trading_models import (
    BacktestTradeResult,
    GapSignal,
    TradeDirection,
    TradeResult,
)
from services.gap_trend_service import GapTrendService
from strategy.order_manager import OrderManager


class GapStrategy:
    def __init__(
        self,
        gap_trend_service: GapTrendService,
        order_manager: OrderManager,
        stock_code: str,
        exchange_code: str,
        quantity: int,
        gap_pct: float = 0.5,
        target_pct: float = 1.0,
        stop_loss_pct: float = 0.5,
        start_date: str = "",
        end_date: str = "",
        interval: str = "1minute",
    ):
        self.gap_trend_service = gap_trend_service
        self.order_manager = order_manager
        self.stock_code = stock_code
        self.exchange_code = exchange_code
        self.quantity = quantity
        self.gap_pct = gap_pct
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

    def _compute_levels(self, signal: GapSignal) -> tuple[float, float]:
        entry = signal.today_open
        if signal.direction == TradeDirection.BUY:
            target = round(entry * (1 + self.target_pct / 100), 2)
            stop_loss = round(entry * (1 - self.stop_loss_pct / 100), 2)
        else:
            target = round(entry * (1 - self.target_pct / 100), 2)
            stop_loss = round(entry * (1 + self.stop_loss_pct / 100), 2)
        return target, stop_loss

    def _simulate_exit(
        self,
        direction: TradeDirection,
        target: float,
        stop_loss: float,
        day_candles: list[dict],
    ) -> tuple[float, str]:
        """
        Walk through intraday candles and return (exit_price, exit_reason).
        For each candle, stop-loss takes priority if both levels are touched.
        Falls back to closing price of the last candle.
        """
        for candle in day_candles:
            high = float(candle["high"])
            low = float(candle["low"])

            if direction == TradeDirection.BUY:
                if low <= stop_loss:
                    return stop_loss, "stop_loss"
                if high >= target:
                    return target, "target"
            else:  # SELL / short
                if high >= stop_loss:
                    return stop_loss, "stop_loss"
                if low <= target:
                    return target, "target"

        return float(day_candles[-1]["close"]), "close"

    def _group_candles_by_date(self, candles: list[dict]) -> dict:
        days: dict = defaultdict(list)
        for candle in candles:
            dt = datetime.fromisoformat(candle["datetime"])
            days[dt.date()].append(candle)
        return days

    def run_backtest(self) -> list[BacktestTradeResult]:
        """Iterate over every trading day in the date range and simulate trades."""
        all_candles = self.gap_trend_service.get_all_candles(
            self.stock_code, self.exchange_code, self.start_date, self.end_date, self.interval
        )

        days = self._group_candles_by_date(all_candles)
        sorted_dates = sorted(days.keys())

        results: list[BacktestTradeResult] = []
        total_pnl = 0.0

        print(f"\n{'='*65}")
        print(f"  BACKTEST: {self.stock_code} | {self.start_date} → {self.end_date}")
        print(f"  Gap threshold: {self.gap_pct}% | Target: {self.target_pct}% | SL: {self.stop_loss_pct}% | Interval: {self.interval}")
        print(f"{'='*65}\n")

        for i, trade_date in enumerate(sorted_dates):
            if i == 0:
                # Need at least one prior day for previous close
                continue

            prev_date = sorted_dates[i - 1]
            prev_candles = days[prev_date]
            today_candles = days[trade_date]

            prev_close = float(prev_candles[-1]["close"])
            today_open = float(today_candles[0]["open"])
            gap_pct = ((today_open - prev_close) / prev_close) * 100

            if gap_pct >= self.gap_pct:
                direction = TradeDirection.BUY
            elif gap_pct <= -self.gap_pct:
                direction = TradeDirection.SELL
            else:
                print(f"  {trade_date}  Gap {gap_pct:+.2f}%  → No trade")
                continue

            if direction == TradeDirection.BUY:
                target = round(today_open * (1 + self.target_pct / 100), 2)
                stop_loss = round(today_open * (1 - self.stop_loss_pct / 100), 2)
            else:
                target = round(today_open * (1 - self.target_pct / 100), 2)
                stop_loss = round(today_open * (1 + self.stop_loss_pct / 100), 2)

            exit_price, exit_reason = self._simulate_exit(direction, target, stop_loss, today_candles)

            if direction == TradeDirection.BUY:
                pnl = round((exit_price - today_open) * self.quantity, 2)
            else:
                pnl = round((today_open - exit_price) * self.quantity, 2)

            total_pnl += pnl

            result = BacktestTradeResult(
                trade_date=trade_date,
                direction=direction,
                prev_close=prev_close,
                entry_price=today_open,
                target=target,
                stop_loss=stop_loss,
                exit_price=exit_price,
                exit_reason=exit_reason,
                pnl=pnl,
                gap_pct=gap_pct,
            )
            results.append(result)

            direction_label = "BUY " if direction == TradeDirection.BUY else "SELL"
            pnl_sign = "+" if pnl >= 0 else ""
            print(
                f"  {trade_date}  Gap {gap_pct:+.2f}%  {direction_label}"
                f"  Entry {today_open:.2f}  Exit {exit_price:.2f}"
                f"  [{exit_reason:10s}]  PnL {pnl_sign}{pnl:.2f}"
            )

        print(f"\n{'='*65}")
        print(f"  Trades executed : {len(results)}")
        pnl_sign = "+" if total_pnl >= 0 else ""
        print(f"  Total PnL       : {pnl_sign}{total_pnl:.2f}")
        print(f"{'='*65}\n")

        return results

    def run(self) -> TradeResult | None:
        signal = self.gap_trend_service.build_gap_signal(
            self.stock_code, self.exchange_code, self.gap_pct,
            self.start_date, self.end_date, self.interval,
        )

        print(f"Previous close : {signal.prev_close:.2f}")
        print(f"Today's open   : {signal.today_open:.2f}")
        print(f"Gap            : {signal.gap_pct:+.2f}%")

        if signal.direction == TradeDirection.NONE:
            print(f"No significant gap ({signal.gap_pct:+.2f}%). No trade today.")
            return None

        direction_label = "UP" if signal.direction == TradeDirection.BUY else "DOWN"
        print(f"Gap {direction_label} detected ({signal.gap_pct:+.2f}%). Placing {signal.direction.value.upper()} order.")

        target, stop_loss = self._compute_levels(signal)

        order_resp = self.order_manager.place_market_order(
            signal.direction.value,
            self.stock_code,
            self.exchange_code,
            self.quantity,
        )

        print(f"{signal.direction.value.upper()} order response : {order_resp}")
        print(f"Target: {target}  |  Stop-loss: {stop_loss}")

        return TradeResult(
            signal=signal,
            entry_price=signal.today_open,
            target=target,
            stop_loss=stop_loss,
            order_response=order_resp,
        )
