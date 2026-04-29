"""
Gap Up / Gap Down Trading Strategy — modular implementation for Breeze API.

Phase 1 — analyse historical gap behaviour over a lookback window.
Phase 2 — decide trade direction based on continuation/reversal rates.
Phase 3 — enter at open, exit via target/stop-loss or market close.
"""

from collections import defaultdict
from datetime import datetime

from models.trading_models import (
    BacktestTradeResult,
    GapSignal,
    GapType,
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
        max_gap_pct: float = 5.0,
        target_pct: float = 1.0,
        stop_loss_pct: float = 0.5,
        start_date: str = "",
        end_date: str = "",
        interval: str = "1minute",
        behavior_lookback_days: int = 30,
        min_gap_history: int = 5,
        continuation_threshold: float = 60.0,
        reversal_threshold: float = 60.0,
    ):
        self.gap_trend_service = gap_trend_service
        self.order_manager = order_manager
        self.stock_code = stock_code
        self.exchange_code = exchange_code
        self.quantity = quantity
        self.gap_pct = gap_pct
        self.max_gap_pct = max_gap_pct
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.behavior_lookback_days = behavior_lookback_days
        self.min_gap_history = min_gap_history
        self.continuation_threshold = continuation_threshold
        self.reversal_threshold = reversal_threshold

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
        """
        Three-phase backtest per trading day:
          Phase 1 — analyse historical gap behaviour over the lookback window.
          Phase 2 — decide trade direction from continuation/reversal rates.
          Phase 3 — enter at open and exit via target, stop-loss, or market close.
        """
        all_candles = self.gap_trend_service.get_all_candles(
            self.stock_code, self.exchange_code, self.start_date, self.end_date, self.interval
        )

        days = self._group_candles_by_date(all_candles)
        sorted_dates = sorted(days.keys())

        results: list[BacktestTradeResult] = []
        total_pnl = 0.0

        print(f"\n{'='*75}")
        print(f"  BACKTEST: {self.stock_code} | {self.start_date} → {self.end_date}")
        print(
            f"  Gap: {self.gap_pct}%–{self.max_gap_pct}% | "
            f"Target: {self.target_pct}% | SL: {self.stop_loss_pct}% | "
            f"Lookback: {self.behavior_lookback_days}d | "
            f"Min history: {self.min_gap_history} | "
            f"Thresholds: cont={self.continuation_threshold}% rev={self.reversal_threshold}%"
        )
        print(f"{'='*75}\n")

        for i, trade_date in enumerate(sorted_dates):
            if i == 0:
                continue

            prev_date = sorted_dates[i - 1]
            prev_candles = days[prev_date]
            today_candles = days[trade_date]

            prev_close = float(prev_candles[-1]["close"])
            prev_high = max(float(c["high"]) for c in prev_candles)
            prev_low = min(float(c["low"]) for c in prev_candles)
            today_open = float(today_candles[0]["open"])

            gap_pct = ((today_open - prev_close) / prev_close) * 100
            abs_gap = abs(gap_pct)

            # Filter by gap range
            if abs_gap < self.gap_pct or abs_gap > self.max_gap_pct:
                print(f"  {trade_date}  Gap {gap_pct:+.2f}%  → No trade (gap out of range)")
                continue

            gap_type = GapTrendService.classify_gap(today_open, prev_close, prev_high, prev_low)

            # Phase 1 — historical behaviour analysis
            gap_up_stats, gap_down_stats = GapTrendService.analyse_historical_gap_behaviour(
                days=days,
                sorted_dates=sorted_dates,
                current_idx=i,
                trade_date=trade_date,
                lookback_days=self.behavior_lookback_days,
                gap_threshold_pct=self.gap_pct,
                max_gap_pct=self.max_gap_pct,
            )

            # Phase 2 — trade direction decision
            if gap_type in (GapType.FULL_GAP_UP, GapType.PARTIAL_GAP_UP):
                stats = gap_up_stats
                is_gap_up = True
            else:
                stats = gap_down_stats
                is_gap_up = False

            if stats.sample_count < self.min_gap_history:
                print(
                    f"  {trade_date}  Gap {gap_pct:+.2f}% [{gap_type.value}]"
                    f"  → No trade (history too small: {stats.sample_count}/{self.min_gap_history})"
                )
                continue

            if stats.continuation_rate >= self.continuation_threshold:
                direction = TradeDirection.BUY if is_gap_up else TradeDirection.SELL
                bias = f"cont {stats.continuation_rate:.0f}%"
            elif stats.reversal_rate >= self.reversal_threshold:
                direction = TradeDirection.SELL if is_gap_up else TradeDirection.BUY
                bias = f"rev {stats.reversal_rate:.0f}%"
            else:
                print(
                    f"  {trade_date}  Gap {gap_pct:+.2f}% [{gap_type.value}]"
                    f"  → No trade (no clear bias: cont={stats.continuation_rate}% rev={stats.reversal_rate}%)"
                )
                continue

            # Phase 3 — entry and exit simulation
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
                gap_type=gap_type,
                prev_close=prev_close,
                entry_price=today_open,
                target=target,
                stop_loss=stop_loss,
                exit_price=exit_price,
                exit_reason=exit_reason,
                pnl=pnl,
                gap_pct=gap_pct,
                continuation_rate=stats.continuation_rate,
                reversal_rate=stats.reversal_rate,
                gap_history_count=stats.sample_count,
            )
            results.append(result)

            direction_label = "BUY " if direction == TradeDirection.BUY else "SELL"
            pnl_sign = "+" if pnl >= 0 else ""
            print(
                f"  {trade_date}  Gap {gap_pct:+.2f}% [{gap_type.value:15s}]"
                f"  {direction_label} ({bias})"
                f"  Entry {today_open:.2f}  Exit {exit_price:.2f}"
                f"  [{exit_reason:10s}]  PnL {pnl_sign}{pnl:.2f}"
            )

        print(f"\n{'='*75}")
        print(f"  Trades executed : {len(results)}")
        pnl_sign = "+" if total_pnl >= 0 else ""
        print(f"  Total PnL       : {pnl_sign}{total_pnl:.2f}")
        print(f"{'='*75}\n")

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
