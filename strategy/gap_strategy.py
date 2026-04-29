"""
Gap Up / Gap Down Trading Strategy — modular implementation for Breeze API.

Gap Up  (gap_pct >= threshold) → BUY
Gap Down (gap_pct <= -threshold) → SELL (short)

Exit: target or stop-loss hit intraday, or square-off at market close.
"""

from models.trading_models import GapSignal, TradeDirection, TradeResult
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

    def _compute_levels(self, signal: GapSignal) -> tuple[float, float]:
        entry = signal.today_open
        if signal.direction == TradeDirection.BUY:
            target = round(entry * (1 + self.target_pct / 100), 2)
            stop_loss = round(entry * (1 - self.stop_loss_pct / 100), 2)
        else:
            target = round(entry * (1 - self.target_pct / 100), 2)
            stop_loss = round(entry * (1 + self.stop_loss_pct / 100), 2)
        return target, stop_loss

    def run(self) -> TradeResult | None:
        signal = self.gap_trend_service.build_gap_signal(
            self.stock_code, self.exchange_code, self.gap_pct,
            self.start_date, self.end_date,
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
