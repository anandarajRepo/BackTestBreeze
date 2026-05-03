"""
Open Range Breakout (ORB) Strategy — backtest implementation for Breeze API.

Logic (mirrors FyersORB):
  Phase 1 — Build opening range from first `orb_minutes` candles (default 15).
  Phase 2 — Scan post-ORB candles for the first breakout above ORB high (BUY)
             or below ORB low (SELL).
  Phase 3 — Enter at breakout price; exit via target, stop-loss, or market close.
"""

from datetime import datetime

from models.orb_models import BreakoutDirection, OpenRange, ORBTradeResult
from services.orb_data_service import ORBDataService


class ORBStrategy:
    def __init__(
        self,
        orb_data_service: ORBDataService,
        stock_code: str,
        exchange_code: str,
        quantity: int,
        orb_minutes: int        = 15,
        stop_loss_pct: float    = 1.5,
        risk_reward_ratio: float = 2.0,
        start_date: str         = "",
        end_date: str           = "",
        interval: str           = "1minute",
    ):
        self.orb_data_service   = orb_data_service
        self.stock_code         = stock_code
        self.exchange_code      = exchange_code
        self.quantity           = quantity
        self.orb_minutes        = orb_minutes
        self.stop_loss_pct      = stop_loss_pct
        self.risk_reward_ratio  = risk_reward_ratio
        self.start_date         = start_date
        self.end_date           = end_date
        self.interval           = interval

    # ── helpers ──────────────────────────────────────────────────────────────

    def _build_open_range(self, orb_candles: list[dict]) -> OpenRange:
        high = max(float(c["high"]) for c in orb_candles)
        low  = min(float(c["low"])  for c in orb_candles)
        return OpenRange(high=high, low=low)

    def _detect_breakout(
        self,
        candle: dict,
        orb: OpenRange,
    ) -> BreakoutDirection:
        """Return direction on first candle that breaks the ORB."""
        high = float(candle["high"])
        low  = float(candle["low"])
        if high > orb.high:
            return BreakoutDirection.BUY
        if low < orb.low:
            return BreakoutDirection.SELL
        return BreakoutDirection.NONE

    def _compute_levels(
        self,
        direction: BreakoutDirection,
        entry: float,
        orb: OpenRange,
    ) -> tuple[float, float]:
        """
        Stop-loss: percentage-based from entry price.
        Target: entry +/- risk * risk_reward_ratio.
        """
        if direction == BreakoutDirection.BUY:
            stop_loss = round(entry * (1 - self.stop_loss_pct / 100), 2)
            risk      = entry - stop_loss
            target    = round(entry + risk * self.risk_reward_ratio, 2)
        else:
            stop_loss = round(entry * (1 + self.stop_loss_pct / 100), 2)
            risk      = stop_loss - entry
            target    = round(entry - risk * self.risk_reward_ratio, 2)
        return target, stop_loss

    def _simulate_exit(
        self,
        direction: BreakoutDirection,
        entry: float,
        target: float,
        stop_loss: float,
        post_breakout_candles: list[dict],
    ) -> tuple[float, str]:
        """
        Walk candles after entry; stop-loss takes priority if both levels hit.
        Falls back to closing price of the last candle (market close).
        """
        for candle in post_breakout_candles:
            high = float(candle["high"])
            low  = float(candle["low"])

            if direction == BreakoutDirection.BUY:
                if low <= stop_loss:
                    return stop_loss, "stop_loss"
                if high >= target:
                    return target, "target"
            else:
                if high >= stop_loss:
                    return stop_loss, "stop_loss"
                if low <= target:
                    return target, "target"

        return float(post_breakout_candles[-1]["close"]), "close"

    # ── backtest ──────────────────────────────────────────────────────────────

    def run_backtest(self) -> list[ORBTradeResult]:
        all_candles = self.orb_data_service.get_intraday_candles(
            self.stock_code, self.exchange_code,
            self.start_date, self.end_date, self.interval,
        )

        days         = ORBDataService.group_by_date(all_candles)
        sorted_dates = sorted(days.keys())

        results: list[ORBTradeResult] = []
        total_pnl = 0.0

        print(f"\n{'='*80}")
        print(f"  ORB BACKTEST: {self.stock_code} | {self.start_date} → {self.end_date}")
        print(
            f"  ORB period: {self.orb_minutes} min | "
            f"SL: {self.stop_loss_pct}% | "
            f"RR: 1:{self.risk_reward_ratio} | "
            f"Qty: {self.quantity}"
        )
        print(f"{'='*80}\n")

        for trade_date in sorted_dates:
            day_candles = days[trade_date]

            if len(day_candles) <= self.orb_minutes:
                print(f"  {trade_date}  → Skipped (insufficient candles: {len(day_candles)})")
                continue

            orb_candles      = ORBDataService.get_orb_candles(day_candles, self.orb_minutes)
            post_orb_candles = ORBDataService.get_post_orb_candles(day_candles, self.orb_minutes)
            orb              = self._build_open_range(orb_candles)

            # Scan post-ORB candles for first breakout
            breakout_idx = None
            direction    = BreakoutDirection.NONE
            for idx, candle in enumerate(post_orb_candles):
                direction = self._detect_breakout(candle, orb)
                if direction != BreakoutDirection.NONE:
                    breakout_idx = idx
                    break

            if direction == BreakoutDirection.NONE or breakout_idx is None:
                print(f"  {trade_date}  ORB [{orb.low:.2f} – {orb.high:.2f}]  → No breakout")
                continue

            breakout_candle = post_orb_candles[breakout_idx]
            breakout_time   = breakout_candle["datetime"]

            # Entry at ORB boundary that was broken
            entry = orb.high if direction == BreakoutDirection.BUY else orb.low

            target, stop_loss = self._compute_levels(direction, entry, orb)

            # Exit simulation on candles after the breakout candle
            remaining = post_orb_candles[breakout_idx + 1:]
            if not remaining:
                exit_price  = float(breakout_candle["close"])
                exit_reason = "close"
            else:
                exit_price, exit_reason = self._simulate_exit(
                    direction, entry, target, stop_loss, remaining
                )

            if direction == BreakoutDirection.BUY:
                pnl = round((exit_price - entry) * self.quantity, 2)
            else:
                pnl = round((entry - exit_price) * self.quantity, 2)

            total_pnl += pnl

            result = ORBTradeResult(
                trade_date     = trade_date,
                direction      = direction,
                orb_high       = orb.high,
                orb_low        = orb.low,
                entry_price    = entry,
                target         = target,
                stop_loss      = stop_loss,
                exit_price     = exit_price,
                exit_reason    = exit_reason,
                pnl            = pnl,
                breakout_time  = breakout_time,
            )
            results.append(result)

            dir_label = "BUY " if direction == BreakoutDirection.BUY else "SELL"
            pnl_sign  = "+" if pnl >= 0 else ""
            bt_time   = datetime.fromisoformat(breakout_time).strftime("%H:%M")
            print(
                f"  {trade_date}  ORB [{orb.low:.2f}–{orb.high:.2f}]  "
                f"{dir_label} @{bt_time}"
                f"  Entry {entry:.2f}  T {target:.2f}  SL {stop_loss:.2f}"
                f"  Exit {exit_price:.2f} [{exit_reason:10s}]"
                f"  PnL {pnl_sign}{pnl:.2f}"
            )

        print(f"\n{'='*80}")
        print(f"  Trades executed : {len(results)}")
        pnl_sign = "+" if total_pnl >= 0 else ""
        print(f"  Total PnL       : {pnl_sign}{total_pnl:.2f}")
        print(f"{'='*80}\n")

        return results
