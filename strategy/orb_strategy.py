"""
Open Range Breakout (ORB) Strategy — backtest implementation for Breeze API.

Logic (mirrors FyersORB):
  Phase 0 — Pre-market: compute momentum score and historical trend direction.
             Trades are skipped when momentum is below *min_momentum_score* or
             when the breakout direction opposes the prevailing trend.
  Phase 1 — Build opening range from first `orb_minutes` candles (default 15).
  Phase 2 — Crossover Skip Logic: skip the first ORB crossover; only trade the
             second same-direction crossover (provided ≥30 s have elapsed).
             An opposite-direction crossover resets the counter entirely.
  Phase 3 — Validate signal with Breakout Quality Score (0–100), gating on
             `min_breakout_quality`.
  Phase 4 — Enter at breakout price.
  Phase 5 — Position management:
               • 1 % take-profit trigger: exit 50 % of position, SL → breakeven.
               • Trailing stop: activates after `trailing_activation_pct` move,
                 trails `trailing_distance_pct` behind the best price.
               • Time exit: forced close at or after 3:10 PM.
               • Hard SL / target as a final backstop.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from models.orb_models import BreakoutDirection, OpenRange, ORBTradeResult
from services.orb_data_service import ORBDataService
from services.momentum_service import MomentumScore, MomentumScoringService
from services.trend_direction_service import (
    TrendAnalysis,
    TrendDirection,
    TrendDirectionService,
)

logger = logging.getLogger(__name__)


@dataclass
class _ExitResult:
    exit_price: float
    exit_reason: str
    pnl: float
    partial_exit_price: Optional[float] = None
    partial_exit_qty: int = 0


class ORBStrategy:
    def __init__(
        self,
        orb_data_service: ORBDataService,
        stock_code: str,
        exchange_code: str,
        quantity: int,
        orb_minutes: int             = 15,
        stop_loss_pct: float         = 1.5,
        risk_reward_ratio: float     = 2.0,
        start_date: str              = "",
        end_date: str                = "",
        interval: str                = "1minute",
        # ── Momentum filter ───────────────────────────────────────────
        momentum_service: Optional[MomentumScoringService] = None,
        min_momentum_score: float    = 50.0,
        momentum_lookback_days: int  = 30,
        # ── Trend filter ──────────────────────────────────────────────
        trend_service: Optional[TrendDirectionService] = None,
        trend_filter: bool           = True,
        trend_filter_mode: str       = "STRICT",   # "STRICT" | "LENIENT"
        trend_lookback_days: int     = 10,
        historical_weight: float     = 0.6,
        intraday_weight: float       = 0.4,
        # ── Breakout quality ──────────────────────────────────────────
        min_breakout_quality: float  = 0.0,   # 0 = disabled; set e.g. 40 to filter weak breakouts
        # ── Position management ───────────────────────────────────────
        enable_partial_exit: bool            = True,
        partial_exit_trigger_pct: float      = 1.0,   # % move to trigger 50 % exit
        enable_trailing_stop: bool           = True,
        trailing_activation_pct: float       = 1.5,   # % move to activate trailing stop
        trailing_distance_pct: float         = 0.75,  # trail this far behind best price
        time_exit_hour: int                  = 15,
        time_exit_minute: int                = 10,
    ):
        self.orb_data_service    = orb_data_service
        self.stock_code          = stock_code
        self.exchange_code       = exchange_code
        self.quantity            = quantity
        self.orb_minutes         = orb_minutes
        self.stop_loss_pct       = stop_loss_pct
        self.risk_reward_ratio   = risk_reward_ratio
        self.start_date          = start_date
        self.end_date            = end_date
        self.interval            = interval

        self.momentum_service       = momentum_service
        self.min_momentum_score     = min_momentum_score
        self.momentum_lookback_days = momentum_lookback_days

        self.trend_service       = trend_service
        self.trend_filter        = trend_filter
        self.trend_filter_mode   = trend_filter_mode
        self.trend_lookback_days = trend_lookback_days
        self.historical_weight   = historical_weight
        self.intraday_weight     = intraday_weight

        self.min_breakout_quality       = min_breakout_quality

        self.enable_partial_exit        = enable_partial_exit
        self.partial_exit_trigger_pct   = partial_exit_trigger_pct
        self.enable_trailing_stop       = enable_trailing_stop
        self.trailing_activation_pct    = trailing_activation_pct
        self.trailing_distance_pct      = trailing_distance_pct
        self.time_exit_hour             = time_exit_hour
        self.time_exit_minute           = time_exit_minute

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _interval_to_seconds(self) -> int:
        mapping = {
            "1minute": 60, "5minute": 300, "10minute": 600,
            "15minute": 900, "30minute": 1800, "60minute": 3600,
        }
        return mapping.get(self.interval, 60)

    def _build_open_range(self, orb_candles: list[dict]) -> OpenRange:
        high = max(float(c["high"]) for c in orb_candles)
        low  = min(float(c["low"])  for c in orb_candles)
        return OpenRange(high=high, low=low)

    def _detect_breakout(self, candle: dict, orb: OpenRange) -> BreakoutDirection:
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
        if direction == BreakoutDirection.BUY:
            stop_loss = round(entry * (1 - self.stop_loss_pct / 100), 2)
            risk      = entry - stop_loss
            target    = round(entry + risk * self.risk_reward_ratio, 2)
        else:
            stop_loss = round(entry * (1 + self.stop_loss_pct / 100), 2)
            risk      = stop_loss - entry
            target    = round(entry - risk * self.risk_reward_ratio, 2)
        return target, stop_loss

    def _is_after_time_exit(self, candle_datetime: str) -> bool:
        try:
            dt = datetime.fromisoformat(candle_datetime)
            return (dt.hour * 60 + dt.minute) >= (self.time_exit_hour * 60 + self.time_exit_minute)
        except Exception:
            return False

    # ── Phase 2: Crossover Skip Logic ────────────────────────────────────────

    def _detect_breakout_with_skip(
        self,
        post_orb_candles: list[dict],
        orb: OpenRange,
    ) -> tuple[Optional[int], BreakoutDirection, bool]:
        """
        Implements the crossover skip state machine:
          1. First crossover in any direction → skip, record direction & index.
          2. Opposite direction crossover → reset; treat as new first crossover.
          3. Second same-direction crossover with ≥30 s elapsed → tradeable signal.

        Returns (breakout_idx, direction, crossover_skip_applied).
        crossover_skip_applied=True means the skip rule was exercised (at least
        one crossover was skipped before the trade-triggering one).
        """
        interval_seconds = self._interval_to_seconds()
        min_candles_gap  = max(1, int(30 / interval_seconds))  # candles needed for 30 s

        first_dir: Optional[BreakoutDirection] = None
        first_idx: Optional[int] = None

        for idx, candle in enumerate(post_orb_candles):
            new_state = self._detect_breakout(candle, orb)
            if new_state == BreakoutDirection.NONE:
                continue

            if first_dir is None:
                # First crossover — skip and record
                first_dir = new_state
                first_idx = idx

            elif new_state != first_dir:
                # Opposite direction — reset; treat as new first crossover
                first_dir = new_state
                first_idx = idx

            else:
                # Same direction again — check 30 s gap
                elapsed_candles = idx - first_idx
                if elapsed_candles >= min_candles_gap:
                    return idx, new_state, True  # crossover_skip_applied

        # Never reached a tradeable second crossover
        return None, BreakoutDirection.NONE, False

    # ── Phase 3: Breakout Quality Score ──────────────────────────────────────

    def _compute_breakout_quality(
        self,
        direction: BreakoutDirection,
        orb: OpenRange,
        breakout_candle: dict,
        breakout_idx: int,
        total_post_orb_candles: int,
        orb_candles: list[dict],
        momentum: Optional[MomentumScore],
    ) -> float:
        """
        Score 0–100 for breakout quality from five components:
          1. Volume ratio          (25 %): breakout candle volume vs avg ORB volume
          2. Momentum distance     (25 %): how far price closed beyond the ORB boundary
          3. Range quality         (20 %): ORB range as % of mid-price
          4. RSI positioning       (20 %): from pre-market momentum score
          5. Time decay            (10 %): earlier breakouts score higher
        """
        score = 0.0

        # 1. Volume ratio
        orb_volumes = [float(c.get("volume", 0)) for c in orb_candles]
        orb_avg_vol = float(np.mean(orb_volumes)) if orb_volumes else 0.0
        bt_vol      = float(breakout_candle.get("volume", 0))
        if orb_avg_vol > 0:
            ratio = bt_vol / orb_avg_vol
            if ratio >= 2.0:
                vol_score = 100.0
            elif ratio >= 1.5:
                vol_score = 80.0 + (ratio - 1.5) * 40.0
            elif ratio >= 1.0:
                vol_score = 50.0 + (ratio - 1.0) * 60.0
            else:
                vol_score = max(0.0, ratio * 50.0)
        else:
            vol_score = 50.0
        score += min(100.0, vol_score) * 0.25

        # 2. Momentum distance (close beyond ORB boundary)
        close = float(breakout_candle["close"])
        if direction == BreakoutDirection.BUY:
            dist_pct = (close - orb.high) / orb.high * 100 if orb.high > 0 else 0.0
        else:
            dist_pct = (orb.low - close) / orb.low * 100 if orb.low > 0 else 0.0
        if dist_pct >= 1.0:
            dist_score = 100.0
        elif dist_pct >= 0.5:
            dist_score = 70.0 + (dist_pct - 0.5) * 60.0
        elif dist_pct >= 0.0:
            dist_score = dist_pct * 140.0
        else:
            dist_score = max(0.0, 50.0 + dist_pct * 50.0)
        score += min(100.0, dist_score) * 0.25

        # 3. Range quality (tight but not too tight; 0.3–1.5 % of mid is ideal)
        mid = (orb.high + orb.low) / 2.0 if (orb.high + orb.low) > 0 else 1.0
        range_pct = orb.range_size / mid * 100 if mid > 0 else 0.0
        if 0.3 <= range_pct <= 1.5:
            range_score = 100.0
        elif range_pct < 0.3:
            range_score = max(0.0, range_pct / 0.3 * 100)
        elif range_pct <= 3.0:
            range_score = max(50.0, 100.0 - (range_pct - 1.5) * 33.0)
        else:
            range_score = max(0.0, 50.0 - (range_pct - 3.0) * 10.0)
        score += min(100.0, range_score) * 0.20

        # 4. RSI positioning (from pre-market momentum)
        if momentum:
            rsi = momentum.rsi_14
            if direction == BreakoutDirection.BUY:
                if 55 <= rsi <= 70:
                    rsi_score = 100.0
                elif 50 <= rsi < 55:
                    rsi_score = 70.0 + (rsi - 50) * 6
                elif rsi > 70:
                    rsi_score = max(30.0, 100.0 - (rsi - 70) * 5)
                else:
                    rsi_score = max(0.0, rsi * 1.0)
            else:
                if 30 <= rsi <= 45:
                    rsi_score = 100.0
                elif 45 < rsi <= 50:
                    rsi_score = 100.0 - (rsi - 45) * 10
                elif rsi < 30:
                    rsi_score = max(30.0, 100.0 - (30 - rsi) * 5)
                else:
                    rsi_score = max(0.0, 100.0 - (rsi - 50) * 2)
        else:
            rsi_score = 50.0
        score += min(100.0, rsi_score) * 0.20

        # 5. Time decay (earlier = better; first candle = 100, last = 0)
        max_idx    = max(total_post_orb_candles - 1, 1)
        time_score = max(0.0, 100.0 - (breakout_idx / max_idx * 100))
        score += time_score * 0.10

        return min(100.0, max(0.0, score))

    # ── Phase 5: Advanced exit simulation ────────────────────────────────────

    def _simulate_exit_advanced(
        self,
        direction: BreakoutDirection,
        entry: float,
        target: float,
        stop_loss: float,
        post_breakout_candles: list[dict],
    ) -> _ExitResult:
        """
        Walk candles after entry and apply full position management:
          1. 1 % take-profit → exit 50 % of qty, move SL to breakeven.
          2. Trailing stop   → activates at `trailing_activation_pct` % move;
                              trails `trailing_distance_pct` % behind best price;
                              never moves against the position.
          3. Time exit       → forced close at the first candle at/after 3:10 PM.
          4. Hard target / SL as backstop (SL checked first).

        PnL is calculated on `self.quantity` shares split across partial + final exit.
        """
        partial_qty   = self.quantity // 2
        remaining_qty = self.quantity - partial_qty

        partial_exit_price: Optional[float] = None
        partial_done                        = False
        trailing_active                     = False

        is_buy        = direction == BreakoutDirection.BUY
        best_price    = entry
        current_sl    = stop_loss
        breakeven     = entry

        partial_trigger = entry * (1 + self.partial_exit_trigger_pct / 100) if is_buy \
                     else entry * (1 - self.partial_exit_trigger_pct / 100)
        trail_trigger   = entry * (1 + self.trailing_activation_pct / 100) if is_buy \
                     else entry * (1 - self.trailing_activation_pct / 100)

        for candle in post_breakout_candles:
            high  = float(candle["high"])
            low   = float(candle["low"])
            close = float(candle["close"])
            dt    = candle.get("datetime", "")

            # ── Update best price and trailing stop ──────────────────────────
            if is_buy:
                best_price = max(best_price, high)
                if self.enable_trailing_stop and best_price >= trail_trigger:
                    trailing_active = True
                if trailing_active:
                    trail_sl = round(best_price * (1 - self.trailing_distance_pct / 100), 2)
                    current_sl = max(current_sl, trail_sl)   # only move up for BUY
            else:
                best_price = min(best_price, low)
                if self.enable_trailing_stop and best_price <= trail_trigger:
                    trailing_active = True
                if trailing_active:
                    trail_sl = round(best_price * (1 + self.trailing_distance_pct / 100), 2)
                    current_sl = min(current_sl, trail_sl)   # only move down for SELL

            # ── 1 % partial take-profit (first occurrence) ──────────────────
            if self.enable_partial_exit and not partial_done:
                if (is_buy and high >= partial_trigger) or (not is_buy and low <= partial_trigger):
                    partial_exit_price = partial_trigger
                    partial_done       = True
                    # Move SL to breakeven (never worse than current SL)
                    if is_buy:
                        current_sl = max(current_sl, breakeven)
                    else:
                        current_sl = min(current_sl, breakeven)

            # ── Hard stop-loss (checked before target) ───────────────────────
            if is_buy and low <= current_sl:
                final_price  = current_sl
                final_reason = "stop_loss"
                return self._build_exit_result(
                    direction, entry, final_price, final_reason,
                    partial_exit_price, partial_qty, remaining_qty,
                )

            if not is_buy and high >= current_sl:
                final_price  = current_sl
                final_reason = "stop_loss"
                return self._build_exit_result(
                    direction, entry, final_price, final_reason,
                    partial_exit_price, partial_qty, remaining_qty,
                )

            # ── Hard target ──────────────────────────────────────────────────
            if is_buy and high >= target:
                return self._build_exit_result(
                    direction, entry, target, "target",
                    partial_exit_price, partial_qty, remaining_qty,
                )
            if not is_buy and low <= target:
                return self._build_exit_result(
                    direction, entry, target, "target",
                    partial_exit_price, partial_qty, remaining_qty,
                )

            # ── Time exit ────────────────────────────────────────────────────
            if dt and self._is_after_time_exit(dt):
                return self._build_exit_result(
                    direction, entry, close, "time_exit",
                    partial_exit_price, partial_qty, remaining_qty,
                )

        # Fallback: market close of last candle
        last_close = float(post_breakout_candles[-1]["close"])
        return self._build_exit_result(
            direction, entry, last_close, "close",
            partial_exit_price, partial_qty, remaining_qty,
        )

    def _build_exit_result(
        self,
        direction: BreakoutDirection,
        entry: float,
        final_price: float,
        reason: str,
        partial_exit_price: Optional[float],
        partial_qty: int,
        remaining_qty: int,
    ) -> _ExitResult:
        is_buy = direction == BreakoutDirection.BUY
        pnl    = 0.0

        if partial_exit_price is not None:
            p_pnl = (partial_exit_price - entry) * partial_qty if is_buy \
                    else (entry - partial_exit_price) * partial_qty
            r_pnl = (final_price - entry) * remaining_qty if is_buy \
                    else (entry - final_price) * remaining_qty
            pnl   = round(p_pnl + r_pnl, 2)
        else:
            full_qty = partial_qty + remaining_qty
            pnl      = round(((final_price - entry) if is_buy else (entry - final_price)) * full_qty, 2)

        return _ExitResult(
            exit_price        = final_price,
            exit_reason       = reason,
            pnl               = pnl,
            partial_exit_price= partial_exit_price,
            partial_exit_qty  = partial_qty if partial_exit_price is not None else 0,
        )

    # ── Pre-market analysis ───────────────────────────────────────────────────

    def _compute_pre_market_analysis(
        self, as_of_date: datetime
    ) -> tuple[Optional[MomentumScore], Optional[TrendAnalysis]]:
        """
        Run momentum scoring and historical trend analysis once before the
        backtest loop.  *as_of_date* should be the day before the test starts
        so no future data leaks into the filters.
        """
        momentum = None
        trend    = None

        if self.momentum_service:
            print(f"  Computing momentum score (lookback {self.momentum_lookback_days}d)…")
            momentum = self.momentum_service.calculate_momentum_score(
                stock_code    = self.stock_code,
                exchange_code = self.exchange_code,
                as_of_date    = as_of_date,
                lookback_days = self.momentum_lookback_days,
            )
            tag = (
                "[STRONG]"  if momentum.is_strong_momentum else
                "[BULLISH]" if momentum.is_bullish          else
                "[WEAK]"
            )
            print(
                f"  Momentum score : {momentum.composite_score:.1f}/100 {tag}  "
                f"ROC5d:{momentum.roc_5d:+.1f}%  RSI:{momentum.rsi_14:.0f}  "
                f"VolRatio:{momentum.volume_ratio_5d:.2f}"
            )

        if self.trend_service and self.trend_filter:
            print(f"  Computing historical trend (lookback {self.trend_lookback_days}d)…")
            trend = self.trend_service.analyze_trend(
                stock_code    = self.stock_code,
                exchange_code = self.exchange_code,
                as_of_date    = as_of_date,
                lookback_days = self.trend_lookback_days,
            )
            print(
                f"  Historical trend: {trend.trend.value}  "
                f"strength={trend.strength:.1f}  slope={trend.price_slope:+.2f}%  "
                f"EMA={trend.ema_signal.value}  ADX={trend.adx:.1f}"
            )

        return momentum, trend

    # ── Backtest entry point ──────────────────────────────────────────────────

    def run_backtest(self) -> list[ORBTradeResult]:
        all_candles = self.orb_data_service.get_intraday_candles(
            self.stock_code, self.exchange_code,
            self.start_date, self.end_date, self.interval,
        )

        days         = ORBDataService.group_by_date(all_candles)
        sorted_dates = sorted(days.keys())

        first_trade_date = sorted_dates[0]
        as_of_date = datetime(
            first_trade_date.year,
            first_trade_date.month,
            first_trade_date.day,
        )

        filters_active = bool(
            self.momentum_service or (self.trend_service and self.trend_filter)
        )

        print(f"\n{'='*80}")
        print(f"  ORB BACKTEST: {self.stock_code} | {self.start_date} → {self.end_date}")
        print(
            f"  ORB period: {self.orb_minutes} min | "
            f"SL: {self.stop_loss_pct}% | "
            f"RR: 1:{self.risk_reward_ratio} | "
            f"Qty: {self.quantity}"
        )
        print(
            f"  Crossover skip: ON (2nd crossover required, ≥30 s gap) | "
            f"Min quality: {self.min_breakout_quality:.0f}"
        )
        print(
            f"  Partial exit: {'ON' if self.enable_partial_exit else 'OFF'} "
            f"({self.partial_exit_trigger_pct}% trigger) | "
            f"Trailing stop: {'ON' if self.enable_trailing_stop else 'OFF'} "
            f"(activate {self.trailing_activation_pct}%, trail {self.trailing_distance_pct}%) | "
            f"Time exit: {self.time_exit_hour:02d}:{self.time_exit_minute:02d}"
        )
        if filters_active:
            print(
                f"  Momentum filter: {'ON' if self.momentum_service else 'OFF'} "
                f"(min={self.min_momentum_score:.0f}) | "
                f"Trend filter: {'ON' if self.trend_service and self.trend_filter else 'OFF'} "
                f"({self.trend_filter_mode})"
            )
        print(f"{'='*80}\n")

        momentum, hist_trend = self._compute_pre_market_analysis(as_of_date)

        if filters_active:
            print()

        # ── Momentum gate ────────────────────────────────────────────────────
        if momentum and momentum.composite_score < self.min_momentum_score:
            print(
                f"  Momentum score {momentum.composite_score:.1f} < "
                f"{self.min_momentum_score:.0f} threshold — skipping entire backtest.\n"
            )
            print(f"{'='*80}")
            print(f"  Trades executed : 0  (momentum filter blocked all trades)")
            print(f"{'='*80}\n")
            return []

        results: list[ORBTradeResult] = []
        total_pnl      = 0.0
        skipped_trend  = 0
        skipped_quality = 0

        for trade_date in sorted_dates:
            day_candles = days[trade_date]

            if len(day_candles) <= self.orb_minutes:
                print(f"  {trade_date}  → Skipped (insufficient candles: {len(day_candles)})")
                continue

            orb_candles      = ORBDataService.get_orb_candles(day_candles, self.orb_minutes)
            post_orb_candles = ORBDataService.get_post_orb_candles(day_candles, self.orb_minutes)
            orb              = self._build_open_range(orb_candles)

            # ── Phase 2: Crossover skip state machine ────────────────────────
            breakout_idx, direction, crossover_skip = self._detect_breakout_with_skip(
                post_orb_candles, orb
            )

            if direction == BreakoutDirection.NONE or breakout_idx is None:
                print(f"  {trade_date}  ORB [{orb.low:.2f}–{orb.high:.2f}]  → No valid 2nd crossover")
                continue

            # ── Intraday trend (per day, uses ORB candles as proxy) ──────────
            intraday_trend = None
            if self.trend_service and self.trend_filter:
                intraday_trend = self.trend_service.analyze_intraday_trend(
                    self.stock_code, orb_candles
                )

            # ── Trend alignment gate ─────────────────────────────────────────
            if hist_trend and self.trend_filter:
                is_buy = direction == BreakoutDirection.BUY
                aligned, reason = self.trend_service.is_signal_aligned(
                    breakout_is_buy    = is_buy,
                    stock_trend        = hist_trend,
                    intraday           = intraday_trend,
                    filter_mode        = self.trend_filter_mode,
                    historical_weight  = self.historical_weight,
                    intraday_weight    = self.intraday_weight,
                )
                if not aligned:
                    skipped_trend += 1
                    dir_label = "BUY " if is_buy else "SELL"
                    print(
                        f"  {trade_date}  ORB [{orb.low:.2f}–{orb.high:.2f}]  "
                        f"{dir_label} → SKIPPED trend ({reason})"
                    )
                    continue

            # ── Phase 3: Breakout quality score ─────────────────────────────
            breakout_candle   = post_orb_candles[breakout_idx]
            quality_score     = self._compute_breakout_quality(
                direction        = direction,
                orb              = orb,
                breakout_candle  = breakout_candle,
                breakout_idx     = breakout_idx,
                total_post_orb_candles = len(post_orb_candles),
                orb_candles      = orb_candles,
                momentum         = momentum,
            )

            if quality_score < self.min_breakout_quality:
                skipped_quality += 1
                dir_label = "BUY " if direction == BreakoutDirection.BUY else "SELL"
                print(
                    f"  {trade_date}  ORB [{orb.low:.2f}–{orb.high:.2f}]  "
                    f"{dir_label} → SKIPPED quality ({quality_score:.1f} < {self.min_breakout_quality:.0f})"
                )
                continue

            # ── Phase 4: Entry ───────────────────────────────────────────────
            breakout_time = breakout_candle["datetime"]
            entry         = orb.high if direction == BreakoutDirection.BUY else orb.low
            target, stop_loss = self._compute_levels(direction, entry, orb)

            # ── Phase 5: Advanced exit simulation ───────────────────────────
            remaining_candles = post_orb_candles[breakout_idx + 1:]
            if not remaining_candles:
                er = _ExitResult(
                    exit_price         = float(breakout_candle["close"]),
                    exit_reason        = "close",
                    pnl                = 0.0,
                    partial_exit_price = None,
                    partial_exit_qty   = 0,
                )
                # recalculate pnl properly
                er = self._build_exit_result(
                    direction, entry, float(breakout_candle["close"]), "close",
                    None, self.quantity // 2, self.quantity - self.quantity // 2,
                )
            else:
                er = self._simulate_exit_advanced(
                    direction, entry, target, stop_loss, remaining_candles
                )

            total_pnl += er.pnl

            result = ORBTradeResult(
                trade_date      = trade_date,
                direction       = direction,
                orb_high        = orb.high,
                orb_low         = orb.low,
                entry_price     = entry,
                target          = target,
                stop_loss       = stop_loss,
                exit_price      = er.exit_price,
                exit_reason     = er.exit_reason,
                pnl             = er.pnl,
                breakout_time   = breakout_time,
                momentum_score  = momentum.composite_score if momentum else None,
                trend_direction = hist_trend.trend.value   if hist_trend else None,
                trend_strength  = hist_trend.strength      if hist_trend else None,
                intraday_trend  = intraday_trend.trend.value if intraday_trend else None,
                crossover_skip_applied  = crossover_skip,
                breakout_quality_score  = quality_score,
                partial_exit_price      = er.partial_exit_price,
                partial_exit_qty        = er.partial_exit_qty,
            )
            results.append(result)

            dir_label    = "BUY " if direction == BreakoutDirection.BUY else "SELL"
            pnl_sign     = "+" if er.pnl >= 0 else ""
            bt_time      = datetime.fromisoformat(breakout_time).strftime("%H:%M")
            momentum_tag = f"  Mom:{momentum.composite_score:.0f}" if momentum else ""
            intraday_tag = f"  ITrend:{intraday_trend.trend.value[:2]}" if intraday_trend else ""
            partial_tag  = (
                f"  Partial@{er.partial_exit_price:.2f}(x{er.partial_exit_qty})"
                if er.partial_exit_price else ""
            )
            print(
                f"  {trade_date}  ORB [{orb.low:.2f}–{orb.high:.2f}]  "
                f"{dir_label} @{bt_time}"
                f"  Entry {entry:.2f}  T {target:.2f}  SL {stop_loss:.2f}"
                f"  Exit {er.exit_price:.2f} [{er.exit_reason:10s}]"
                f"  PnL {pnl_sign}{er.pnl:.2f}"
                f"  Q:{quality_score:.0f}"
                f"{partial_tag}{momentum_tag}{intraday_tag}"
            )

        print(f"\n{'='*80}")
        print(f"  Trades executed : {len(results)}")
        if skipped_trend:
            print(f"  Trend-filtered  : {skipped_trend}")
        if skipped_quality:
            print(f"  Quality-filtered: {skipped_quality}")
        pnl_sign = "+" if total_pnl >= 0 else ""
        print(f"  Total PnL       : {pnl_sign}{total_pnl:.2f}")
        print(f"{'='*80}\n")

        return results


# ── Portfolio pre-market trend analysis ──────────────────────────────────────


def run_premarket_trend_analysis(
    trend_svc: TrendDirectionService,
    top_stocks: list[tuple[str, str, MomentumScore]],
    as_of_date: datetime,
    trend_lookback_days: int = 10,
    analyze_nifty: bool = True,
    nifty_stock_code: str = "NIFTY",
    nifty_exchange: str = "NSE",
) -> dict[str, TrendAnalysis]:
    """
    Mirror FyersORB pre-market trend direction analysis:
      • Optionally analyze Nifty 50 as market-wide context.
      • Analyze each selected stock's historical trend.
      • Log summary counts and per-stock details.

    Returns stock_code → TrendAnalysis mapping used to gate orders.
    """
    logger.info("=" * 60)
    logger.info("RUNNING PRE-MARKET TREND DIRECTION ANALYSIS")
    logger.info("=" * 60)

    # ── Nifty 50 ─────────────────────────────────────────────────────────────
    nifty_trend: Optional[TrendAnalysis] = None
    if analyze_nifty:
        try:
            nifty_trend = trend_svc.analyze_trend(
                stock_code    = nifty_stock_code,
                exchange_code = nifty_exchange,
                as_of_date    = as_of_date,
                lookback_days = trend_lookback_days,
            )
        except Exception as exc:
            logger.warning(f"Nifty 50 trend analysis failed: {exc}")

    # ── Per-stock historical trend ────────────────────────────────────────────
    hist_trends: dict[str, TrendAnalysis] = {}
    for stock_code, exchange_code, _ in top_stocks:
        try:
            trend = trend_svc.analyze_trend(
                stock_code    = stock_code,
                exchange_code = exchange_code,
                as_of_date    = as_of_date,
                lookback_days = trend_lookback_days,
            )
            hist_trends[stock_code] = trend
        except Exception as exc:
            logger.warning(f"Trend analysis failed for {stock_code}: {exc}")

    # ── Summary counts ────────────────────────────────────────────────────────
    up_count   = sum(1 for t in hist_trends.values() if t.trend == TrendDirection.UPTREND)
    down_count = sum(1 for t in hist_trends.values() if t.trend == TrendDirection.DOWNTREND)
    side_count = sum(1 for t in hist_trends.values() if t.trend == TrendDirection.SIDEWAYS)

    logger.info(
        f"Trend analysis complete: {up_count} UPTREND | {down_count} DOWNTREND | {side_count} SIDEWAYS"
    )

    # ── Nifty 50 detail line ──────────────────────────────────────────────────
    if nifty_trend:
        logger.info(
            f"Nifty 50: {nifty_trend.trend.value} "
            f"(strength={nifty_trend.strength:.1f}, slope={nifty_trend.price_slope:+.2f}%, "
            f"EMA={nifty_trend.ema_signal.value})"
        )

    # ── Per-stock detail lines ────────────────────────────────────────────────
    for stock_code, _, ms in top_stocks:
        trend = hist_trends.get(stock_code)
        if trend:
            logger.info(
                f"  {stock_code}: {trend.trend.value}  "
                f"strength={trend.strength:.1f}  slope={trend.price_slope:+.2f}%"
            )

    return hist_trends
