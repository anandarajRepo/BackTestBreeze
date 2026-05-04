"""
Open Range Breakout (ORB) Strategy — backtest implementation for Breeze API.

Logic (mirrors FyersORB):
  Phase 0 — Pre-market: compute momentum score and historical trend direction.
             Trades are skipped when momentum is below *min_momentum_score* or
             when the breakout direction opposes the prevailing trend.
  Phase 1 — Build opening range from first `orb_minutes` candles (default 15).
  Phase 2 — Scan post-ORB candles for the first breakout above ORB high (BUY)
             or below ORB low (SELL).
  Phase 3 — Enter at breakout price; exit via target, stop-loss, or market close.
"""

from datetime import datetime
from typing import Optional

from models.orb_models import BreakoutDirection, OpenRange, ORBTradeResult
from services.orb_data_service import ORBDataService
from services.momentum_service import MomentumScore, MomentumScoringService
from services.trend_direction_service import (
    TrendAnalysis,
    TrendDirection,
    TrendDirectionService,
)


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

        self.momentum_service      = momentum_service
        self.min_momentum_score    = min_momentum_score
        self.momentum_lookback_days = momentum_lookback_days

        self.trend_service        = trend_service
        self.trend_filter         = trend_filter
        self.trend_filter_mode    = trend_filter_mode
        self.trend_lookback_days  = trend_lookback_days
        self.historical_weight    = historical_weight
        self.intraday_weight      = intraday_weight

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

    # ── backtest ──────────────────────────────────────────────────────────────

    def run_backtest(self) -> list[ORBTradeResult]:
        all_candles = self.orb_data_service.get_intraday_candles(
            self.stock_code, self.exchange_code,
            self.start_date, self.end_date, self.interval,
        )

        days         = ORBDataService.group_by_date(all_candles)
        sorted_dates = sorted(days.keys())

        # ── Pre-market analysis (once per backtest run) ──────────────────────
        # Use one day before the first trading day as the cutoff so daily candle
        # data doesn't include any days from the test period.
        first_trade_date = sorted_dates[0]
        as_of_date = datetime(
            first_trade_date.year,
            first_trade_date.month,
            first_trade_date.day,
        )

        filters_active = bool(self.momentum_service or (self.trend_service and self.trend_filter))

        print(f"\n{'='*80}")
        print(f"  ORB BACKTEST: {self.stock_code} | {self.start_date} → {self.end_date}")
        print(
            f"  ORB period: {self.orb_minutes} min | "
            f"SL: {self.stop_loss_pct}% | "
            f"RR: 1:{self.risk_reward_ratio} | "
            f"Qty: {self.quantity}"
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

            # ── Intraday trend check (per day, uses ORB candles as proxy) ───
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
                        f"{dir_label} → SKIPPED ({reason})"
                    )
                    continue

            breakout_candle = post_orb_candles[breakout_idx]
            breakout_time   = breakout_candle["datetime"]
            entry = orb.high if direction == BreakoutDirection.BUY else orb.low

            target, stop_loss = self._compute_levels(direction, entry, orb)

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
                momentum_score = momentum.composite_score if momentum else None,
                trend_direction = hist_trend.trend.value if hist_trend else None,
                trend_strength  = hist_trend.strength    if hist_trend else None,
                intraday_trend  = intraday_trend.trend.value if intraday_trend else None,
            )
            results.append(result)

            dir_label = "BUY " if direction == BreakoutDirection.BUY else "SELL"
            pnl_sign  = "+" if pnl >= 0 else ""
            bt_time   = datetime.fromisoformat(breakout_time).strftime("%H:%M")
            momentum_tag = (
                f"  Mom:{momentum.composite_score:.0f}" if momentum else ""
            )
            intraday_tag = (
                f"  ITrend:{intraday_trend.trend.value[:2]}" if intraday_trend else ""
            )
            print(
                f"  {trade_date}  ORB [{orb.low:.2f}–{orb.high:.2f}]  "
                f"{dir_label} @{bt_time}"
                f"  Entry {entry:.2f}  T {target:.2f}  SL {stop_loss:.2f}"
                f"  Exit {exit_price:.2f} [{exit_reason:10s}]"
                f"  PnL {pnl_sign}{pnl:.2f}"
                f"{momentum_tag}{intraday_tag}"
            )

        print(f"\n{'='*80}")
        print(f"  Trades executed : {len(results)}")
        if skipped_trend:
            print(f"  Trend-filtered  : {skipped_trend}")
        pnl_sign = "+" if total_pnl >= 0 else ""
        print(f"  Total PnL       : {pnl_sign}{total_pnl:.2f}")
        print(f"{'='*80}\n")

        return results
