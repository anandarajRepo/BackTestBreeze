"""
Momentum Scoring Service for ORB Backtest Strategy.

Mirrors the logic from FyersORB/services/momentum_service.py, adapted to use
the Breeze API for historical daily data instead of the Fyers API.

Momentum Score Components (weights):
  1. Price Rate of Change  (ROC 5/10/20-day)  — 30%
  2. RSI positioning (14-day)                 — 15%
  3. Volume trend (5-day vs 20-day avg)       — 20%
  4. Moving average alignment (SMA5>10>20)    — 25%
  5. Consecutive up/down day streak           — 10%
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MomentumScore:
    """Momentum score breakdown for a single stock."""
    symbol: str
    composite_score: float = 0.0        # 0–100 overall score

    # Component scores (each 0–100)
    roc_score: float = 0.0
    rsi_score: float = 0.0
    volume_trend_score: float = 0.0
    ma_alignment_score: float = 0.0
    streak_score: float = 0.0

    # Raw indicator values
    roc_5d: float = 0.0
    roc_10d: float = 0.0
    roc_20d: float = 0.0
    rsi_14: float = 50.0
    volume_ratio_5d: float = 1.0
    consecutive_up_days: int = 0
    consecutive_down_days: int = 0
    price_vs_sma20: float = 0.0

    last_close: float = 0.0
    avg_daily_volume: float = 0.0
    scored_at: datetime = field(default_factory=datetime.now)
    data_quality: str = "UNKNOWN"       # GOOD | PARTIAL | INSUFFICIENT

    @property
    def is_bullish(self) -> bool:
        return self.composite_score >= 60.0

    @property
    def is_strong_momentum(self) -> bool:
        return self.composite_score >= 75.0


class MomentumScoringService:
    """
    Screens a stock for momentum using historical daily candles from Breeze API.
    Produced scores are used by ORBStrategy to decide whether to enter a trade.
    """

    WEIGHT_ROC          = 0.30
    WEIGHT_RSI          = 0.15
    WEIGHT_VOLUME_TREND = 0.20
    WEIGHT_MA_ALIGNMENT = 0.25
    WEIGHT_STREAK       = 0.10

    def __init__(self, breeze):
        self.breeze = breeze

    # ── Data fetching ─────────────────────────────────────────────────────────

    def _fetch_daily_candles(
        self,
        stock_code: str,
        exchange_code: str,
        lookback_days: int = 60,
        as_of_date: Optional[datetime] = None,
    ) -> list[dict]:
        """Fetch daily OHLCV candles from Breeze API."""
        end_dt   = as_of_date or datetime.now()
        start_dt = end_dt - timedelta(days=lookback_days + 20)  # buffer for weekends
        try:
            resp = self.breeze.get_historical_data_v2(
                interval      = "1day",
                from_date     = start_dt,
                to_date       = end_dt,
                stock_code    = stock_code,
                exchange_code = exchange_code,
                product_type  = "cash",
            )
            candles = resp.get("Success") or []
            logger.debug(f"Fetched {len(candles)} daily candles for momentum: {stock_code}")
            return candles
        except Exception as e:
            logger.error(f"Error fetching daily candles for {stock_code}: {e}")
            return []

    # ── Indicator calculations (identical to FyersORB) ────────────────────────

    def _calculate_roc(self, closes: np.ndarray, period: int) -> float:
        if len(closes) <= period:
            return 0.0
        return ((closes[-1] - closes[-(period + 1)]) / closes[-(period + 1)]) * 100

    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def _calculate_sma(self, closes: np.ndarray, period: int) -> float:
        if len(closes) < period:
            return 0.0
        return float(np.mean(closes[-period:]))

    def _calculate_consecutive_days(self, closes: np.ndarray) -> tuple[int, int]:
        if len(closes) < 2:
            return 0, 0
        up_days = down_days = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] > closes[i - 1]:
                if down_days > 0:
                    break
                up_days += 1
            elif closes[i] < closes[i - 1]:
                if up_days > 0:
                    break
                down_days += 1
            else:
                break
        return up_days, down_days

    # ── Scoring functions (identical to FyersORB) ─────────────────────────────

    def _score_roc(self, roc_5d: float, roc_10d: float, roc_20d: float) -> float:
        score = 0.0
        # 5-day ROC — 40% of ROC score
        if roc_5d > 0:
            if 1.0 <= roc_5d <= 8.0:
                score += 40.0
            elif roc_5d > 8.0:
                score += max(20.0, 40.0 - (roc_5d - 8.0) * 2)
            else:
                score += roc_5d * 40.0
        else:
            score += max(0.0, 10.0 + roc_5d * 2)
        # 10-day ROC — 35%
        if roc_10d > 0:
            if 2.0 <= roc_10d <= 15.0:
                score += 35.0
            elif roc_10d > 15.0:
                score += max(15.0, 35.0 - (roc_10d - 15.0) * 1.5)
            else:
                score += roc_10d * 17.5
        else:
            score += max(0.0, 5.0 + roc_10d)
        # 20-day ROC — 25%
        if roc_20d > 0:
            if 3.0 <= roc_20d <= 20.0:
                score += 25.0
            elif roc_20d > 20.0:
                score += max(10.0, 25.0 - (roc_20d - 20.0))
            else:
                score += roc_20d * 8.33
        else:
            score += max(0.0, 5.0 + roc_20d * 0.5)
        return min(max(score, 0.0), 100.0)

    def _score_rsi(self, rsi: float) -> float:
        if 55 <= rsi <= 70:
            return 100.0
        elif 50 <= rsi < 55:
            return 70.0 + (rsi - 50) * 6
        elif 70 < rsi <= 80:
            return 100.0 - (rsi - 70) * 5
        elif 40 <= rsi < 50:
            return 40.0 + (rsi - 40) * 3
        elif rsi > 80:
            return max(0.0, 50.0 - (rsi - 80) * 5)
        elif 30 <= rsi < 40:
            return 20.0 + (rsi - 30) * 2
        return max(0.0, rsi * 0.67)

    def _score_volume_trend(self, recent_avg: float, longer_avg: float) -> float:
        if longer_avg <= 0:
            return 50.0
        ratio = recent_avg / longer_avg
        if ratio >= 1.5:
            return 100.0
        elif ratio >= 1.2:
            return 80.0 + (ratio - 1.2) * 66.7
        elif ratio >= 1.0:
            return 60.0 + (ratio - 1.0) * 100
        elif ratio >= 0.8:
            return 30.0 + (ratio - 0.8) * 150
        return max(0.0, ratio * 37.5)

    def _score_ma_alignment(
        self, price: float, sma5: float, sma10: float, sma20: float
    ) -> float:
        if not sma5 or not sma10 or not sma20:
            return 50.0
        score = 0.0
        if price > sma5:  score += 25.0
        if price > sma20: score += 25.0
        if sma5  > sma10: score += 25.0
        if sma10 > sma20: score += 25.0
        return score

    def _score_streak(self, up_days: int, down_days: int) -> float:
        if 2 <= up_days <= 4:
            return 100.0
        elif up_days == 1:
            return 60.0
        elif up_days == 5:
            return 70.0
        elif up_days > 5:
            return max(30.0, 70.0 - (up_days - 5) * 10)
        elif down_days == 1:
            return 40.0
        elif down_days == 2:
            return 25.0
        return max(0.0, 20.0 - down_days * 5)

    # ── Public API ────────────────────────────────────────────────────────────

    def score_all_symbols(
        self,
        symbols: list[tuple[str, str]],  # [(stock_code, exchange_code), ...]
        as_of_date: Optional[datetime] = None,
        lookback_days: int = 30,
    ) -> list["MomentumScore"]:
        """
        Score every symbol in *symbols* and return list sorted by composite
        score descending.  Failures produce a score of 0 so they naturally
        rank last.
        """
        scores: list[MomentumScore] = []
        total = len(symbols)
        for i, (stock_code, exchange_code) in enumerate(symbols, 1):
            print(f"  [{i}/{total}] Scoring {stock_code}…", end="\r", flush=True)
            score = self.calculate_momentum_score(
                stock_code=stock_code,
                exchange_code=exchange_code,
                as_of_date=as_of_date,
                lookback_days=lookback_days,
            )
            scores.append(score)
        print()  # newline after progress
        scores.sort(key=lambda s: s.composite_score, reverse=True)
        return scores

    def calculate_momentum_score(
        self,
        stock_code: str,
        exchange_code: str,
        as_of_date: Optional[datetime] = None,
        lookback_days: int = 30,
    ) -> MomentumScore:
        """
        Calculate composite momentum score for a single stock.

        Uses daily candles ending on or before *as_of_date* (defaults to now).
        In a backtest, pass the date just before the test period begins so
        only pre-test data is used for scoring.
        """
        symbol = stock_code
        try:
            candles = self._fetch_daily_candles(
                stock_code, exchange_code, lookback_days, as_of_date
            )
            if not candles or len(candles) < 10:
                logger.warning(f"Insufficient daily data for momentum scoring: {symbol}")
                return MomentumScore(symbol=symbol, data_quality="INSUFFICIENT")

            closes  = np.array([float(c["close"])  for c in candles])
            volumes = np.array([float(c["volume"]) for c in candles])

            score = MomentumScore(symbol=symbol)
            score.last_close       = float(closes[-1])
            score.avg_daily_volume = float(np.mean(volumes[-min(20, len(volumes)):]))
            score.data_quality = (
                "GOOD"         if len(candles) >= 25 else
                "PARTIAL"      if len(candles) >= 15 else
                "INSUFFICIENT"
            )

            # 1. Rate of Change
            score.roc_5d  = self._calculate_roc(closes, 5)  if len(closes) > 5  else 0.0
            score.roc_10d = self._calculate_roc(closes, 10) if len(closes) > 10 else 0.0
            score.roc_20d = self._calculate_roc(closes, 20) if len(closes) > 20 else 0.0
            score.roc_score = self._score_roc(score.roc_5d, score.roc_10d, score.roc_20d)

            # 2. RSI
            score.rsi_14    = self._calculate_rsi(closes, 14)
            score.rsi_score = self._score_rsi(score.rsi_14)

            # 3. Volume trend
            vol_5d  = float(np.mean(volumes[-5:]))  if len(volumes) >= 5  else float(np.mean(volumes))
            vol_20d = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
            score.volume_ratio_5d    = vol_5d / vol_20d if vol_20d > 0 else 1.0
            score.volume_trend_score = self._score_volume_trend(vol_5d, vol_20d)

            # 4. Moving average alignment
            sma5  = self._calculate_sma(closes, 5)
            sma10 = self._calculate_sma(closes, 10)
            sma20 = self._calculate_sma(closes, min(20, len(closes)))
            score.ma_alignment_score = self._score_ma_alignment(
                score.last_close, sma5, sma10, sma20
            )
            score.price_vs_sma20 = (
                (score.last_close - sma20) / sma20 * 100 if sma20 > 0 else 0.0
            )

            # 5. Consecutive day streak
            up_days, down_days = self._calculate_consecutive_days(closes)
            score.consecutive_up_days   = up_days
            score.consecutive_down_days = down_days
            score.streak_score = self._score_streak(up_days, down_days)

            # Composite (weighted)
            score.composite_score = (
                score.roc_score          * self.WEIGHT_ROC +
                score.rsi_score          * self.WEIGHT_RSI +
                score.volume_trend_score * self.WEIGHT_VOLUME_TREND +
                score.ma_alignment_score * self.WEIGHT_MA_ALIGNMENT +
                score.streak_score       * self.WEIGHT_STREAK
            )
            score.scored_at = datetime.now()

            logger.info(
                f"Momentum [{symbol}]: {score.composite_score:.1f}/100 "
                f"(ROC:{score.roc_score:.0f} RSI:{score.rsi_score:.0f} "
                f"Vol:{score.volume_trend_score:.0f} MA:{score.ma_alignment_score:.0f} "
                f"Streak:{score.streak_score:.0f}) "
                f"ROC5d:{score.roc_5d:+.1f}% RSI:{score.rsi_14:.0f} "
                f"VolRatio:{score.volume_ratio_5d:.2f}"
            )
            return score

        except Exception as e:
            logger.error(f"Error calculating momentum score for {symbol}: {e}")
            return MomentumScore(symbol=symbol, data_quality="INSUFFICIENT")
