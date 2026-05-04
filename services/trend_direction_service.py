"""
Trend Direction Service for ORB Backtest Strategy.

Mirrors the logic from FyersORB/services/trend_direction_service.py, adapted to
use the Breeze API for historical daily data and intraday candles from the backtest
data already in memory (no extra API call needed for intraday trend).

Historical Trend Analysis (daily candles, computed once before backtest):
  1. EMA Alignment   — EMA9 vs EMA21 vs EMA50          (35%)
  2. Swing Structure — Higher-highs / lower-lows        (30%)
  3. Price Slope     — % change over lookback period    (20%)
  4. ADX / DI        — Average Directional Index        (15%)

Intraday Trend Analysis (minute candles, evaluated per trading day):
  1. EMA Alignment  — EMA5 vs EMA13 on intraday bars   (33%)
  2. VWAP Signal    — price vs session VWAP             (33%)
  3. Session Slope  — % change from open to last price  (33%)

Combined direction: 60% historical + 40% intraday (mirrors FyersORB defaults).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    UPTREND   = "UPTREND"
    DOWNTREND = "DOWNTREND"
    SIDEWAYS  = "SIDEWAYS"


@dataclass
class TrendAnalysis:
    """Multi-day (historical) trend result for a single symbol."""
    symbol: str
    trend: TrendDirection
    strength: float = 0.0       # 0–100

    ema_signal:   TrendDirection = TrendDirection.SIDEWAYS
    swing_signal: TrendDirection = TrendDirection.SIDEWAYS
    slope_signal: TrendDirection = TrendDirection.SIDEWAYS
    adx_signal:   TrendDirection = TrendDirection.SIDEWAYS

    ema9: float = 0.0
    ema21: float = 0.0
    ema50: float = 0.0
    price_slope: float = 0.0
    higher_highs: bool = False
    lower_lows: bool = False
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0

    last_close: float = 0.0
    analyzed_at: datetime = field(default_factory=datetime.now)
    data_quality: str = "UNKNOWN"

    @property
    def is_uptrend(self) -> bool:
        return self.trend == TrendDirection.UPTREND

    @property
    def is_downtrend(self) -> bool:
        return self.trend == TrendDirection.DOWNTREND

    @property
    def is_sideways(self) -> bool:
        return self.trend == TrendDirection.SIDEWAYS

    @property
    def is_strong_trend(self) -> bool:
        return self.strength >= 60.0


@dataclass
class IntradayTrendAnalysis:
    """Intraday trend derived from current-session minute candles."""
    symbol: str
    trend: TrendDirection
    strength: float = 0.0

    ema_signal:   TrendDirection = TrendDirection.SIDEWAYS
    vwap_signal:  TrendDirection = TrendDirection.SIDEWAYS
    slope_signal: TrendDirection = TrendDirection.SIDEWAYS

    ema5: float = 0.0
    ema13: float = 0.0
    vwap: float = 0.0
    session_slope: float = 0.0
    candles_count: int = 0
    analyzed_at: datetime = field(default_factory=datetime.now)
    data_quality: str = "UNKNOWN"

    @property
    def is_uptrend(self) -> bool:
        return self.trend == TrendDirection.UPTREND

    @property
    def is_downtrend(self) -> bool:
        return self.trend == TrendDirection.DOWNTREND


class TrendDirectionService:
    """
    Determines multi-day trend direction from Breeze API historical data and
    intraday trend from the current day's minute candles (already fetched
    by ORBDataService, so no extra API call is needed).
    """

    def __init__(self, breeze):
        self.breeze = breeze

    # ── Data fetching ─────────────────────────────────────────────────────────

    def _fetch_daily_candles(
        self,
        stock_code: str,
        exchange_code: str,
        lookback_days: int = 70,
        as_of_date: Optional[datetime] = None,
    ) -> list[dict]:
        end_dt   = as_of_date or datetime.now()
        start_dt = end_dt - timedelta(days=lookback_days + 20)
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
            logger.debug(f"Fetched {len(candles)} daily candles for trend: {stock_code}")
            return candles
        except Exception as e:
            logger.error(f"Error fetching daily candles for trend ({stock_code}): {e}")
            return []

    # ── Indicator calculations (identical to FyersORB) ────────────────────────

    def _calculate_ema(self, closes: np.ndarray, period: int) -> float:
        if len(closes) < 2:
            return float(closes[-1]) if len(closes) > 0 else 0.0
        period = min(period, len(closes))
        multiplier = 2.0 / (period + 1)
        ema = float(np.mean(closes[:period]))
        for price in closes[period:]:
            ema = price * multiplier + ema * (1 - multiplier)
        return ema

    def _calculate_adx(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> tuple[float, float, float]:
        if len(closes) < period + 2:
            return 25.0, 50.0, 50.0

        tr_list, plus_dm, minus_dm = [], [], []
        for i in range(1, len(closes)):
            hl  = highs[i]  - lows[i]
            hpc = abs(highs[i]  - closes[i - 1])
            lpc = abs(lows[i]   - closes[i - 1])
            tr_list.append(max(hl, hpc, lpc))

            up   = highs[i] - highs[i - 1]
            down = lows[i - 1] - lows[i]
            plus_dm.append(up   if up   > down and up   > 0 else 0.0)
            minus_dm.append(down if down > up   and down > 0 else 0.0)

        tr_sum = float(np.sum(tr_list[-period:]))
        if tr_sum == 0:
            return 0.0, 50.0, 50.0

        plus_di  = float(np.sum(plus_dm[-period:]))  / tr_sum * 100
        minus_di = float(np.sum(minus_dm[-period:])) / tr_sum * 100
        di_sum   = plus_di + minus_di
        adx = (abs(plus_di - minus_di) / di_sum * 100) if di_sum > 0 else 0.0

        return adx, plus_di, minus_di

    def _analyze_swing_structure(
        self, highs: np.ndarray, lows: np.ndarray, lookback: int = 5
    ) -> tuple[bool, bool]:
        n = min(lookback, len(highs))
        if n < 2:
            return False, False
        rh, rl = highs[-n:], lows[-n:]
        threshold = (n - 1) * 0.6
        hh = sum(1 for i in range(1, n) if rh[i] > rh[i - 1]) >= threshold
        ll = sum(1 for i in range(1, n) if rl[i] < rl[i - 1]) >= threshold
        return hh, ll

    def _calculate_price_slope(self, closes: np.ndarray, lookback: int) -> float:
        lookback = min(lookback, len(closes) - 1)
        if lookback <= 0:
            return 0.0
        start = closes[-(lookback + 1)]
        end   = closes[-1]
        return ((end - start) / start) * 100 if start > 0 else 0.0

    def _calculate_vwap(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> float:
        total_vol = float(np.sum(volumes))
        if total_vol <= 0:
            return float(closes[-1])
        typical = (highs + lows + closes) / 3.0
        return float(np.sum(typical * volumes) / total_vol)

    # ── Core analysis ─────────────────────────────────────────────────────────

    def analyze_trend(
        self,
        stock_code: str,
        exchange_code: str,
        as_of_date: Optional[datetime] = None,
        lookback_days: int = 10,
    ) -> TrendAnalysis:
        """
        Multi-day trend analysis for a single symbol.

        Pass *as_of_date* equal to the day before the backtest start so only
        pre-test daily data is used, preserving look-ahead-free evaluation.
        """
        symbol = stock_code
        try:
            candles = self._fetch_daily_candles(
                stock_code, exchange_code,
                lookback_days=max(lookback_days * 3, 70),
                as_of_date=as_of_date,
            )
            if not candles or len(candles) < 5:
                logger.warning(f"Insufficient daily data for trend analysis: {symbol}")
                return TrendAnalysis(
                    symbol=symbol, trend=TrendDirection.SIDEWAYS,
                    strength=0.0, data_quality="INSUFFICIENT"
                )

            closes = np.array([float(c["close"]) for c in candles])
            highs  = np.array([float(c["high"])  for c in candles])
            lows   = np.array([float(c["low"])   for c in candles])

            result = TrendAnalysis(
                symbol=symbol, trend=TrendDirection.SIDEWAYS, strength=0.0
            )
            result.last_close   = float(closes[-1])
            result.data_quality = "GOOD" if len(candles) >= 30 else "PARTIAL"

            # 1. EMA Alignment
            result.ema9  = self._calculate_ema(closes, 9)
            result.ema21 = self._calculate_ema(closes, 21)
            result.ema50 = self._calculate_ema(closes, min(50, len(closes)))

            price = float(closes[-1])
            ema_bullish = price > result.ema9  and result.ema9  > result.ema21
            ema_bearish = price < result.ema9  and result.ema9  < result.ema21
            result.ema_signal = (
                TrendDirection.UPTREND   if ema_bullish else
                TrendDirection.DOWNTREND if ema_bearish else
                TrendDirection.SIDEWAYS
            )

            # 2. Swing Structure
            n = min(lookback_days, len(highs))
            result.higher_highs, result.lower_lows = self._analyze_swing_structure(
                highs[-n:], lows[-n:], lookback=n
            )
            if result.higher_highs and not result.lower_lows:
                result.swing_signal = TrendDirection.UPTREND
            elif result.lower_lows and not result.higher_highs:
                result.swing_signal = TrendDirection.DOWNTREND
            elif result.higher_highs and result.lower_lows:
                result.swing_signal = result.ema_signal   # expansion: follow EMA
            else:
                result.swing_signal = TrendDirection.SIDEWAYS

            # 3. Price Slope
            result.price_slope = self._calculate_price_slope(closes, lookback_days)
            result.slope_signal = (
                TrendDirection.UPTREND   if result.price_slope >  1.5 else
                TrendDirection.DOWNTREND if result.price_slope < -1.5 else
                TrendDirection.SIDEWAYS
            )

            # 4. ADX / DI
            result.adx, result.plus_di, result.minus_di = self._calculate_adx(
                highs, lows, closes, period=14
            )
            if result.plus_di > result.minus_di and result.adx > 20:
                result.adx_signal = TrendDirection.UPTREND
            elif result.minus_di > result.plus_di and result.adx > 20:
                result.adx_signal = TrendDirection.DOWNTREND
            else:
                result.adx_signal = TrendDirection.SIDEWAYS

            # 5. Weighted voting — EMA 35%, Swing 30%, Slope 20%, ADX 15%
            signal_weights = [
                (result.ema_signal,   0.35),
                (result.swing_signal, 0.30),
                (result.slope_signal, 0.20),
                (result.adx_signal,   0.15),
            ]
            up_score   = sum(w for sig, w in signal_weights if sig == TrendDirection.UPTREND)
            down_score = sum(w for sig, w in signal_weights if sig == TrendDirection.DOWNTREND)

            if up_score >= 0.55:
                result.trend    = TrendDirection.UPTREND
                result.strength = min(100.0, up_score * 100 + result.adx * 0.3)
            elif down_score >= 0.55:
                result.trend    = TrendDirection.DOWNTREND
                result.strength = min(100.0, down_score * 100 + result.adx * 0.3)
            else:
                result.trend    = TrendDirection.SIDEWAYS
                result.strength = min(100.0, max(up_score, down_score) * 100)

            result.analyzed_at = datetime.now()

            logger.info(
                f"Trend [{symbol}]: {result.trend.value}  "
                f"strength={result.strength:.1f}  slope={result.price_slope:+.2f}%  "
                f"EMA={result.ema_signal.value}  Swing={result.swing_signal.value}  "
                f"ADX={result.adx:.1f}(+DI={result.plus_di:.1f} -DI={result.minus_di:.1f})"
            )
            return result

        except Exception as e:
            logger.error(f"Error analyzing trend for {symbol}: {e}")
            return TrendAnalysis(
                symbol=symbol, trend=TrendDirection.SIDEWAYS,
                strength=0.0, data_quality="ERROR"
            )

    def analyze_intraday_trend(
        self, symbol: str, day_candles: list[dict]
    ) -> IntradayTrendAnalysis:
        """
        Derive intraday trend from the current session's minute candles.

        Called during the backtest with the candles already fetched for that day,
        so no additional API call is required.  Mirrors FyersORB intraday analysis:
          1. EMA5 vs EMA13        (33%)
          2. Price vs VWAP        (33%)
          3. Session slope        (33%)
        """
        try:
            if not day_candles or len(day_candles) < 3:
                return IntradayTrendAnalysis(
                    symbol=symbol, trend=TrendDirection.SIDEWAYS,
                    strength=0.0, data_quality="INSUFFICIENT"
                )

            opens   = np.array([float(c["open"])   for c in day_candles])
            highs   = np.array([float(c["high"])   for c in day_candles])
            lows    = np.array([float(c["low"])    for c in day_candles])
            closes  = np.array([float(c["close"])  for c in day_candles])
            volumes = np.array([float(c["volume"]) for c in day_candles])

            result = IntradayTrendAnalysis(
                symbol=symbol,
                trend=TrendDirection.SIDEWAYS,
                strength=0.0,
                candles_count=len(day_candles),
                data_quality="GOOD" if len(day_candles) >= 6 else "PARTIAL",
            )

            last_price = float(closes[-1])

            # 1. EMA Alignment
            result.ema5  = self._calculate_ema(closes, min(5,  len(closes)))
            result.ema13 = self._calculate_ema(closes, min(13, len(closes)))
            if last_price > result.ema5 and result.ema5 > result.ema13:
                result.ema_signal = TrendDirection.UPTREND
            elif last_price < result.ema5 and result.ema5 < result.ema13:
                result.ema_signal = TrendDirection.DOWNTREND
            else:
                result.ema_signal = TrendDirection.SIDEWAYS

            # 2. VWAP Signal
            result.vwap = self._calculate_vwap(highs, lows, closes, volumes)
            if last_price > result.vwap * 1.001:
                result.vwap_signal = TrendDirection.UPTREND
            elif last_price < result.vwap * 0.999:
                result.vwap_signal = TrendDirection.DOWNTREND
            else:
                result.vwap_signal = TrendDirection.SIDEWAYS

            # 3. Session Slope
            session_open = float(opens[0])
            result.session_slope = (
                (last_price - session_open) / session_open * 100
                if session_open > 0 else 0.0
            )
            if result.session_slope > 0.5:
                result.slope_signal = TrendDirection.UPTREND
            elif result.session_slope < -0.5:
                result.slope_signal = TrendDirection.DOWNTREND
            else:
                result.slope_signal = TrendDirection.SIDEWAYS

            # Composite: equal 1/3 weights
            signals    = [result.ema_signal, result.vwap_signal, result.slope_signal]
            up_score   = sum(1 for s in signals if s == TrendDirection.UPTREND)   / 3.0
            down_score = sum(1 for s in signals if s == TrendDirection.DOWNTREND) / 3.0

            if up_score >= 0.55:
                result.trend    = TrendDirection.UPTREND
                result.strength = min(100.0, up_score * 100)
            elif down_score >= 0.55:
                result.trend    = TrendDirection.DOWNTREND
                result.strength = min(100.0, down_score * 100)
            else:
                result.trend    = TrendDirection.SIDEWAYS
                result.strength = min(100.0, max(up_score, down_score) * 100)

            result.analyzed_at = datetime.now()
            return result

        except Exception as e:
            logger.error(f"Error analyzing intraday trend for {symbol}: {e}")
            return IntradayTrendAnalysis(
                symbol=symbol, trend=TrendDirection.SIDEWAYS,
                strength=0.0, data_quality="ERROR"
            )

    # ── Combined direction ────────────────────────────────────────────────────

    def get_combined_trend_direction(
        self,
        historical: TrendAnalysis,
        intraday: Optional[IntradayTrendAnalysis],
        historical_weight: float = 0.6,
        intraday_weight: float   = 0.4,
    ) -> tuple[TrendDirection, float]:
        """
        Blend historical and intraday trend signals (mirrors FyersORB logic).

        Returns (combined_direction, abs_score).
        """
        def _to_score(trend: TrendDirection, strength: float) -> float:
            if trend == TrendDirection.UPTREND:
                return  strength / 100.0
            if trend == TrendDirection.DOWNTREND:
                return -strength / 100.0
            return 0.0

        hist_score = _to_score(historical.trend, historical.strength)

        if intraday is not None and intraday.data_quality not in ("INSUFFICIENT", "ERROR"):
            intra_score = _to_score(intraday.trend, intraday.strength)
            combined    = historical_weight * hist_score + intraday_weight * intra_score
        else:
            combined = hist_score

        if combined > 0.20:
            return TrendDirection.UPTREND,   abs(combined)
        if combined < -0.20:
            return TrendDirection.DOWNTREND, abs(combined)
        return TrendDirection.SIDEWAYS, abs(combined)

    # ── Signal alignment check (mirrors FyersORB is_signal_aligned_with_trend) ─

    def is_signal_aligned(
        self,
        breakout_is_buy: bool,
        stock_trend: TrendAnalysis,
        intraday: Optional[IntradayTrendAnalysis] = None,
        filter_mode: str = "STRICT",
        historical_weight: float = 0.6,
        intraday_weight: float   = 0.4,
    ) -> tuple[bool, str]:
        """
        Decide whether a breakout direction is aligned with the prevailing trend.

        filter_mode:
          STRICT  — reject if combined stock trend opposes the signal.
          LENIENT — reject only if the combined trend clearly opposes (score > 0.20).

        Returns (is_aligned, reason_string).
        """
        combined_dir, combined_score = self.get_combined_trend_direction(
            stock_trend, intraday, historical_weight, intraday_weight
        )

        if breakout_is_buy:
            if combined_dir == TrendDirection.DOWNTREND:
                return False, (
                    f"BUY rejected: combined trend=DOWNTREND "
                    f"(hist={stock_trend.trend.value}, slope={stock_trend.price_slope:+.2f}%)"
                )
            return True, (
                f"BUY aligned: combined={combined_dir.value} "
                f"score={combined_score:.2f}"
            )
        else:
            if combined_dir == TrendDirection.UPTREND:
                return False, (
                    f"SELL rejected: combined trend=UPTREND "
                    f"(hist={stock_trend.trend.value}, slope={stock_trend.price_slope:+.2f}%)"
                )
            return True, (
                f"SELL aligned: combined={combined_dir.value} "
                f"score={combined_score:.2f}"
            )
