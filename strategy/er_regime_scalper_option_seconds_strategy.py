"""
Xiznit ER Regime Scalper — Nifty weekly options on 1-second data.
==================================================================

This is an options adaptation of the "Xiznit ER Regime Scalper", a
momentum-based scalping strategy originally built for the 2-minute timeframe on
futures (MNQ, MGC, SIL). The trading *signal* is computed on the Nifty 50 spot
series (resampled to the strategy timeframe); the resulting long/short bias is
expressed by buying the ATM option for the expiry week:

  * a bullish (green/Uptrend) regime  -> buy the ATM **CE**  (long)
  * a bearish (red/Downtrend) regime   -> buy the ATM **PE**  (short)

Buying an option is always a long-premium position, so for both CE and PE the
take-profit sits above the entry and the stop-loss below it.

Efficiency Ratio Regime Filter
------------------------------
Kaufman's Efficiency Ratio (ER) over `er_period` classifies every bar into one
of four regimes:

  * **green**  (Uptrend)      : ER >= `er_trend_threshold` and net change up
  * **red**    (Downtrend)    : ER >= `er_trend_threshold` and net change down
  * **orange** (Chop)         : ER below the trend threshold but the recent
                                range is wide (>= `consolidation_range_pct`)
  * **grey**   (Consolidation): ER below the trend threshold and the recent
                                range is tight (< `consolidation_range_pct`)

Entries are only taken on the **first** qualifying candle after the market
transitions into a trending state (green/red) from a non-trending state
(orange/grey). Consecutive same-colour candles never re-trigger — a full regime
reset (back through a non-trending state) is required before a new entry.

Open trades are flattened immediately when the regime shifts away from the trade
direction (a long exits the moment the regime is no longer green; a short the
moment it is no longer red).

Entry Modes (`entry_mode`)
--------------------------
  * ``full_filter`` : price on the correct side of VWAP, fast MA above/below
                      slow MA, AND both MAs on the correct side of VWAP.
  * ``vwap_only``   : price above VWAP for longs / below for shorts.
  * ``ema_only``    : fast MA above slow MA for longs / below for shorts.
  * ``regime_only`` : first qualifying regime candle, no extra filters.
  * ``fresh_cross`` : the MA crossover happened within `fresh_cross_bars` bars.
  * ``pullback``    : price pulled back to the fast MA within `pullback_bars`
                      bars and then printed a confirming regime candle.

Entry Filters (each independently toggleable)
---------------------------------------------
  * ``block_first_minutes``      : block the first N minutes of the session.
  * ``require_prior_alignment``  : the mode alignment must also have held on the
                                   prior bar.
  * ``min_body_points``          : skip doji/indecision signal candles below a
                                   minimum body size (in index points).
  * ``require_ma_slope``         : both MAs must be rising (long) / falling
                                   (short).
  * ``require_prior_bar_break``  : signal candle must close above the prior
                                   bar's high (long) / below its low (short).
  * ``block_lunch``              : block entries inside the lunch window.
  * ``stronger_er``              : require ER >= `er_strong_threshold` instead of
                                   the standard trend threshold.

Trade Management
----------------
  * Take-profit / stop-loss configured in **ticks** (`tp_ticks` / `sl_ticks`,
    converted to option-premium points via `tick_size`).
  * Optional move-to-breakeven once the trade is `breakeven_ticks` in favour.
  * EOD flatten: all open positions are closed at `eod_flatten_time` (IST).
"""

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from math import floor

import pandas as pd

from models.er_regime_models import (
    ERSymbolMetrics,
    ERTradeResult,
    ERWeeklyExpiryResult,
)
from services.nifty_option_service import NiftyOptionService
from strategy.adx_option_strategy import resample_candles


ENTRY_MODES = (
    "full_filter",
    "vwap_only",
    "ema_only",
    "regime_only",
    "fresh_cross",
    "pullback",
)

_GREEN, _RED, _ORANGE, _GREY = "green", "red", "orange", "grey"
_TRENDING = (_GREEN, _RED)


def _capital_allocation_pct(price: float) -> float:
    """Fraction of capital to allocate based on option price (mirrors the other
    *WeeklyOptionsSeconds* strategies)."""
    if price <= 20:
        return 0.30
    return 1.00


def compute_er_regime(
    candles: list[dict],
    er_period: int,
    er_trend_threshold: float,
    consolidation_range_pct: float,
    ema_fast: int,
    ema_slow: int,
) -> pd.DataFrame | None:
    """
    Build the signal DataFrame: OHLCV plus the Efficiency Ratio, the four-state
    regime label, the session VWAP, the dual EMAs and a few precomputed helper
    columns used by the entry filters.
    """
    if not candles:
        return None

    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0

    n = er_period

    # ── Kaufman Efficiency Ratio ────────────────────────────────────────────
    net_change = (df["close"] - df["close"].shift(n)).abs()
    volatility = df["close"].diff().abs().rolling(n).sum()
    df["er"] = (net_change / volatility).fillna(0.0)
    df["er"] = df["er"].replace([float("inf"), float("-inf")], 0.0)
    df["net"] = df["close"] - df["close"].shift(n)

    # ── Regime classification ──────────────────────────────────────────────
    rng = (df["high"].rolling(n).max() - df["low"].rolling(n).min())
    range_pct = (rng / df["close"]).fillna(0.0) * 100.0

    def _classify(row) -> str:
        er = row["er"]
        net = row["net"]
        if pd.isna(net):
            return _GREY
        if er >= er_trend_threshold and net > 0:
            return _GREEN
        if er >= er_trend_threshold and net < 0:
            return _RED
        if row["range_pct"] >= consolidation_range_pct:
            return _ORANGE
        return _GREY

    df["range_pct"] = range_pct
    df["regime"] = df.apply(_classify, axis=1)

    # ── Session VWAP (resets each trading day) ──────────────────────────────
    df["d"] = df["datetime"].dt.date
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_pv = (tp * df["volume"]).groupby(df["d"]).cumsum()
    cum_v = df["volume"].groupby(df["d"]).cumsum()
    vwap = cum_pv / cum_v
    # Index data often carries no volume; fall back to the running average of the
    # typical price within the day so the VWAP filters still have a reference.
    fallback = tp.groupby(df["d"]).expanding().mean().reset_index(level=0, drop=True)
    df["vwap"] = vwap.where(cum_v > 0, fallback)

    # ── Dual EMAs ───────────────────────────────────────────────────────────
    df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
    df["ema_fast_rising"] = df["ema_fast"] > df["ema_fast"].shift(1)
    df["ema_slow_rising"] = df["ema_slow"] > df["ema_slow"].shift(1)

    # ── Bars since the last fast/slow EMA crossover ─────────────────────────
    above = df["ema_fast"] > df["ema_slow"]
    crossed = above != above.shift(1)
    bars_since = []
    last = 10 ** 9
    for c in crossed:
        last = 0 if c else last + 1
        bars_since.append(last)
    df["bars_since_cross"] = bars_since

    df["body"] = (df["close"] - df["open"]).abs()
    return df


def _compute_metrics(symbol: str, trades: list[ERTradeResult]) -> ERSymbolMetrics:
    if not trades:
        return ERSymbolMetrics(
            symbol=symbol, total_trades=0, wins=0, losses=0, win_rate=0.0,
            total_pnl=0.0, avg_pnl=0.0, profit_factor=0.0, best_trade=0.0,
            worst_trade=0.0, avg_duration_minutes=0.0, max_consecutive_losses=0,
        )

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss else float("inf")

    max_consec = cur_consec = 0
    for p in pnls:
        if p <= 0:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    return ERSymbolMetrics(
        symbol=symbol,
        total_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=round(len(wins) / len(trades) * 100, 1),
        total_pnl=round(sum(pnls), 2),
        avg_pnl=round(sum(pnls) / len(trades), 2),
        profit_factor=profit_factor,
        best_trade=round(max(pnls), 2),
        worst_trade=round(min(pnls), 2),
        avg_duration_minutes=round(
            sum(t.duration_minutes for t in trades) / len(trades), 1
        ),
        max_consecutive_losses=max_consec,
    )


class ERRegimeScalperOptionSecondsStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        # ── Regime / indicators ──
        er_period: int = 10,
        er_trend_threshold: float = 0.4,
        er_strong_threshold: float = 0.6,
        consolidation_range_pct: float = 0.15,
        ema_fast: int = 9,
        ema_slow: int = 21,
        # ── Entry mode & filters ──
        entry_mode: str = "full_filter",
        fresh_cross_bars: int = 5,
        pullback_bars: int = 5,
        block_first_minutes: bool = True,
        first_minutes: int = 20,
        require_prior_alignment: bool = False,
        min_body_enabled: bool = False,
        min_body_points: float = 2.0,
        require_ma_slope: bool = False,
        require_prior_bar_break: bool = False,
        block_lunch: bool = False,
        lunch_start: time = time(12, 0),
        lunch_end: time = time(13, 0),
        stronger_er: bool = False,
        # ── Trade management ──
        tick_size: float = 0.05,
        tp_ticks: int = 100,
        sl_ticks: int = 100,
        breakeven_enabled: bool = False,
        breakeven_ticks: int = 50,
        eod_flatten_time: time = time(15, 20),
        max_trades_per_day: int = 20,
        # ── Data handling ──
        start_date: str = "",
        end_date: str = "",
        interval: str = "1second",
        resample_seconds: int = 120,
        print_trades: bool = False,
        cache_only: bool = False,
        market_holidays: set[date] | None = None,
        per_day_atm: bool = False,
    ):
        self.nifty_service = nifty_service
        self.capital = capital

        self.er_period = er_period
        self.er_trend_threshold = er_trend_threshold
        self.er_strong_threshold = er_strong_threshold
        self.consolidation_range_pct = consolidation_range_pct
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

        if entry_mode not in ENTRY_MODES:
            raise ValueError(f"entry_mode must be one of {ENTRY_MODES}, got {entry_mode!r}")
        self.entry_mode = entry_mode
        self.fresh_cross_bars = fresh_cross_bars
        self.pullback_bars = pullback_bars
        self.block_first_minutes = block_first_minutes
        self.first_minutes = first_minutes
        self.require_prior_alignment = require_prior_alignment
        self.min_body_enabled = min_body_enabled
        self.min_body_points = min_body_points
        self.require_ma_slope = require_ma_slope
        self.require_prior_bar_break = require_prior_bar_break
        self.block_lunch = block_lunch
        self.lunch_start = lunch_start
        self.lunch_end = lunch_end
        self.stronger_er = stronger_er

        self.tick_size = tick_size
        self.tp_ticks = tp_ticks
        self.sl_ticks = sl_ticks
        self.breakeven_enabled = breakeven_enabled
        self.breakeven_ticks = breakeven_ticks
        self.eod_flatten_time = eod_flatten_time
        self.max_trades_per_day = max_trades_per_day

        self.start_date = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date = datetime.strptime(end_date, "%d-%b-%Y").date()
        self.interval = interval
        self.resample_seconds = resample_seconds
        self.print_trades = print_trades
        self.cache_only = cache_only
        self.market_holidays = set(market_holidays) if market_holidays else set()
        self.per_day_atm = per_day_atm

    # ── Signal / option data helpers ────────────────────────────────────────

    def _signal_df(self, from_dt: datetime, to_dt: datetime) -> pd.DataFrame | None:
        spot_candles = self.nifty_service.get_nifty_spot_candles(
            start=from_dt, end=to_dt, interval=self.interval
        )
        if not spot_candles:
            return None
        if self.resample_seconds > 1:
            spot_candles = resample_candles(spot_candles, self.resample_seconds)
        return compute_er_regime(
            candles=spot_candles,
            er_period=self.er_period,
            er_trend_threshold=self.er_trend_threshold,
            consolidation_range_pct=self.consolidation_range_pct,
            ema_fast=self.ema_fast,
            ema_slow=self.ema_slow,
        )

    def _option_ohlc(
        self, strike: int, expiry: date, from_dt: datetime, to_dt: datetime
    ) -> dict[str, dict[datetime, tuple[float, float, float]]]:
        """CE/PE option candles as datetime -> (high, low, close) maps."""
        prices: dict[str, dict[datetime, tuple[float, float, float]]] = {"CE": {}, "PE": {}}
        for opt_type in ("CE", "PE"):
            try:
                opt_candles = self.nifty_service.get_option_candles(
                    strike=strike, expiry_date=expiry, option_type=opt_type,
                    start=from_dt, end=to_dt, interval=self.interval,
                    cache_only=self.cache_only,
                ) or []
                if self.resample_seconds > 1:
                    opt_candles = resample_candles(opt_candles, self.resample_seconds)
                m: dict[datetime, tuple[float, float, float]] = {}
                for c in opt_candles:
                    dt = pd.to_datetime(c["datetime"]).to_pydatetime()
                    m[dt] = (float(c["high"]), float(c["low"]), float(c["close"]))
                prices[opt_type] = m
                print(f"    {opt_type}: {len(opt_candles)} candles fetched")
            except Exception as exc:
                print(f"    [{opt_type}] Fetch error: {exc}")
        return prices

    def _option_data_cached(
        self, strike: int, expiry: date, from_dt: datetime, to_dt: datetime
    ) -> bool:
        for opt_type in ("CE", "PE"):
            cached = self.nifty_service.get_option_candles(
                strike=strike, expiry_date=expiry, option_type=opt_type,
                start=from_dt, end=to_dt, interval=self.interval, cache_only=True,
            )
            if not cached:
                return False
        return True

    # ── Entry-condition helpers ─────────────────────────────────────────────

    def _alignment(self, row, direction: str) -> bool:
        """The mode's alignment condition for `direction` evaluated on `row`."""
        close, vwap = row.close, row.vwap
        fast, slow = row.ema_fast, row.ema_slow
        if direction == "long":
            vwap_ok = close > vwap
            ema_ok = fast > slow
            ma_vs_vwap = fast > vwap and slow > vwap
        else:
            vwap_ok = close < vwap
            ema_ok = fast < slow
            ma_vs_vwap = fast < vwap and slow < vwap

        mode = self.entry_mode
        if mode == "full_filter":
            return vwap_ok and ema_ok and ma_vs_vwap
        if mode == "vwap_only":
            return vwap_ok
        if mode == "ema_only":
            return ema_ok
        if mode == "regime_only":
            return True
        if mode == "fresh_cross":
            return ema_ok and row.bars_since_cross <= self.fresh_cross_bars
        if mode == "pullback":
            # Confirming regime candle; the pullback-touch is checked separately.
            return ema_ok
        return False

    def _pullback_touched(self, rows: list, direction: str) -> bool:
        """True if price pulled back to the fast MA within the last
        `pullback_bars` bars (low<=fast<=high)."""
        window = rows[-self.pullback_bars:] if self.pullback_bars > 0 else []
        for r in window:
            if r.low <= r.ema_fast <= r.high:
                return True
        return False

    def _filters_pass(self, row, prev_row, rows: list, direction: str) -> bool:
        t: time = row.datetime.time()
        session_open = time(9, 15)

        if self.block_first_minutes:
            cutoff = (
                datetime.combine(date.min, session_open)
                + timedelta(minutes=self.first_minutes)
            ).time()
            if t < cutoff:
                return False

        if self.block_lunch and self.lunch_start <= t < self.lunch_end:
            return False

        if self.stronger_er and row.er < self.er_strong_threshold:
            return False

        if self.min_body_enabled and row.body < self.min_body_points:
            return False

        if self.require_ma_slope:
            if direction == "long" and not (row.ema_fast_rising and row.ema_slow_rising):
                return False
            if direction == "short" and not (
                not row.ema_fast_rising and not row.ema_slow_rising
            ):
                return False

        if self.require_prior_bar_break and prev_row is not None:
            if direction == "long" and not (row.close > prev_row.high):
                return False
            if direction == "short" and not (row.close < prev_row.low):
                return False

        if self.require_prior_alignment and prev_row is not None:
            if not self._alignment(prev_row, direction):
                return False

        if self.entry_mode == "pullback" and not self._pullback_touched(rows, direction):
            return False

        return True

    # ── Core per-expiry / per-day state machine ─────────────────────────────

    def _run_signals(
        self,
        signal_df: pd.DataFrame,
        ce_prices: dict[datetime, tuple[float, float, float]],
        pe_prices: dict[datetime, tuple[float, float, float]],
        strike: int,
        expiry: date,
    ) -> tuple[list[ERTradeResult], list[ERTradeResult]]:
        ce_trades: list[ERTradeResult] = []
        pe_trades: list[ERTradeResult] = []

        tp_pts = self.tp_ticks * self.tick_size
        sl_pts = self.sl_ticks * self.tick_size
        be_pts = self.breakeven_ticks * self.tick_size

        in_position = False
        direction = ""
        opt_type = ""
        entry_price = 0.0
        entry_dt: datetime | None = None
        shares = 0
        stop = 0.0
        target = 0.0
        moved_be = False
        regime_at_entry = ""
        er_at_entry = 0.0

        prev_regime = _GREY
        trades_today = 0
        cur_day: date | None = None
        rows_seen: list = []

        rows = list(signal_df.itertuples(index=False))

        def _record(exit_dt, exit_price, reason):
            nonlocal in_position, entry_dt
            duration = int((exit_dt - entry_dt).total_seconds() / 60)
            pnl = round(shares * (exit_price - entry_price), 2)
            trade = ERTradeResult(
                symbol=f"NIFTY{strike}{opt_type}",
                option_type=opt_type,
                strike=strike,
                expiry_date=expiry,
                entry_time=entry_dt,
                exit_time=exit_dt,
                entry_price=round(entry_price, 2),
                exit_price=round(exit_price, 2),
                shares=shares,
                pnl=pnl,
                exit_reason=reason,
                direction=direction,
                entry_mode=self.entry_mode,
                regime_at_entry=regime_at_entry,
                er_at_entry=round(er_at_entry, 3),
                duration_minutes=duration,
            )
            (ce_trades if opt_type == "CE" else pe_trades).append(trade)
            in_position = False

        for idx, row in enumerate(rows):
            dt: datetime = row.datetime
            day = dt.date()
            if day != cur_day:
                cur_day = day
                trades_today = 0
                rows_seen = []
            rows_seen.append(row)

            prices = ce_prices if opt_type == "CE" else pe_prices
            opt = prices.get(dt)

            # ── Exit logic ──────────────────────────────────────────────────
            if in_position:
                t = dt.time()
                # EOD flatten — use the option close if available.
                if t >= self.eod_flatten_time:
                    exit_price = opt[2] if opt else entry_price
                    _record(dt, exit_price, "EOD_FLATTEN")
                # Regime shifted away from the trade direction → flatten now.
                elif (direction == "long" and row.regime != _GREEN) or (
                    direction == "short" and row.regime != _RED
                ):
                    exit_price = opt[2] if opt else entry_price
                    _record(dt, exit_price, "REGIME_SHIFT")
                elif opt is not None:
                    high, low, close = opt
                    # Move stop to breakeven once far enough in favour.
                    if (
                        self.breakeven_enabled
                        and not moved_be
                        and high >= entry_price + be_pts
                    ):
                        stop = entry_price
                        moved_be = True
                    if low <= stop:
                        reason = "BREAKEVEN" if moved_be and stop >= entry_price else "STOP_LOSS"
                        _record(dt, stop, reason)
                    elif high >= target:
                        _record(dt, target, "TARGET")
                # else: no option price this bar — hold.

            # Update regime memory AFTER exits so a regime-shift exit uses the
            # transition, then continue to entry evaluation on the same bar.
            regime = row.regime

            # ── Entry logic ─────────────────────────────────────────────────
            if (
                not in_position
                and idx >= self.er_period
                and trades_today < self.max_trades_per_day
                and regime in _TRENDING
                and prev_regime not in _TRENDING  # fresh transition into a trend
            ):
                direction = "long" if regime == _GREEN else "short"
                opt_type = "CE" if direction == "long" else "PE"
                prev_row = rows[idx - 1] if idx > 0 else None

                entry_opt = (ce_prices if opt_type == "CE" else pe_prices).get(dt)
                if (
                    entry_opt is not None
                    and entry_opt[2] > 0
                    and self._alignment(row, direction)
                    and self._filters_pass(row, prev_row, rows_seen, direction)
                ):
                    entry_price = entry_opt[2]
                    stop = round(entry_price - sl_pts, 2)
                    target = round(entry_price + tp_pts, 2)
                    alloc = _capital_allocation_pct(entry_price)
                    shares = max(floor(self.capital * alloc / entry_price), 1)
                    moved_be = False
                    regime_at_entry = regime
                    er_at_entry = row.er
                    entry_dt = dt
                    in_position = True
                    trades_today += 1

            prev_regime = regime

        # Force-flatten any open position at the end of the data.
        if in_position and entry_dt is not None and rows:
            last = rows[-1]
            prices = ce_prices if opt_type == "CE" else pe_prices
            opt = prices.get(last.datetime)
            exit_price = opt[2] if opt else entry_price
            _record(last.datetime, exit_price, "EOD_FLATTEN")

        return ce_trades, pe_trades

    # ── Weekly expiry orchestration ─────────────────────────────────────────

    def run_weekly_backtest(self) -> list[ERWeeklyExpiryResult]:
        expiry_results: list[ERWeeklyExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        effective_tf = (
            f"{self.resample_seconds}s" if self.resample_seconds > 1 else self.interval
        )
        print(f"\n{'='*70}")
        print(f"  NIFTY XIZNIT ER REGIME SCALPER — WEEKLY EXPIRY BACKTEST (seconds data)")
        print(f"  Period   : {self.start_date}  →  {self.end_date}")
        print(f"  Fetch TF : {self.interval}  |  Effective TF: {effective_tf}")
        print(f"  Capital  : ₹{self.capital:,.0f}")
        print(f"  ER       : period {self.er_period}  trend≥{self.er_trend_threshold}"
              f"  strong≥{self.er_strong_threshold}")
        print(f"  EMA      : {self.ema_fast}/{self.ema_slow}  |  Mode: {self.entry_mode}")
        print(f"  TP/SL    : {self.tp_ticks}/{self.sl_ticks} ticks @ {self.tick_size}"
              f"  |  BE: {'ON ' + str(self.breakeven_ticks) + 't' if self.breakeven_enabled else 'OFF'}")
        print(f"  Mode     : {'per-day ATM' if self.per_day_atm else 'weekly ATM'}"
              f"  |  Cache-only: {self.cache_only}")
        print(f"  Expiries found: {len(wednesdays)}")
        print(f"{'='*70}\n")

        for tuesday in wednesdays:
            monday = NiftyOptionService.monday_of_week(tuesday)
            win_start, _ = NiftyOptionService.week_window(tuesday)

            expiry = NiftyOptionService.adjust_expiry_for_holidays(
                tuesday, self.market_holidays
            )
            if expiry != tuesday:
                print(f"  [holiday] Expiry {tuesday} is a holiday — rolled back to {expiry}")
            win_end = expiry

            if self.per_day_atm:
                week_result = self._run_expiry_per_day(expiry, win_start, win_end)
            else:
                week_result = self._run_expiry_weekly(expiry, monday, win_start, win_end)

            if week_result is not None:
                expiry_results.append(week_result)

        return expiry_results

    def _run_expiry_weekly(
        self, expiry: date, monday: date, win_start: date, win_end: date
    ) -> ERWeeklyExpiryResult | None:
        try:
            nifty_open = self.nifty_service.get_nifty_open(monday)
        except Exception as exc:
            print(f"  [{expiry}] Could not get Nifty open for {monday}: {exc}")
            return None

        strike = NiftyOptionService.atm_strike(nifty_open)
        print(f"  Expiry {expiry}  |  Monday open {nifty_open:.2f}  |  ATM {strike}")

        from_dt = datetime(win_start.year, win_start.month, win_start.day, 9, 15, 0)
        to_dt = datetime(win_end.year, win_end.month, win_end.day, 15, 30, 0)

        if self.cache_only and not self._option_data_cached(strike, expiry, from_dt, to_dt):
            print(f"    [cache-only] No cached data — skipping expiry {expiry}")
            return None

        signal_df = self._signal_df(from_dt, to_dt)
        if signal_df is None:
            print(f"    No spot candles — skipping")
            return None

        prices = self._option_ohlc(strike, expiry, from_dt, to_dt)

        week_result = ERWeeklyExpiryResult(
            expiry_date=expiry, atm_strike=strike, nifty_open=nifty_open,
        )
        ce_trades, pe_trades = self._run_signals(
            signal_df, prices["CE"], prices["PE"], strike, expiry
        )
        week_result.ce_trades = ce_trades
        week_result.pe_trades = pe_trades

        self._print_week_summary(week_result, ce_trades, pe_trades)
        return week_result

    def _run_expiry_per_day(
        self, expiry: date, win_start: date, win_end: date
    ) -> ERWeeklyExpiryResult | None:
        print(f"  Expiry {expiry}  |  per-day ATM")

        week_result = ERWeeklyExpiryResult(
            expiry_date=expiry, atm_strike=0, nifty_open=0.0,
        )

        days = self.nifty_service.trading_days(win_start, win_end, self.market_holidays)

        for day in days:
            try:
                nifty_open = self.nifty_service.get_nifty_open(day)
            except Exception as exc:
                print(f"    [{day}] Could not get Nifty open: {exc}")
                continue

            strike = NiftyOptionService.atm_strike(nifty_open)
            if day == expiry:
                week_result.atm_strike = strike
                week_result.nifty_open = nifty_open

            from_dt = datetime(day.year, day.month, day.day, 9, 15, 0)
            to_dt = datetime(day.year, day.month, day.day, 15, 30, 0)

            print(f"    {day}  |  open {nifty_open:.2f}  |  ATM {strike}")

            if self.cache_only and not self._option_data_cached(
                strike, expiry, from_dt, to_dt
            ):
                print(f"      [cache-only] No cached data — skipping {day}")
                continue

            signal_df = self._signal_df(from_dt, to_dt)
            if signal_df is None:
                print(f"      No spot candles — skipping {day}")
                continue

            prices = self._option_ohlc(strike, expiry, from_dt, to_dt)
            ce_trades, pe_trades = self._run_signals(
                signal_df, prices["CE"], prices["PE"], strike, expiry
            )
            week_result.ce_trades.extend(ce_trades)
            week_result.pe_trades.extend(pe_trades)
            print(f"      Trades: CE {len(ce_trades)}  PE {len(pe_trades)}")

        if week_result.all_trades:
            self._print_week_summary(week_result, [], [], header=False)
        return week_result

    @staticmethod
    def _print_week_summary(week_result, ce_trades, pe_trades, header=True):
        total = len(week_result.all_trades)
        week_pnl = sum(t.pnl for t in week_result.all_trades)
        pnl_sign = "+" if week_pnl >= 0 else ""
        if header:
            print(f"    Trades: {total} (CE:{len(ce_trades)} PE:{len(pe_trades)})"
                  f"  |  PnL: {pnl_sign}{week_pnl:.2f}")
        else:
            print(f"    Week total: {total} trades  |  PnL: {pnl_sign}{week_pnl:.2f}")

    # ── Reporting ────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(expiry_results: list[ERWeeklyExpiryResult]) -> None:
        all_trades: list[ERTradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[ERTradeResult]] = defaultdict(list)
        for t in all_trades:
            by_symbol[t.symbol].append(t)

        sep = "─" * 96
        print(f"\n{'='*96}")
        print("  RESULTS BY EXPIRY")
        print(f"{'='*96}")

        for er in expiry_results:
            if not er.all_trades:
                continue
            total_pnl = sum(t.pnl for t in er.all_trades)
            pnl_sign = "+" if total_pnl >= 0 else ""
            print(f"\n  Expiry {er.expiry_date}  |  ATM {er.atm_strike}"
                  f"  |  Nifty open {er.nifty_open:.2f}"
                  f"  |  Trades {len(er.all_trades)}"
                  f"  |  PnL {pnl_sign}{total_pnl:.2f}")
            print(f"  {sep}")

            header = (
                f"  {'Symbol':<18} {'Dir':<5} {'Entry':>14} {'Exit':>14}"
                f" {'Entry₹':>8} {'Exit₹':>8} {'Qty':>5}"
                f" {'PnL':>10} {'Reason':<14} {'ER':>5}"
            )
            print(header)
            print(f"  {sep}")

            for t in sorted(er.all_trades, key=lambda x: x.entry_time):
                pnl_sign = "+" if t.pnl >= 0 else ""
                print(
                    f"  {t.symbol:<18} {t.direction:<5}"
                    f" {t.entry_time.strftime('%d-%b %H:%M'):>14}"
                    f" {t.exit_time.strftime('%d-%b %H:%M'):>14}"
                    f" {t.entry_price:>8.2f} {t.exit_price:>8.2f} {t.shares:>5}"
                    f" {pnl_sign+f'{t.pnl:.2f}':>10} {t.exit_reason:<14} {t.er_at_entry:>5.2f}"
                )

        print(f"\n{'='*96}")
        print("  SYMBOL-WISE METRICS")
        print(f"{'='*96}")

        col_w = {"sym": 18, "trades": 7, "wins": 5, "loss": 6, "wr": 6, "pnl": 10,
                 "avg": 9, "pf": 7, "best": 9, "worst": 9, "dur": 7, "cons": 5}

        hdr = (
            f"  {'Symbol':<{col_w['sym']}} {'Trades':>{col_w['trades']}}"
            f" {'Wins':>{col_w['wins']}} {'Loss':>{col_w['loss']}}"
            f" {'Win%':>{col_w['wr']}} {'Total PnL':>{col_w['pnl']}}"
            f" {'Avg PnL':>{col_w['avg']}} {'PF':>{col_w['pf']}}"
            f" {'Best':>{col_w['best']}} {'Worst':>{col_w['worst']}}"
            f" {'AvgMin':>{col_w['dur']}} {'MaxCL':>{col_w['cons']}}"
        )
        print(hdr)
        print(f"  {'─'*94}")

        overall_pnl = 0.0
        overall_trades = overall_wins = overall_losses = 0

        for sym, trades in sorted(by_symbol.items()):
            m = _compute_metrics(sym, trades)
            overall_pnl += m.total_pnl
            overall_trades += m.total_trades
            overall_wins += m.wins
            overall_losses += m.losses

            pnl_s = f"{'+' if m.total_pnl >= 0 else ''}{m.total_pnl:.2f}"
            avg_s = f"{'+' if m.avg_pnl >= 0 else ''}{m.avg_pnl:.2f}"
            pf_s = f"{m.profit_factor:.2f}" if m.profit_factor != float("inf") else "∞"

            print(
                f"  {m.symbol:<{col_w['sym']}} {m.total_trades:>{col_w['trades']}}"
                f" {m.wins:>{col_w['wins']}} {m.losses:>{col_w['loss']}}"
                f" {m.win_rate:>{col_w['wr']}.1f} {pnl_s:>{col_w['pnl']}}"
                f" {avg_s:>{col_w['avg']}} {pf_s:>{col_w['pf']}}"
                f" {m.best_trade:>{col_w['best']}.2f} {m.worst_trade:>{col_w['worst']}.2f}"
                f" {m.avg_duration_minutes:>{col_w['dur']}.1f} {m.max_consecutive_losses:>{col_w['cons']}}"
            )

        wr_overall = round(overall_wins / overall_trades * 100, 1) if overall_trades else 0.0
        pnl_s = f"{'+' if overall_pnl >= 0 else ''}{overall_pnl:.2f}"
        print(f"  {'─'*94}")
        print(
            f"  {'OVERALL':<{col_w['sym']}} {overall_trades:>{col_w['trades']}}"
            f" {overall_wins:>{col_w['wins']}} {overall_losses:>{col_w['loss']}}"
            f" {wr_overall:>{col_w['wr']}.1f} {pnl_s:>{col_w['pnl']}}"
        )
        print(f"{'='*96}\n")
