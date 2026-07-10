"""
Bowman RSI Signals strategy for Nifty weekly options (seconds data).

Python adaptation of the TradingView Pine script "Bowman RSI Signals
Backtester v3", run on the option premium instead of the underlying chart.

Entry rules (long the option premium, CE & PE legs):
  - The fast RSI (`rsi_length`) crosses UP through the oversold level (30),
  - the higher-timeframe RSI (`htf_rsi_length` computed on `htf_seconds`
    resampled bars) is below `htf_rsi_cutoff` (the move starts from a
    depressed higher-timeframe RSI), and
  - optionally, price is above a higher-timeframe EMA (`ema_length` on
    `ema_seconds` resampled bars).

Exit rules — one of four modes (`exit_mode`):
  - "ATR_TRAILING_STOP":   trailing stop `atr_mult` × ATR(`atr_length`) below
                           the peak price since entry (activated once price has
                           moved that distance above entry, mirroring the Pine
                           `trail_offset`).
  - "BEAR_DIV_PIVOT_HIGH": close when a bearish RSI divergence (price higher,
                           RSI lower over `div_lookback` bars, RSI > 70)
                           coincides with a confirmed pivot high.
  - "PIVOT_HIGH_ONLY":     close on any confirmed pivot high.
  - "FIXED_PERCENTAGES":   fixed take-profit / stop-loss percentages
                           (`fixed_profit_pct` / `fixed_loss_pct`).

Common intraday guards (as in the other option strategies):
  - Square-off at 15:20 IST                 → close position
  - No new entries before 9:30 or after 14:45
  - Max 5 trades per day per symbol

Higher-timeframe series are built by resampling the option's own candles to
`htf_seconds` / `ema_seconds` buckets; only COMPLETED higher-timeframe bars
are used (values are forward-filled from the previous closed bucket) so the
signals do not look ahead within an unfinished HTF bar.
"""

from collections import defaultdict
from datetime import date, datetime, time
from math import floor

import numpy as np
import pandas as pd

from models.bowman_rsi_models import (
    BowmanRSISymbolMetrics,
    BowmanRSITradeResult,
    BowmanRSIWeeklyExpiryResult,
)
from services.nifty_option_service import NiftyOptionService


def resample_candles(candles: list[dict], seconds: int) -> list[dict]:
    """
    Resample a list of 1-second OHLC dicts into N-second candles.

    Each input dict must have keys: datetime, open, high, low, close, volume
    (optional). Output dicts have the same schema; datetime is the candle
    open-time (floor of the N-second bucket).

    seconds=1 returns the original data unchanged.
    """
    if seconds <= 1 or not candles:
        return candles

    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    df["_bucket"] = df["datetime"].apply(lambda ts: ts.floor(f"{seconds}s"))

    agg: dict = {
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
    }
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        agg["volume"] = "sum"

    resampled = (
        df.groupby("_bucket")
        .agg(agg)
        .reset_index()
        .rename(columns={"_bucket": "datetime"})
    )
    resampled["datetime"] = pd.to_datetime(resampled["datetime"])
    return resampled.to_dict("records")


_ENTRY_START  = time(9, 30)
_ENTRY_CUTOFF = time(14, 45)
_SQUARE_OFF   = time(15, 20)
_MAX_TRADES_PER_DAY = 5

EXIT_MODES = (
    "ATR_TRAILING_STOP",
    "BEAR_DIV_PIVOT_HIGH",
    "PIVOT_HIGH_ONLY",
    "FIXED_PERCENTAGES",
)


def _capital_allocation_pct(price: float) -> float:
    """
    Return the fraction of capital to allocate based on option price.
    Mirrors the allocation tiers used by the other option strategies.
    """
    if price <= 20:
        return 0.30
    else:
        return 1.00


def wilder_rsi(close: pd.Series, length: int) -> pd.Series:
    """
    RSI using Wilder's smoothing (RMA), matching Pine's ta.rsi().
    The first `length` bars are masked to NaN as warm-up.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

    rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    rsi = rsi.where(avg_loss > 0, 100.0)
    rsi[avg_gain.isna() | avg_loss.isna()] = np.nan
    return rsi


def wilder_atr(df: pd.DataFrame, length: int) -> pd.Series:
    """ATR using Wilder's smoothing (RMA), matching Pine's ta.atr()."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def pivot_high_flags(high: pd.Series, left: int, right: int) -> pd.Series:
    """
    Boolean series matching Pine's `not na(ta.pivothigh(high, left, right))`:
    True at bar i when the bar `right` bars ago is a strict local maximum over
    its `left` preceding and `right` following bars (i.e. the pivot is
    CONFIRMED at bar i, `right` bars after it happened).
    """
    n = len(high)
    values = high.to_numpy(dtype=float)
    flags = np.zeros(n, dtype=bool)
    for i in range(left + right, n):
        p = i - right
        candidate = values[p]
        window = values[p - left: p + right + 1]
        # Strict maximum: strictly greater than every neighbour in the window.
        if candidate > np.delete(window, left).max():
            flags[i] = True
    return pd.Series(flags, index=high.index)


def _htf_series_ffill(
    df: pd.DataFrame,
    htf_seconds: int,
    compute: "callable[[pd.DataFrame], pd.Series]",
) -> pd.Series:
    """
    Resample `df` (datetime/open/high/low/close) into `htf_seconds` buckets,
    run `compute` on the resampled frame to get one value per HTF bar, then
    map each base bar to the value of the last COMPLETED HTF bar (no
    lookahead into the still-forming bucket).
    """
    bucket = df["datetime"].dt.floor(f"{htf_seconds}s")
    htf = (
        df.groupby(bucket)
        .agg(open=("open", "first"), high=("high", "max"),
             low=("low", "min"), close=("close", "last"))
        .reset_index()
        .rename(columns={"datetime": "bucket"})
    )
    htf["value"] = compute(htf)
    # Value becomes available only after the bucket has closed → shift by one.
    lookup = htf.set_index("bucket")["value"].shift(1)
    return bucket.map(lookup)


def compute_bowman(
    candles: list[dict],
    rsi_length: int,
    div_lookback: int,
    htf_seconds: int,
    htf_rsi_length: int,
    use_ema_filter: bool,
    ema_seconds: int,
    ema_length: int,
    atr_length: int,
) -> pd.DataFrame:
    """
    Compute all indicator series for the Bowman RSI strategy from a list of
    OHLC dicts.

    Returns a DataFrame with columns: datetime, close, high, low, rsi,
    htf_rsi, htf_ema, atr, pivot_high (bool) and bear_div (bool).
    """
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    rsi = wilder_rsi(df["close"], rsi_length)
    atr = wilder_atr(df, atr_length)

    htf_rsi = _htf_series_ffill(
        df, htf_seconds, lambda h: wilder_rsi(h["close"], htf_rsi_length)
    )

    if use_ema_filter:
        htf_ema = _htf_series_ffill(
            df,
            ema_seconds,
            lambda h: h["close"].ewm(span=ema_length, adjust=False,
                                     min_periods=ema_length).mean(),
        )
    else:
        htf_ema = pd.Series(np.nan, index=df.index)

    pivoth = pivot_high_flags(df["high"], div_lookback, div_lookback)

    # Bearish divergence: price higher high with RSI lower high, in overbought
    # territory (mirrors the Pine bearDiv condition).
    bear_div = (
        (df["close"] > df["close"].shift(div_lookback))
        & (rsi < rsi.shift(div_lookback))
        & (rsi > 70)
    ).fillna(False)

    result = df[["datetime", "close", "high", "low"]].copy()
    result["rsi"]        = rsi
    result["htf_rsi"]    = htf_rsi
    result["htf_ema"]    = htf_ema
    result["atr"]        = atr
    result["pivot_high"] = pivoth
    result["bear_div"]   = bear_div
    return result


def _compute_metrics(
    symbol: str, trades: list[BowmanRSITradeResult]
) -> BowmanRSISymbolMetrics:
    if not trades:
        return BowmanRSISymbolMetrics(
            symbol=symbol, total_trades=0, wins=0, losses=0,
            win_rate=0.0, total_pnl=0.0, avg_pnl=0.0,
            profit_factor=0.0, best_trade=0.0, worst_trade=0.0,
            avg_duration_minutes=0.0, max_consecutive_losses=0,
        )

    pnls = [t.pnl for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss else float("inf")

    max_consec = cur_consec = 0
    for p in pnls:
        if p <= 0:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    return BowmanRSISymbolMetrics(
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


class BowmanRSIOptionStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        rsi_length: int = 7,
        rsi_oversold: float = 30.0,
        div_lookback: int = 2,
        htf_seconds: int = 3600,
        htf_rsi_length: int = 5,
        htf_rsi_cutoff: float = 40.0,
        use_ema_filter: bool = True,
        ema_seconds: int = 1800,
        ema_length: int = 200,
        exit_mode: str = "ATR_TRAILING_STOP",
        atr_length: int = 14,
        atr_mult: float = 3.0,
        fixed_profit_pct: float = 2.0,
        fixed_loss_pct: float = 1.0,
        start_date: str = "",
        end_date: str = "",
        interval: str = "1minute",
        resample_seconds: int = 1,
        print_resampled: bool = False,
        cache_only: bool = False,
        market_holidays: set[date] | None = None,
        per_day_atm: bool = False,
    ):
        if exit_mode not in EXIT_MODES:
            raise ValueError(
                f"exit_mode must be one of {EXIT_MODES}, got {exit_mode!r}"
            )

        self.nifty_service  = nifty_service
        self.capital        = capital
        # Fast RSI on the option premium; a cross UP through `rsi_oversold`
        # is the entry trigger.
        self.rsi_length     = rsi_length
        self.rsi_oversold   = rsi_oversold
        # Lookback (bars) for the bearish-divergence comparison, and the
        # left/right width of the pivot-high detection.
        self.div_lookback   = div_lookback
        # Higher-timeframe RSI filter: RSI(`htf_rsi_length`) on `htf_seconds`
        # resampled bars must be below `htf_rsi_cutoff` at entry.
        self.htf_seconds    = htf_seconds
        self.htf_rsi_length = htf_rsi_length
        self.htf_rsi_cutoff = htf_rsi_cutoff
        # Optional EMA trend filter: price must be above EMA(`ema_length`)
        # computed on `ema_seconds` resampled bars.
        self.use_ema_filter = use_ema_filter
        self.ema_seconds    = ema_seconds
        self.ema_length     = ema_length
        self.exit_mode      = exit_mode
        self.atr_length     = atr_length
        self.atr_mult       = atr_mult
        # Fixed-percentage exit levels (used only in FIXED_PERCENTAGES mode).
        self.fixed_profit_pct = fixed_profit_pct
        self.fixed_loss_pct   = fixed_loss_pct
        self.start_date     = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date       = datetime.strptime(end_date,   "%d-%b-%Y").date()
        self.interval       = interval
        self.resample_seconds = resample_seconds
        self.print_resampled  = print_resampled
        self.cache_only       = cache_only
        self.market_holidays  = set(market_holidays) if market_holidays else set()
        self.per_day_atm      = per_day_atm

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _compute_indicators(self, candles: list[dict]) -> pd.DataFrame:
        return compute_bowman(
            candles,
            rsi_length=self.rsi_length,
            div_lookback=self.div_lookback,
            htf_seconds=self.htf_seconds,
            htf_rsi_length=self.htf_rsi_length,
            use_ema_filter=self.use_ema_filter,
            ema_seconds=self.ema_seconds,
            ema_length=self.ema_length,
            atr_length=self.atr_length,
        )

    def _run_symbol(
        self,
        candles: list[dict],
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[BowmanRSITradeResult]:
        """
        Run the Bowman RSI strategy on a single option contract's candle data.
        Returns a list of completed trades.
        """
        if not candles:
            return []

        indicators = self._compute_indicators(candles)
        symbol_label = f"NIFTY{strike}{option_type}"
        trades: list[BowmanRSITradeResult] = []

        daily_trade_count: dict[date, int] = defaultdict(int)

        in_position   = False
        entry_row     = None
        entry_price   = 0.0
        shares        = 0
        atr_at_entry  = 0.0
        trail_dist    = 0.0     # ATR trailing-stop distance (points)
        trail_active  = False   # becomes True once price moves trail_dist above entry
        peak_price    = 0.0
        tp_price      = 0.0     # FIXED_PERCENTAGES take-profit
        sl_price      = 0.0     # FIXED_PERCENTAGES stop-loss

        prev_rsi = None

        def _record_trade(dt: datetime, row, exit_price: float, reason: str) -> None:
            duration = int((dt - entry_row["datetime"]).total_seconds() / 60)
            trades.append(BowmanRSITradeResult(
                symbol=symbol_label,
                option_type=option_type,
                strike=strike,
                expiry_date=expiry_date,
                entry_time=entry_row["datetime"],
                exit_time=dt,
                entry_price=entry_price,
                exit_price=round(exit_price, 2),
                shares=shares,
                pnl=round(shares * (exit_price - entry_price), 2),
                exit_reason=reason,
                rsi_at_entry=round(entry_row["rsi"], 2),
                htf_rsi_at_entry=round(entry_row["htf_rsi"], 2),
                ema_at_entry=(
                    round(entry_row["htf_ema"], 2)
                    if not pd.isna(entry_row["htf_ema"]) else float("nan")
                ),
                rsi_at_exit=round(row["rsi"], 2) if not pd.isna(row["rsi"]) else float("nan"),
                atr_at_entry=round(atr_at_entry, 4),
                duration_minutes=duration,
            ))

        for _, row in indicators.iterrows():
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            today: date  = dt.date()

            price   = row["close"]
            rsi     = row["rsi"]
            htf_rsi = row["htf_rsi"]
            htf_ema = row["htf_ema"]

            # ── Exit logic (checked every bar while in position) ──────────
            if in_position:
                if price > peak_price:
                    peak_price = price

                exit_reason = None
                if t >= _SQUARE_OFF:
                    exit_reason = "SQUARE_OFF"
                elif self.exit_mode == "ATR_TRAILING_STOP":
                    if not trail_active and price >= entry_price + trail_dist:
                        trail_active = True
                    if trail_active and price <= peak_price - trail_dist:
                        exit_reason = "ATR_TRAIL"
                elif self.exit_mode == "BEAR_DIV_PIVOT_HIGH":
                    if row["bear_div"] and row["pivot_high"]:
                        exit_reason = "BEAR_DIV_PIVOT"
                elif self.exit_mode == "PIVOT_HIGH_ONLY":
                    if row["pivot_high"]:
                        exit_reason = "PIVOT_HIGH"
                elif self.exit_mode == "FIXED_PERCENTAGES":
                    if price >= tp_price:
                        exit_reason = "FIXED_TARGET"
                    elif price <= sl_price:
                        exit_reason = "FIXED_STOP"

                if exit_reason:
                    _record_trade(dt, row, price, exit_reason)
                    in_position = False
                    entry_row   = None
                    prev_rsi    = rsi
                    continue

            # ── Entry logic ───────────────────────────────────────────────
            if (
                not in_position
                and not pd.isna(rsi)
                and prev_rsi is not None
                and not pd.isna(prev_rsi)
                and _ENTRY_START <= t <= _ENTRY_CUTOFF
                and daily_trade_count[today] < _MAX_TRADES_PER_DAY
            ):
                cross_up = prev_rsi <= self.rsi_oversold and rsi > self.rsi_oversold
                htf_ok   = not pd.isna(htf_rsi) and htf_rsi < self.htf_rsi_cutoff
                ema_ok   = (
                    not self.use_ema_filter
                    or (not pd.isna(htf_ema) and price > htf_ema)
                )

                if cross_up and htf_ok and ema_ok and price > 0:
                    atr = row["atr"]
                    if self.exit_mode == "ATR_TRAILING_STOP" and (
                        pd.isna(atr) or atr <= 0
                    ):
                        prev_rsi = rsi
                        continue  # ATR not warmed up yet — skip the signal

                    entry_price = price
                    alloc_pct   = _capital_allocation_pct(entry_price)
                    shares      = max(floor(self.capital * alloc_pct / entry_price), 1)
                    in_position = True
                    entry_row   = row
                    peak_price  = entry_price

                    atr_at_entry = float(atr) if not pd.isna(atr) else 0.0
                    trail_dist   = atr_at_entry * self.atr_mult
                    trail_active = False
                    tp_price = entry_price * (1 + self.fixed_profit_pct / 100.0)
                    sl_price = entry_price * (1 - self.fixed_loss_pct / 100.0)

                    daily_trade_count[today] += 1

            prev_rsi = rsi

        # Force-close any open position at end of data
        if in_position and entry_row is not None:
            last = indicators.iloc[-1]
            _record_trade(last["datetime"], last, last["close"], "SQUARE_OFF")

        return trades

    # ── Debug / inspection helpers ────────────────────────────────────────────

    def _print_resampled_with_trades(
        self,
        candles: list[dict],
        trades: list[BowmanRSITradeResult],
        strike: int,
        option_type: str,
        expiry_date: date,
    ) -> None:
        symbol_label = f"NIFTY{strike}{option_type}"
        print(f"\n{'='*90}")
        print(f"  RESAMPLED DATA + TRADES — {symbol_label}  (expiry {expiry_date})")
        print(f"{'='*90}")

        if not candles:
            print("  (no candle data)")
            return

        indicators = self._compute_indicators(candles)

        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", None,
        ):
            print(indicators.to_string(index=False))

        print(f"\n  Trades for {symbol_label}: {len(trades)}")
        for t in sorted(trades, key=lambda x: x.entry_time):
            pnl_sign = "+" if t.pnl >= 0 else ""
            print(
                f"    {t.entry_time.strftime('%d-%b %H:%M:%S')} → "
                f"{t.exit_time.strftime('%d-%b %H:%M:%S')}"
                f"  entry {t.entry_price:.2f}  exit {t.exit_price:.2f}"
                f"  qty {t.shares}  pnl {pnl_sign}{t.pnl:.2f}"
                f"  ({t.exit_reason})"
            )

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[BowmanRSIWeeklyExpiryResult]:
        expiry_results: list[BowmanRSIWeeklyExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        effective_tf = (
            f"{self.resample_seconds}s" if self.resample_seconds > 1 else self.interval
        )
        print(f"\n{'='*70}")
        print(f"  NIFTY BOWMAN RSI SIGNALS — WEEKLY EXPIRY BACKTEST")
        print(f"  Period    : {self.start_date}  →  {self.end_date}")
        print(f"  Fetch TF  : {self.interval}  |  Effective TF: {effective_tf}")
        print(f"  Capital   : ₹{self.capital:,.0f}  |  RSI: {self.rsi_length}"
              f"  |  HTF RSI: {self.htf_rsi_length}@{self.htf_seconds}s"
              f" <{self.htf_rsi_cutoff}")
        print(f"  EMA filter: {self.use_ema_filter}"
              f" (EMA{self.ema_length}@{self.ema_seconds}s)"
              f"  |  Exit: {self.exit_mode}"
              f"  |  ATR: {self.atr_length}x{self.atr_mult}")
        print(f"  Expiries found: {len(wednesdays)}")
        print(f"{'='*70}\n")

        for tuesday in wednesdays:
            monday       = NiftyOptionService.monday_of_week(tuesday)
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
    ) -> BowmanRSIWeeklyExpiryResult | None:
        """
        Single ATM strike for the whole expiry week, anchored to the week's
        Monday open, traded across the full window.
        """
        try:
            nifty_open = self.nifty_service.get_nifty_open(monday)
        except Exception as exc:
            print(f"  [{expiry}] Could not get Nifty open for {monday}: {exc}")
            return None

        strike = NiftyOptionService.atm_strike(nifty_open)

        print(f"  Expiry {expiry}  |  Monday open {nifty_open:.2f}  |  ATM {strike}")

        week_result = BowmanRSIWeeklyExpiryResult(
            expiry_date=expiry,
            atm_strike=strike,
            nifty_open=nifty_open,
        )

        from_dt = datetime(win_start.year, win_start.month, win_start.day, 9, 15, 0)
        to_dt   = datetime(win_end.year,   win_end.month,   win_end.day,   15, 30, 0)

        if self.cache_only:
            missing = False
            for opt_type in ("CE", "PE"):
                cached = self.nifty_service.get_option_candles(
                    strike=strike,
                    expiry_date=expiry,
                    option_type=opt_type,
                    start=from_dt,
                    end=to_dt,
                    interval=self.interval,
                    cache_only=True,
                )
                if not cached:
                    missing = True
                    break
            if missing:
                print(f"    [cache-only] No cached data — skipping expiry {expiry}")
                return None

        for opt_type in ("CE", "PE"):
            try:
                candles = self.nifty_service.get_option_candles(
                    strike=strike,
                    expiry_date=expiry,
                    option_type=opt_type,
                    start=from_dt,
                    end=to_dt,
                    interval=self.interval,
                    cache_only=self.cache_only,
                )
                if self.resample_seconds > 1:
                    candles = resample_candles(candles, self.resample_seconds)
                trades = self._run_symbol(candles, opt_type, strike, expiry)

                if opt_type == "CE":
                    week_result.ce_trades = trades
                else:
                    week_result.pe_trades = trades

                if self.print_resampled:
                    self._print_resampled_with_trades(
                        candles, trades, strike, opt_type, expiry
                    )

                print(f"    {opt_type}: {len(trades)} trades")
            except Exception as exc:
                print(f"    [{opt_type}] Error: {exc}")

        return week_result

    def _run_expiry_per_day(
        self, expiry: date, win_start: date, win_end: date
    ) -> BowmanRSIWeeklyExpiryResult | None:
        """
        Per-day ATM mode: for each trading day in the expiry window, choose a
        fresh ATM strike from that day's Nifty open and trade only that day.
        """
        print(f"  Expiry {expiry}  |  per-day ATM")

        week_result = BowmanRSIWeeklyExpiryResult(
            expiry_date=expiry,
            atm_strike=0,
            nifty_open=0.0,
        )

        days = self.nifty_service.trading_days(
            win_start, win_end, self.market_holidays
        )

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
            to_dt   = datetime(day.year, day.month, day.day, 15, 30, 0)

            print(f"    {day}  |  open {nifty_open:.2f}  |  ATM {strike}")

            if self.cache_only:
                missing = False
                for opt_type in ("CE", "PE"):
                    cached = self.nifty_service.get_option_candles(
                        strike=strike,
                        expiry_date=expiry,
                        option_type=opt_type,
                        start=from_dt,
                        end=to_dt,
                        interval=self.interval,
                        cache_only=True,
                    )
                    if not cached:
                        missing = True
                        break
                if missing:
                    print(f"      [cache-only] No cached data — skipping {day}")
                    continue

            for opt_type in ("CE", "PE"):
                try:
                    candles = self.nifty_service.get_option_candles(
                        strike=strike,
                        expiry_date=expiry,
                        option_type=opt_type,
                        start=from_dt,
                        end=to_dt,
                        interval=self.interval,
                        cache_only=self.cache_only,
                    )
                    if self.resample_seconds > 1:
                        candles = resample_candles(candles, self.resample_seconds)
                    trades = self._run_symbol(candles, opt_type, strike, expiry)

                    if opt_type == "CE":
                        week_result.ce_trades.extend(trades)
                    else:
                        week_result.pe_trades.extend(trades)

                    if self.print_resampled:
                        self._print_resampled_with_trades(
                            candles, trades, strike, opt_type, expiry
                        )

                    print(f"      {opt_type}: {len(trades)} trades")
                except Exception as exc:
                    print(f"      [{opt_type}] Error: {exc}")

        return week_result

    # ── Reporting ─────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(expiry_results: list[BowmanRSIWeeklyExpiryResult]) -> None:
        all_trades: list[BowmanRSITradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[BowmanRSITradeResult]] = defaultdict(list)
        for t in all_trades:
            by_symbol[t.symbol].append(t)

        sep = "─" * 100
        print(f"\n{'='*100}")
        print("  RESULTS BY EXPIRY")
        print(f"{'='*100}")

        for er in expiry_results:
            if not er.all_trades:
                continue
            total_pnl = sum(t.pnl for t in er.all_trades)
            pnl_sign  = "+" if total_pnl >= 0 else ""
            print(f"\n  Expiry {er.expiry_date}  |  ATM {er.atm_strike}"
                  f"  |  Nifty open {er.nifty_open:.2f}"
                  f"  |  Trades {len(er.all_trades)}"
                  f"  |  PnL {pnl_sign}{total_pnl:.2f}")
            print(f"  {sep}")

            header = (
                f"  {'Symbol':<22} {'Entry':>15} {'Exit':>15}"
                f" {'Entry₹':>8} {'Exit₹':>8} {'RSI@In':>7} {'Qty':>5}"
                f" {'PnL':>10} {'Reason':<16}"
            )
            print(header)
            print(f"  {sep}")

            for t in sorted(er.all_trades, key=lambda x: x.entry_time):
                pnl_sign = "+" if t.pnl >= 0 else ""
                print(
                    f"  {t.symbol:<22}"
                    f" {t.entry_time.strftime('%d-%b %H:%M'):>15}"
                    f" {t.exit_time.strftime('%d-%b %H:%M'):>15}"
                    f" {t.entry_price:>8.2f}"
                    f" {t.exit_price:>8.2f}"
                    f" {t.rsi_at_entry:>7.2f}"
                    f" {t.shares:>5}"
                    f" {pnl_sign+f'{t.pnl:.2f}':>10}"
                    f" {t.exit_reason:<16}"
                )

        print(f"\n{'='*100}")
        print("  SYMBOL-WISE METRICS")
        print(f"{'='*100}")

        col_w = {"sym": 22, "trades": 7, "wins": 5, "loss": 6,
                 "wr": 6, "pnl": 10, "avg": 9, "pf": 7,
                 "best": 9, "worst": 9, "dur": 7, "cons": 5}

        hdr = (
            f"  {'Symbol':<{col_w['sym']}} {'Trades':>{col_w['trades']}}"
            f" {'Wins':>{col_w['wins']}} {'Loss':>{col_w['loss']}}"
            f" {'Win%':>{col_w['wr']}} {'Total PnL':>{col_w['pnl']}}"
            f" {'Avg PnL':>{col_w['avg']}} {'PF':>{col_w['pf']}}"
            f" {'Best':>{col_w['best']}} {'Worst':>{col_w['worst']}}"
            f" {'AvgMin':>{col_w['dur']}} {'MaxCL':>{col_w['cons']}}"
        )
        print(hdr)
        print(f"  {'─'*98}")

        overall_pnl = 0.0
        overall_trades = overall_wins = overall_losses = 0

        for sym, trades in sorted(by_symbol.items()):
            m = _compute_metrics(sym, trades)
            overall_pnl    += m.total_pnl
            overall_trades += m.total_trades
            overall_wins   += m.wins
            overall_losses += m.losses

            pnl_s = f"{'+' if m.total_pnl >= 0 else ''}{m.total_pnl:.2f}"
            avg_s = f"{'+' if m.avg_pnl >= 0 else ''}{m.avg_pnl:.2f}"
            pf_s  = f"{m.profit_factor:.2f}" if m.profit_factor != float("inf") else "∞"

            print(
                f"  {m.symbol:<{col_w['sym']}} {m.total_trades:>{col_w['trades']}}"
                f" {m.wins:>{col_w['wins']}} {m.losses:>{col_w['loss']}}"
                f" {m.win_rate:>{col_w['wr']}.1f} {pnl_s:>{col_w['pnl']}}"
                f" {avg_s:>{col_w['avg']}} {pf_s:>{col_w['pf']}}"
                f" {m.best_trade:>{col_w['best']}.2f} {m.worst_trade:>{col_w['worst']}.2f}"
                f" {m.avg_duration_minutes:>{col_w['dur']}.1f} {m.max_consecutive_losses:>{col_w['cons']}}"
            )

        wr_overall = round(overall_wins / overall_trades * 100, 1) if overall_trades else 0.0
        pnl_s      = f"{'+' if overall_pnl >= 0 else ''}{overall_pnl:.2f}"
        print(f"  {'─'*98}")
        print(
            f"  {'OVERALL':<{col_w['sym']}} {overall_trades:>{col_w['trades']}}"
            f" {overall_wins:>{col_w['wins']}} {overall_losses:>{col_w['loss']}}"
            f" {wr_overall:>{col_w['wr']}.1f} {pnl_s:>{col_w['pnl']}}"
        )
        print(f"{'='*100}\n")
