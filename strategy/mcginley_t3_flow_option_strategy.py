"""
McGinley T3 Flow campaign strategy for Nifty weekly options (1-second data).

A trend-following strategy built around an adaptive signal trail and a
campaign-style trade management model. It does not try to predict tops or
bottoms; it waits for the selected flow engine to shift direction, opens a
long campaign on the option premium, and then manages the trade against a
TP1 / TP2 / TP3 target structure.

Signal engine (the adaptive basis):
  • MCGINLEY — McGinley Dynamic only (adapts to changes in price speed)
  • T3       — Tillson T3 smoothing only (a smoother trend basis)
  • BLEND    — average of the McGinley Dynamic and T3 curves (default)

After the engine basis is computed, an ATR signal trail is built around it
(SuperTrend-style): the trail updates with volatility distance and locks in a
directional path. A confirmed transition of this trail creates the signal.
Each option leg (CE & PE) is traded long on its OWN price's flow signal.

Entry:
  Long campaign when the ATR signal trail confirms an UPWARD transition
  (a "main BUY"). Signals are confirmed on closed bars. Continuation refreshes
  in the same direction do not open a new campaign while one is already open.

Campaign target model (TP1 / TP2 / TP3 projected from a base distance):
  • FLOW_TRAIL   — base distance = entry price − active signal trail, so the
                   targets expand and contract with the trend structure (default)
  • ATR_BASELINE — base distance = ATR at entry (a traditional volatility model)

Take-profit exit modes:
  • TP1 / TP2 / TP3 — exit the FULL position at the selected single target
  • SCALE_OUT       — reduce the position across TP1, TP2 and TP3 using the
                      tp1_pct / tp2_pct percentage settings (remainder at TP3)

Other exits:
  - Flow flip: the signal trail confirms a downward transition  → close all
  - Optional stop-loss at the campaign SL level (below entry)    → close all
  - Trailing stop-loss (optional, percentage-based)             → close all
  - Square-off at 15:20 IST                                     → close all
  - Max 5 trades per day per symbol
  - No new entries before 9:30 or after 14:45
"""

from collections import defaultdict
from datetime import date, datetime, time
from math import floor

import numpy as np
import pandas as pd

from models.mcginley_t3_flow_models import (
    FlowPartialExit,
    FlowSymbolMetrics,
    FlowWeeklyExpiryResult,
    McGinleyT3FlowTradeResult,
)
from services.nifty_option_service import NiftyOptionService


def resample_candles(candles: list[dict], seconds: int) -> list[dict]:
    """
    Resample a list of 1-second OHLC dicts into N-second candles.
    seconds <= 1 returns the original data unchanged.
    """
    if seconds <= 1 or not candles:
        return candles

    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    df["_bucket"] = df["datetime"].apply(lambda ts: ts.floor(f"{seconds}s"))

    agg: dict = {"open": "first", "high": "max", "low": "min", "close": "last"}
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

# McGinley Dynamic smoothing constant. The denominator k·N·(close/md)^4 controls
# how quickly the dynamic line adapts; 0.6 is the conventional value.
_MCGINLEY_K = 0.6


def _capital_allocation_pct(price: float) -> float:
    """Fraction of capital to allocate based on the option price."""
    if price <= 20:
        return 0.30
    elif price <= 60:
        return 1.00
    elif price <= 100:
        return 1.00
    else:
        return 1.00


def _wilder_ewm(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing: EWM with alpha = 1/period, adjust=False."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _mcginley_dynamic(close: pd.Series, period: int) -> np.ndarray:
    """
    McGinley Dynamic line. Adapts to changes in price speed more smoothly than a
    standard moving average:

        md[i] = md[i-1] + (close[i] - md[i-1]) / (k · N · (close[i]/md[i-1])^4)

    Seeded from the first close.
    """
    cl = close.to_numpy(dtype=float)
    n = len(cl)
    md = np.full(n, np.nan)
    if n == 0:
        return md

    md[0] = cl[0]
    for i in range(1, n):
        prev = md[i - 1]
        if prev <= 0 or not np.isfinite(prev):
            md[i] = cl[i]
            continue
        ratio = cl[i] / prev
        denom = _MCGINLEY_K * period * (ratio ** 4)
        if denom == 0 or not np.isfinite(denom):
            md[i] = prev
        else:
            md[i] = prev + (cl[i] - prev) / denom
    return md


def _t3(close: pd.Series, period: int, vfactor: float) -> np.ndarray:
    """
    Tillson T3 smoothing — a chain of six EMAs blended with a volume factor `v`
    for a smooth, low-lag trend basis.
    """
    def ema(series: pd.Series) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    e1 = ema(close)
    e2 = ema(e1)
    e3 = ema(e2)
    e4 = ema(e3)
    e5 = ema(e4)
    e6 = ema(e5)

    v  = vfactor
    c1 = -v ** 3
    c2 = 3 * v ** 2 + 3 * v ** 3
    c3 = -6 * v ** 2 - 3 * v - 3 * v ** 3
    c4 = 1 + 3 * v + v ** 3 + 3 * v ** 2

    t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    return t3.to_numpy(dtype=float)


def compute_mcginley_t3_flow(
    candles: list[dict],
    engine_mode: str = "BLEND",
    mcginley_period: int = 14,
    t3_period: int = 8,
    t3_volume_factor: float = 0.7,
    atr_period: int = 10,
    atr_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Compute the McGinley T3 Flow signal engine and its ATR signal trail.

    Returns a DataFrame with columns:
        datetime, close, atr, engine, trail, trend
    where `engine` is the adaptive basis (McGinley / T3 / blend), `trail` is the
    locked-in ATR signal trail and `trend` is +1 (bullish / price above trail)
    or -1 (bearish / price below trail), with 0 during the warm-up region.
    """
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = _wilder_ewm(tr, atr_period)

    mode = engine_mode.upper()
    md = _mcginley_dynamic(close, mcginley_period)
    t3 = _t3(close, t3_period, t3_volume_factor)
    if mode == "MCGINLEY":
        engine = md
    elif mode == "T3":
        engine = t3
    else:  # BLEND — balanced engine between responsiveness and smoothness
        engine = (md + t3) / 2.0

    # ATR signal trail built around the adaptive engine basis (SuperTrend-style
    # band locking applied to the engine instead of HL2).
    upper_basic = engine + atr_multiplier * atr.to_numpy()
    lower_basic = engine - atr_multiplier * atr.to_numpy()

    n = len(df)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    trail       = np.full(n, np.nan)
    trend       = np.full(n, 0, dtype=int)

    ub = upper_basic
    lb = lower_basic
    cl = close.to_numpy()

    for i in range(n):
        if i == 0 or not np.isfinite(ub[i]) or not np.isfinite(lb[i]):
            final_upper[i] = ub[i]
            final_lower[i] = lb[i]
            trail[i]       = ub[i]
            trend[i]       = -1
            continue

        final_upper[i] = (
            ub[i] if (ub[i] < final_upper[i - 1] or cl[i - 1] > final_upper[i - 1])
            else final_upper[i - 1]
        )
        final_lower[i] = (
            lb[i] if (lb[i] > final_lower[i - 1] or cl[i - 1] < final_lower[i - 1])
            else final_lower[i - 1]
        )

        if trail[i - 1] == final_upper[i - 1]:
            # was riding the upper band (bearish)
            if cl[i] <= final_upper[i]:
                trail[i]  = final_upper[i]
                trend[i]  = -1
            else:
                trail[i]  = final_lower[i]
                trend[i]  = 1
        else:
            # was riding the lower band (bullish)
            if cl[i] >= final_lower[i]:
                trail[i]  = final_lower[i]
                trend[i]  = 1
            else:
                trail[i]  = final_upper[i]
                trend[i]  = -1

    # Warm-up masking: the ATR and T3 chain need ~max(period) bars to settle.
    warmup = min(max(atr_period, t3_period, mcginley_period), n)
    trend[:warmup] = 0

    result = df[["datetime"]].copy()
    result["close"]  = close
    result["atr"]    = pd.Series(atr).round(4)
    result["engine"] = pd.Series(engine).round(4)
    result["trail"]  = pd.Series(trail).round(4)
    result["trend"]  = trend
    return result


def _compute_metrics(
    symbol: str, trades: list[McGinleyT3FlowTradeResult]
) -> FlowSymbolMetrics:
    if not trades:
        return FlowSymbolMetrics(
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

    return FlowSymbolMetrics(
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


class McGinleyT3FlowOptionStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        engine_mode: str = "BLEND",
        mcginley_period: int = 14,
        t3_period: int = 8,
        t3_volume_factor: float = 0.7,
        atr_period: int = 10,
        atr_multiplier: float = 2.0,
        target_mode: str = "FLOW_TRAIL",
        tp1_mult: float = 1.0,
        tp2_mult: float = 2.0,
        tp3_mult: float = 3.0,
        tp_exit_mode: str = "SCALE_OUT",
        tp1_pct: float = 0.40,
        tp2_pct: float = 0.30,
        sl_enabled: bool = True,
        sl_mult: float = 1.0,
        start_date: str = "",
        end_date: str = "",
        interval: str = "1minute",
        resample_seconds: int = 1,
        print_resampled: bool = False,
        cache_only: bool = False,
        market_holidays: set[date] | None = None,
        per_day_atm: bool = False,
        trailing_stop_enabled: bool = False,
        trailing_stop_pct: float = 0.0,
    ):
        self.nifty_service    = nifty_service
        self.capital          = capital
        # Signal engine: "MCGINLEY", "T3" or "BLEND".
        self.engine_mode      = engine_mode.upper()
        self.mcginley_period  = mcginley_period
        self.t3_period        = t3_period
        self.t3_volume_factor = t3_volume_factor
        # ATR signal-trail parameters.
        self.atr_period       = atr_period
        self.atr_multiplier   = atr_multiplier
        # Campaign target model: "FLOW_TRAIL" projects targets from the distance
        # between price and the active signal trail; "ATR_BASELINE" projects them
        # from the ATR at entry.
        self.target_mode      = target_mode.upper()
        self.tp1_mult         = tp1_mult
        self.tp2_mult         = tp2_mult
        self.tp3_mult         = tp3_mult
        # Take-profit behaviour: "TP1" / "TP2" / "TP3" exit the full position at
        # the chosen target; "SCALE_OUT" reduces across TP1/TP2/TP3.
        self.tp_exit_mode     = tp_exit_mode.upper()
        self.tp1_pct          = tp1_pct
        self.tp2_pct          = tp2_pct
        # Optional campaign stop-loss, sl_mult × base distance below entry.
        self.sl_enabled       = sl_enabled
        self.sl_mult          = sl_mult
        self.start_date       = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date         = datetime.strptime(end_date,   "%d-%b-%Y").date()
        self.interval         = interval
        self.resample_seconds = resample_seconds
        self.print_resampled  = print_resampled
        self.cache_only       = cache_only
        self.market_holidays  = set(market_holidays) if market_holidays else set()
        self.per_day_atm      = per_day_atm
        # Trailing stop-loss off the running peak.
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_pct     = trailing_stop_pct

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _run_symbol(
        self,
        candles: list[dict],
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[McGinleyT3FlowTradeResult]:
        """Run the McGinley T3 Flow campaign on a single option contract."""
        if not candles:
            return []

        ind = compute_mcginley_t3_flow(
            candles,
            engine_mode=self.engine_mode,
            mcginley_period=self.mcginley_period,
            t3_period=self.t3_period,
            t3_volume_factor=self.t3_volume_factor,
            atr_period=self.atr_period,
            atr_multiplier=self.atr_multiplier,
        )
        symbol_label = f"NIFTY{strike}{option_type}"
        trades: list[McGinleyT3FlowTradeResult] = []

        daily_trade_count: dict[date, int] = defaultdict(int)

        in_position   = False
        entry_row     = None
        entry_price   = 0.0
        total_shares  = 0
        remaining     = 0
        peak_price    = 0.0
        tp1_price = tp2_price = tp3_price = sl_price = 0.0
        tp1_done  = tp2_done  = False
        partials: list[FlowPartialExit] = []

        prev_trend = 0

        def close_position(dt, last_trail):
            """Finalise the trade record from accumulated partial legs."""
            nonlocal in_position, entry_row
            sold = sum(p.shares for p in partials)
            total_pnl = round(sum(p.pnl for p in partials), 2)
            wavg_exit = (
                round(sum(p.price * p.shares for p in partials) / sold, 2)
                if sold else entry_price
            )
            duration = int((dt - entry_row["datetime"]).total_seconds() / 60)
            trades.append(McGinleyT3FlowTradeResult(
                symbol=symbol_label,
                option_type=option_type,
                strike=strike,
                expiry_date=expiry_date,
                entry_time=entry_row["datetime"],
                exit_time=partials[-1].time if partials else dt,
                entry_price=entry_price,
                exit_price=wavg_exit,
                shares=total_shares,
                pnl=total_pnl,
                exit_reason=partials[-1].reason if partials else "SQUARE_OFF",
                engine_at_entry=entry_row["engine"],
                trail_at_entry=entry_row["trail"],
                atr_at_entry=entry_row["atr"],
                tp1_price=round(tp1_price, 2),
                tp2_price=round(tp2_price, 2),
                tp3_price=round(tp3_price, 2),
                sl_price=round(sl_price, 2),
                duration_minutes=duration,
                partials=list(partials),
            ))
            in_position = False
            entry_row   = None

        for _, row in ind.iterrows():
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            today: date  = dt.date()

            trend = int(row["trend"])
            price = row["close"]
            trail = row["trail"]

            if trend == 0:
                prev_trend = trend
                continue

            # ── Exit / campaign management (every bar while in position) ───
            if in_position:
                if price > peak_price:
                    peak_price = price

                def book(shares, reason):
                    nonlocal remaining
                    shares = min(shares, remaining)
                    if shares <= 0:
                        return
                    pnl = round(shares * (price - entry_price), 2)
                    partials.append(FlowPartialExit(dt, price, shares, pnl, reason))
                    remaining -= shares

                # 1) Square-off at 15:20 — dump everything.
                if t >= _SQUARE_OFF:
                    book(remaining, "SQUARE_OFF")
                    close_position(dt, trail)
                    prev_trend = trend
                    continue

                # 2) Take-profit targets (single-target or scale-out).
                if self.tp_exit_mode == "TP1":
                    if price >= tp1_price:
                        book(remaining, "TP1")
                        close_position(dt, trail)
                        prev_trend = trend
                        continue
                elif self.tp_exit_mode == "TP2":
                    if price >= tp2_price:
                        book(remaining, "TP2")
                        close_position(dt, trail)
                        prev_trend = trend
                        continue
                elif self.tp_exit_mode == "TP3":
                    if price >= tp3_price:
                        book(remaining, "TP3")
                        close_position(dt, trail)
                        prev_trend = trend
                        continue
                else:  # SCALE_OUT across TP1 / TP2 / TP3
                    if price >= tp3_price:
                        book(remaining, "TP3")
                        close_position(dt, trail)
                        prev_trend = trend
                        continue
                    if not tp1_done and price >= tp1_price:
                        book(int(round(total_shares * self.tp1_pct)), "TP1")
                        tp1_done = True
                    if not tp2_done and price >= tp2_price:
                        book(int(round(total_shares * self.tp2_pct)), "TP2")
                        tp2_done = True

                # 3) Optional campaign stop-loss.
                if self.sl_enabled and sl_price > 0 and price <= sl_price:
                    book(remaining, "STOP_LOSS")
                    close_position(dt, trail)
                    prev_trend = trend
                    continue

                # 4) Trailing stop off the peak.
                if (
                    self.trailing_stop_enabled
                    and self.trailing_stop_pct > 0
                    and peak_price > 0
                    and price <= peak_price * (1 - self.trailing_stop_pct / 100.0)
                ):
                    book(remaining, "TRAILING_STOP")
                    close_position(dt, trail)
                    prev_trend = trend
                    continue

                # 5) Flow flip — signal trail confirms a downward transition.
                if prev_trend == 1 and trend == -1:
                    book(remaining, "FLOW_FLIP")
                    close_position(dt, trail)
                    prev_trend = trend
                    continue

                if remaining <= 0:
                    close_position(dt, trail)

            # ── Entry logic — main upward flow transition ─────────────────
            if (
                not in_position
                and prev_trend != 0
                and _ENTRY_START <= t <= _ENTRY_CUTOFF
                and daily_trade_count[today] < _MAX_TRADES_PER_DAY
            ):
                # Confirmed upward transition of the ATR signal trail.
                signal = prev_trend == -1 and trend == 1
                if signal and price > 0:
                    entry_price  = price
                    # Base distance for the campaign target projection.
                    if self.target_mode == "ATR_BASELINE":
                        base_dist = float(row["atr"])
                    else:  # FLOW_TRAIL — distance from price to the active trail
                        base_dist = max(price - float(trail), 0.0)
                    if base_dist <= 0 or not np.isfinite(base_dist):
                        base_dist = float(row["atr"])

                    tp1_price = entry_price + self.tp1_mult * base_dist
                    tp2_price = entry_price + self.tp2_mult * base_dist
                    tp3_price = entry_price + self.tp3_mult * base_dist
                    sl_price  = (
                        entry_price - self.sl_mult * base_dist
                        if self.sl_enabled else 0.0
                    )

                    alloc_pct         = _capital_allocation_pct(entry_price)
                    allocated_capital = self.capital * alloc_pct
                    total_shares      = max(floor(allocated_capital / entry_price), 1)
                    remaining         = total_shares
                    in_position       = True
                    entry_row         = row
                    peak_price        = entry_price
                    tp1_done          = False
                    tp2_done          = False
                    partials          = []
                    daily_trade_count[today] += 1

            prev_trend = trend

        # Force-close any open position at end of data.
        if in_position and entry_row is not None:
            last  = ind.iloc[-1]
            price = last["close"]
            dt    = last["datetime"]
            if remaining > 0:
                pnl = round(remaining * (price - entry_price), 2)
                partials.append(FlowPartialExit(dt, price, remaining, pnl, "SQUARE_OFF"))
                remaining = 0
            close_position(dt, last["trail"])

        return trades

    # ── Debug / inspection helpers ────────────────────────────────────────────

    def _print_resampled_with_trades(
        self,
        candles: list[dict],
        trades: list[McGinleyT3FlowTradeResult],
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

        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        ind = compute_mcginley_t3_flow(
            candles,
            engine_mode=self.engine_mode,
            mcginley_period=self.mcginley_period,
            t3_period=self.t3_period,
            t3_volume_factor=self.t3_volume_factor,
            atr_period=self.atr_period,
            atr_multiplier=self.atr_multiplier,
        )
        merged = df.merge(
            ind[["datetime", "atr", "engine", "trail", "trend"]],
            on="datetime",
            how="left",
        )

        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", None,
        ):
            print(merged.to_string(index=False))

        print(f"\n  Trades for {symbol_label}: {len(trades)}")
        for t in sorted(trades, key=lambda x: x.entry_time):
            pnl_sign = "+" if t.pnl >= 0 else ""
            print(
                f"    {t.entry_time.strftime('%d-%b %H:%M:%S')} → "
                f"{t.exit_time.strftime('%d-%b %H:%M:%S')}"
                f"  entry {t.entry_price:.2f}  exit {t.exit_price:.2f}"
                f"  qty {t.shares}  pnl {pnl_sign}{t.pnl:.2f}"
                f"  ({t.exit_reason})"
                f"  TP {t.tp1_price:.2f}/{t.tp2_price:.2f}/{t.tp3_price:.2f}"
            )
            for p in t.partials:
                ps = "+" if p.pnl >= 0 else ""
                print(
                    f"        · {p.time.strftime('%H:%M:%S')}  "
                    f"sell {p.shares} @ {p.price:.2f}  pnl {ps}{p.pnl:.2f}  ({p.reason})"
                )

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[FlowWeeklyExpiryResult]:
        expiry_results: list[FlowWeeklyExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        effective_tf = (
            f"{self.resample_seconds}s" if self.resample_seconds > 1 else self.interval
        )
        print(f"\n{'='*70}")
        print(f"  NIFTY McGINLEY T3 FLOW — WEEKLY EXPIRY BACKTEST")
        print(f"  Period    : {self.start_date}  →  {self.end_date}")
        print(f"  Fetch TF  : {self.interval}  |  Effective TF: {effective_tf}")
        print(f"  Capital   : ₹{self.capital:,.0f}  |  Engine: {self.engine_mode}"
              f"  |  ATR {self.atr_period}×{self.atr_multiplier}")
        print(f"  Targets   : {self.target_mode}  |  Exit: {self.tp_exit_mode}"
              f"  |  TP×{self.tp1_mult}/{self.tp2_mult}/{self.tp3_mult}")
        print(f"  Expiries found: {len(wednesdays)}")
        print(f"{'='*70}\n")

        for tuesday in wednesdays:
            monday     = NiftyOptionService.monday_of_week(tuesday)
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
    ) -> FlowWeeklyExpiryResult | None:
        try:
            nifty_open = self.nifty_service.get_nifty_open(monday)
        except Exception as exc:
            print(f"  [{expiry}] Could not get Nifty open for {monday}: {exc}")
            return None

        strike = NiftyOptionService.atm_strike(nifty_open)
        print(f"  Expiry {expiry}  |  Monday open {nifty_open:.2f}  |  ATM {strike}")

        week_result = FlowWeeklyExpiryResult(
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
                    strike=strike, expiry_date=expiry, option_type=opt_type,
                    start=from_dt, end=to_dt, interval=self.interval, cache_only=True,
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
                    strike=strike, expiry_date=expiry, option_type=opt_type,
                    start=from_dt, end=to_dt, interval=self.interval,
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
    ) -> FlowWeeklyExpiryResult | None:
        print(f"  Expiry {expiry}  |  per-day ATM")

        week_result = FlowWeeklyExpiryResult(
            expiry_date=expiry, atm_strike=0, nifty_open=0.0,
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
                        strike=strike, expiry_date=expiry, option_type=opt_type,
                        start=from_dt, end=to_dt, interval=self.interval, cache_only=True,
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
                        strike=strike, expiry_date=expiry, option_type=opt_type,
                        start=from_dt, end=to_dt, interval=self.interval,
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
    def print_report(expiry_results: list[FlowWeeklyExpiryResult]) -> None:
        all_trades: list[McGinleyT3FlowTradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[McGinleyT3FlowTradeResult]] = defaultdict(list)
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
            pnl_sign  = "+" if total_pnl >= 0 else ""
            print(f"\n  Expiry {er.expiry_date}  |  ATM {er.atm_strike}"
                  f"  |  Nifty open {er.nifty_open:.2f}"
                  f"  |  Trades {len(er.all_trades)}"
                  f"  |  PnL {pnl_sign}{total_pnl:.2f}")
            print(f"  {sep}")

            header = (
                f"  {'Symbol':<22} {'Entry':>16} {'Exit':>16}"
                f" {'Entry₹':>8} {'Exit₹':>8} {'Qty':>5}"
                f" {'PnL':>10} {'Reason':<16} {'Legs':>4}"
            )
            print(header)
            print(f"  {sep}")

            for t in sorted(er.all_trades, key=lambda x: x.entry_time):
                pnl_sign = "+" if t.pnl >= 0 else ""
                print(
                    f"  {t.symbol:<22}"
                    f" {t.entry_time.strftime('%d-%b %H:%M'):>16}"
                    f" {t.exit_time.strftime('%d-%b %H:%M'):>16}"
                    f" {t.entry_price:>8.2f}"
                    f" {t.exit_price:>8.2f}"
                    f" {t.shares:>5}"
                    f" {pnl_sign+f'{t.pnl:.2f}':>10}"
                    f" {t.exit_reason:<16}"
                    f" {len(t.partials):>4}"
                )

        # Per-symbol metrics
        print(f"\n{'='*96}")
        print("  SYMBOL-WISE METRICS")
        print(f"{'='*96}")

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
        print(f"  {'─'*94}")

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
        print(f"  {'─'*94}")
        print(
            f"  {'OVERALL':<{col_w['sym']}} {overall_trades:>{col_w['trades']}}"
            f" {overall_wins:>{col_w['wins']}} {overall_losses:>{col_w['loss']}}"
            f" {wr_overall:>{col_w['wr']}.1f} {pnl_s:>{col_w['pnl']}}"
        )
        print(f"{'='*96}\n")
