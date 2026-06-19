"""
Time-Weighted Average Price (TWAP) trend-following strategy for Nifty weekly
options (seconds data).

The strategy follows the option premium's trend, confirmed by the session TWAP
(Time-Weighted Average Price) and a trend EMA. Unlike VWAP, the TWAP weights
every bar equally by time (not by traded volume), so it is the cumulative
arithmetic mean of the typical price across the session. This makes it well
suited to thin / illiquid option chains where reported volume is unreliable.

Entry rules (long the option premium):
  CE / PE — buy when the option price makes a fresh cross above the session
            TWAP (the time-weighted trend line), the price is also above its
            trend EMA, and the EMA is rising.

  An optional momentum confirmation requires the entry bar's price to be at
  least `momentum_factor` percent above the previous bar's close. Set
  `momentum_factor` to 0 to disable it.

Exit rules:
  - Scaled take-profit against a percentage target (`target_pct`):
        • sell 25% of the position once price reaches 25% of the target
        • sell another 25% once price reaches 50% of the target
        • sell the remaining position once price reaches the full target
  - Trend reversal (price closes below TWAP or EMA crosses down)   → close all
  - Trailing stop-loss (optional, percentage-based)               → close all
  - Break-even stop: once price moves `breakeven_trigger_pct` percent above
    entry, the stop-loss is moved up to the entry price             → close all
  - Square-off at 15:20 IST                                        → close all
  - Max 5 trades per day per symbol
  - No new entries before 9:30 or after 14:45
"""

from collections import defaultdict
from datetime import date, datetime, time
from math import floor

import numpy as np
import pandas as pd

from models.twap_models import (
    TWAPSymbolMetrics,
    TWAPTradeResult,
    TWAPWeeklyExpiryResult,
)
from services.nifty_option_service import NiftyOptionService


def resample_candles(candles: list[dict], seconds: int) -> list[dict]:
    """
    Resample a list of 1-second OHLC dicts into N-second candles.

    Each input dict must have keys: datetime, open, high, low, close, volume (optional).
    Output dicts have the same schema; datetime is the candle open-time (floor of the
    N-second bucket).

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


def _capital_allocation_pct(price: float) -> float:
    """
    Return the fraction of capital to allocate based on option price.
    Mirrors the allocation tiers used by the other option strategies.
    """
    if price <= 20:
        return 0.30
    elif price <= 60:
        return 1.00
    elif price <= 100:
        return 1.00
    else:
        return 1.00


def compute_twap(candles: list[dict], ema_period: int) -> pd.DataFrame:
    """
    Compute the trend EMA and the session (per-day) TWAP from a list of OHLC
    dicts.

    The TWAP is the cumulative arithmetic mean of the typical price
    (H+L+C)/3 across the trading day, reset at the start of each session. Every
    bar is weighted equally (by time), independent of its traded volume.

    Returns a DataFrame with columns: datetime, ema, twap, close, volume.
    """
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0

    # Trend EMA on the close.
    ema = df["close"].ewm(span=ema_period, adjust=False).mean()
    # Mask the warm-up region so the EMA only fires once it has seen ema_period
    # bars of data.
    warmup = min(ema_period, len(df))
    ema.iloc[: max(warmup - 1, 0)] = np.nan

    # Session TWAP, reset at the start of each trading day. Use the typical
    # price (H+L+C)/3 averaged equally over time (the cumulative mean), so every
    # bar contributes the same weight regardless of volume.
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    day = df["datetime"].dt.date
    cum_tp    = typical.groupby(day).cumsum()
    bar_count = typical.groupby(day).cumcount() + 1
    twap = cum_tp / bar_count

    result = df[["datetime"]].copy()
    result["ema"]    = ema.round(4)
    result["twap"]   = twap.round(4)
    result["close"]  = df["close"]
    result["volume"] = df["volume"]
    return result


def _compute_metrics(symbol: str, trades: list[TWAPTradeResult]) -> TWAPSymbolMetrics:
    if not trades:
        return TWAPSymbolMetrics(
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

    return TWAPSymbolMetrics(
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


class TWAPOptionStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        ema_period: int = 21,
        momentum_factor: float = 0.0,
        target_pct: float = 40.0,
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
        breakeven_enabled: bool = False,
        breakeven_trigger_pct: float = 5.0,
    ):
        self.nifty_service    = nifty_service
        self.capital          = capital
        self.ema_period       = ema_period
        # Optional momentum confirmation. The entry bar's price must be at least
        # `momentum_factor` percent above the previous bar's close for the
        # signal to be taken. 0 disables the momentum filter entirely.
        self.momentum_factor  = momentum_factor
        # Profit target as a percentage above the entry price. The position is
        # scaled out in three legs: 25% of the position at 25% of the target,
        # another 25% at 50% of the target, and the remainder at the full target.
        self.target_pct       = target_pct
        self.start_date       = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date         = datetime.strptime(end_date,   "%d-%b-%Y").date()
        self.interval         = interval
        self.resample_seconds = resample_seconds
        self.print_resampled  = print_resampled
        self.cache_only       = cache_only
        self.market_holidays  = set(market_holidays) if market_holidays else set()
        self.per_day_atm      = per_day_atm
        # Trailing stop-loss. When enabled, the strategy tracks the highest
        # option price (peak) since entry and exits the remaining position if
        # the price falls back by `trailing_stop_pct` percent from that peak.
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_pct     = trailing_stop_pct
        # Break-even stop. Once the option price moves `breakeven_trigger_pct`
        # percent above entry, the stop-loss is moved up to the entry price so
        # the trade can no longer turn into a loss (locks in break-even).
        self.breakeven_enabled     = breakeven_enabled
        self.breakeven_trigger_pct = breakeven_trigger_pct

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _run_symbol(
        self,
        candles: list[dict],
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[TWAPTradeResult]:
        """
        Run the time-weighted-average-price trend-following strategy on a single
        option contract's candle data. Returns a list of completed trades.
        """
        if not candles:
            return []

        indicators = compute_twap(candles, self.ema_period)
        symbol_label = f"NIFTY{strike}{option_type}"
        trades: list[TWAPTradeResult] = []

        daily_trade_count: dict[date, int] = defaultdict(int)

        in_position    = False
        entry_row      = None
        entry_price    = 0.0
        init_shares    = 0      # initial position size
        rem_shares     = 0      # shares still held
        peak_price     = 0.0    # highest price seen since entry (trailing stop)
        breakeven_stop = 0.0
        moved_to_breakeven = False

        # Scaled take-profit levels and state.
        target_price = 0.0
        tp1_price    = 0.0      # 25% of the target distance
        tp2_price    = 0.0      # 50% of the target distance
        hit_tp1      = False
        hit_tp2      = False
        realized_pnl = 0.0      # PnL banked from partial scale-out legs
        weighted_exit_value = 0.0  # Σ qty*price across exit legs (for avg exit)
        exit_qty     = 0        # total shares exited so far
        leg_descriptions: list[str] = []

        prev_ema   = None
        prev_twap  = None
        prev_close = None

        def _record_trade(dt: datetime, row, final_reason: str) -> None:
            avg_exit = weighted_exit_value / exit_qty if exit_qty else entry_price
            duration = int((dt - entry_row["datetime"]).total_seconds() / 60)
            trades.append(TWAPTradeResult(
                symbol=symbol_label,
                option_type=option_type,
                strike=strike,
                expiry_date=expiry_date,
                entry_time=entry_row["datetime"],
                exit_time=dt,
                entry_price=entry_price,
                exit_price=round(avg_exit, 2),
                shares=init_shares,
                pnl=round(realized_pnl, 2),
                exit_reason=final_reason,
                twap_at_entry=entry_row["twap"],
                ema_at_entry=entry_row["ema"],
                twap_at_exit=row["twap"],
                ema_at_exit=row["ema"],
                target_price=round(target_price, 2),
                duration_minutes=duration,
                scale_out_legs=" | ".join(leg_descriptions),
            ))

        for _, row in indicators.iterrows():
            dt: datetime  = row["datetime"]
            t: time       = dt.time()
            today: date   = dt.date()

            ema   = row["ema"]
            twap  = row["twap"]
            price = row["close"]

            if pd.isna(ema) or pd.isna(twap):
                prev_ema   = ema
                prev_twap  = twap
                prev_close = price
                continue

            # ── Exit logic (checked every bar while in position) ──────────
            if in_position:
                if price > peak_price:
                    peak_price = price

                # Break-even: once price moves the trigger above entry, ratchet
                # the stop-loss up to the entry price.
                if (
                    self.breakeven_enabled
                    and not moved_to_breakeven
                    and price >= entry_price * (1 + self.breakeven_trigger_pct / 100.0)
                ):
                    breakeven_stop = entry_price
                    moved_to_breakeven = True

                # Protective / reversal exits close the WHOLE remaining position.
                full_exit_reason = None
                if t >= _SQUARE_OFF:
                    full_exit_reason = "SQUARE_OFF"
                elif moved_to_breakeven and price <= breakeven_stop:
                    full_exit_reason = "BREAKEVEN"
                elif (
                    self.trailing_stop_enabled
                    and self.trailing_stop_pct > 0
                    and peak_price > 0
                    and price <= peak_price * (1 - self.trailing_stop_pct / 100.0)
                ):
                    full_exit_reason = "TRAILING_STOP"
                elif prev_ema is not None and not pd.isna(prev_ema):
                    # Trend reversal: price closes below TWAP, or EMA crosses down.
                    if price < twap or (prev_close >= prev_ema and price < ema):
                        full_exit_reason = "TREND_REVERSAL"

                if full_exit_reason:
                    realized_pnl += rem_shares * (price - entry_price)
                    weighted_exit_value += rem_shares * price
                    exit_qty += rem_shares
                    leg_descriptions.append(
                        f"{full_exit_reason}@{price:.2f}x{rem_shares}"
                    )
                    rem_shares = 0
                    _record_trade(dt, row, full_exit_reason)
                    in_position = False
                    entry_row   = None
                    prev_ema    = ema
                    prev_twap   = twap
                    prev_close  = price
                    continue

                # ── Scaled take-profit ────────────────────────────────────
                # Full target reached: sell everything that remains.
                if price >= target_price and rem_shares > 0:
                    realized_pnl += rem_shares * (price - entry_price)
                    weighted_exit_value += rem_shares * price
                    exit_qty += rem_shares
                    leg_descriptions.append(f"TARGET@{price:.2f}x{rem_shares}")
                    rem_shares = 0
                    _record_trade(dt, row, "TARGET")
                    in_position = False
                    entry_row   = None
                    prev_ema    = ema
                    prev_twap   = twap
                    prev_close  = price
                    continue

                # 50% of the target: sell 25% of the initial position.
                if not hit_tp2 and price >= tp2_price and rem_shares > 0:
                    qty = min(floor(init_shares * 0.25), rem_shares)
                    if qty <= 0:
                        qty = rem_shares
                    realized_pnl += qty * (price - entry_price)
                    weighted_exit_value += qty * price
                    exit_qty += qty
                    rem_shares -= qty
                    hit_tp1 = True
                    hit_tp2 = True
                    leg_descriptions.append(f"TP2@{price:.2f}x{qty}")

                # 25% of the target: sell 25% of the initial position.
                elif not hit_tp1 and price >= tp1_price and rem_shares > 0:
                    qty = min(floor(init_shares * 0.25), rem_shares)
                    if qty <= 0:
                        qty = rem_shares
                    realized_pnl += qty * (price - entry_price)
                    weighted_exit_value += qty * price
                    exit_qty += qty
                    rem_shares -= qty
                    hit_tp1 = True
                    leg_descriptions.append(f"TP1@{price:.2f}x{qty}")

            # ── Entry logic ───────────────────────────────────────────────
            if (
                not in_position
                and prev_ema is not None
                and not pd.isna(prev_ema)
                and prev_close is not None
                and _ENTRY_START <= t <= _ENTRY_CUTOFF
                and daily_trade_count[today] < _MAX_TRADES_PER_DAY
            ):
                # Optional momentum confirmation: the entry bar must be at least
                # `momentum_factor` percent above the previous bar's close.
                if self.momentum_factor <= 0:
                    momentum_ok = True
                elif prev_close > 0:
                    momentum_ok = price >= prev_close * (1 + self.momentum_factor / 100.0)
                else:
                    momentum_ok = False

                # Trigger: price makes a fresh cross above the session TWAP (the
                # time-weighted trend line). Confirmation: the move is in the
                # direction of the trend EMA — price is above the EMA and the
                # EMA is rising. Both legs (CE & PE) are bought on a rising
                # premium that breaks above its own time-weighted average.
                cross_twap = (
                    prev_twap is not None
                    and not pd.isna(prev_twap)
                    and prev_close <= prev_twap
                    and price > twap
                )
                above_ema  = price > ema
                ema_rising = price > prev_ema
                signal = cross_twap and above_ema and ema_rising

                if signal and momentum_ok and price > 0:
                    entry_price       = price
                    alloc_pct         = _capital_allocation_pct(entry_price)
                    allocated_capital = self.capital * alloc_pct
                    init_shares       = max(floor(allocated_capital / entry_price), 1)
                    rem_shares        = init_shares
                    in_position       = True
                    entry_row         = row
                    peak_price        = entry_price
                    breakeven_stop    = 0.0
                    moved_to_breakeven = False

                    target_price = entry_price * (1 + self.target_pct / 100.0)
                    tp1_price    = entry_price * (1 + 0.25 * self.target_pct / 100.0)
                    tp2_price    = entry_price * (1 + 0.50 * self.target_pct / 100.0)
                    hit_tp1 = hit_tp2 = False
                    realized_pnl = 0.0
                    weighted_exit_value = 0.0
                    exit_qty = 0
                    leg_descriptions = []

                    daily_trade_count[today] += 1

            prev_ema   = ema
            prev_twap  = twap
            prev_close = price

        # Force-close any open position at end of data
        if in_position and entry_row is not None and rem_shares > 0:
            last  = indicators.iloc[-1]
            price = last["close"]
            dt    = last["datetime"]
            realized_pnl += rem_shares * (price - entry_price)
            weighted_exit_value += rem_shares * price
            exit_qty += rem_shares
            leg_descriptions.append(f"SQUARE_OFF@{price:.2f}x{rem_shares}")
            rem_shares = 0
            _record_trade(dt, last, "SQUARE_OFF")

        return trades

    # ── Debug / inspection helpers ────────────────────────────────────────────

    def _print_resampled_with_trades(
        self,
        candles: list[dict],
        trades: list[TWAPTradeResult],
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

        indicators = compute_twap(candles, self.ema_period)
        merged = df.merge(
            indicators[["datetime", "ema", "twap"]],
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
                f"  ({t.exit_reason})  [{t.scale_out_legs}]"
            )

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[TWAPWeeklyExpiryResult]:
        expiry_results: list[TWAPWeeklyExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        effective_tf = (
            f"{self.resample_seconds}s" if self.resample_seconds > 1 else self.interval
        )
        print(f"\n{'='*70}")
        print(f"  NIFTY TIME-WEIGHTED AVERAGE PRICE — WEEKLY EXPIRY BACKTEST")
        print(f"  Period    : {self.start_date}  →  {self.end_date}")
        print(f"  Fetch TF  : {self.interval}  |  Effective TF: {effective_tf}")
        print(f"  Capital   : ₹{self.capital:,.0f}  |  EMA period: {self.ema_period}"
              f"  |  Target%: {self.target_pct}  |  Momentum%: {self.momentum_factor}")
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
    ) -> TWAPWeeklyExpiryResult | None:
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

        week_result = TWAPWeeklyExpiryResult(
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
    ) -> TWAPWeeklyExpiryResult | None:
        """
        Per-day ATM mode: for each trading day in the expiry window, choose a
        fresh ATM strike from that day's Nifty open and trade only that day.
        """
        print(f"  Expiry {expiry}  |  per-day ATM")

        week_result = TWAPWeeklyExpiryResult(
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
    def print_report(expiry_results: list[TWAPWeeklyExpiryResult]) -> None:
        all_trades: list[TWAPTradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[TWAPTradeResult]] = defaultdict(list)
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
                f" {'Entry₹':>8} {'AvgExit₹':>9} {'Tgt₹':>8} {'Qty':>5}"
                f" {'PnL':>10} {'Reason':<15}"
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
                    f" {t.exit_price:>9.2f}"
                    f" {t.target_price:>8.2f}"
                    f" {t.shares:>5}"
                    f" {pnl_sign+f'{t.pnl:.2f}':>10}"
                    f" {t.exit_reason:<15}"
                )
                if t.scale_out_legs:
                    print(f"      legs: {t.scale_out_legs}")

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
