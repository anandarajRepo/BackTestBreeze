"""
Supertrend strategy for Nifty weekly options (1-second data).

Entry rules:
  CE — buy when the CE contract's Supertrend flips from bearish to bullish
  PE — buy when the PE contract's Supertrend flips from bearish to bullish
       (each option leg is traded long on its OWN price's Supertrend signal)

Exit / position management:
  - Scaled take-profit against a percentage TARGET:
        • sell 25% of the position once price reaches 25% of the target move
        • sell a further 25% once price reaches 50% of the target move
        • sell ALL remaining shares once price reaches the full target
  - BREAKEVEN_TRIGGER: once price gains BREAKEVEN_TRIGGER_PCT, the stop is
    raised to the entry price so the trade can no longer lose money.
  - TRAILING_STOP: exit the remaining shares if price falls TRAILING_STOP_PCT
    below the highest price reached since entry (ratchets up, never down).
  - Supertrend flip back to bearish closes the remaining position.
  - Square-off at 15:20 IST.
  - Max 5 trades per day per symbol.
  - No new entries before 9:30 or after 14:45.
"""

from collections import defaultdict
from datetime import date, datetime, time
from math import floor

import numpy as np
import pandas as pd

from models.supertrend_models import (
    PartialExit,
    SuperTrendTradeResult,
    SymbolMetrics,
    WeeklyExpiryResult,
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


def compute_supertrend(
    candles: list[dict], period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Compute the Supertrend indicator from a list of OHLC dicts.

    Returns a DataFrame with columns: datetime, close, atr, supertrend and
    trend (+1 = bullish / price above supertrend, -1 = bearish).
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
    atr = _wilder_ewm(tr, period)

    hl2 = (high + low) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    n = len(df)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    supertrend  = np.full(n, np.nan)
    trend       = np.full(n, 0, dtype=int)

    ub = upper_basic.to_numpy()
    lb = lower_basic.to_numpy()
    cl = close.to_numpy()

    for i in range(n):
        if i == 0:
            final_upper[i] = ub[i]
            final_lower[i] = lb[i]
            supertrend[i]  = ub[i]
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

        if supertrend[i - 1] == final_upper[i - 1]:
            # was in the upper band (bearish)
            if cl[i] <= final_upper[i]:
                supertrend[i] = final_upper[i]
                trend[i]      = -1
            else:
                supertrend[i] = final_lower[i]
                trend[i]      = 1
        else:
            # was in the lower band (bullish)
            if cl[i] >= final_lower[i]:
                supertrend[i] = final_lower[i]
                trend[i]      = 1
            else:
                supertrend[i] = final_upper[i]
                trend[i]      = -1

    # Warm-up masking: the ATR needs ~`period` bars to be meaningful.
    warmup = min(period, n)
    trend[:warmup] = 0

    result = df[["datetime"]].copy()
    result["close"]      = close
    result["atr"]        = pd.Series(atr).round(4)
    result["supertrend"] = pd.Series(supertrend).round(4)
    result["trend"]      = trend
    return result


def _compute_metrics(symbol: str, trades: list[SuperTrendTradeResult]) -> SymbolMetrics:
    if not trades:
        return SymbolMetrics(
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

    return SymbolMetrics(
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


class SuperTrendOptionStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        st_period: int = 10,
        st_multiplier: float = 3.0,
        target_pct: float = 30.0,
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
        breakeven_trigger_enabled: bool = False,
        breakeven_trigger_pct: float = 0.0,
    ):
        self.nifty_service    = nifty_service
        self.capital          = capital
        self.st_period        = st_period
        self.st_multiplier    = st_multiplier
        # Full take-profit target, in percent of the entry price. The scaled
        # exits fire at 25%, 50% and 100% of this target move.
        self.target_pct       = target_pct
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
        # Breakeven trigger: once price gains `breakeven_trigger_pct`, raise the
        # stop to the entry price so the trade can no longer turn into a loss.
        self.breakeven_trigger_enabled = breakeven_trigger_enabled
        self.breakeven_trigger_pct     = breakeven_trigger_pct

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _run_symbol(
        self,
        candles: list[dict],
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[SuperTrendTradeResult]:
        """Run the Supertrend strategy on a single option contract's candles."""
        if not candles:
            return []

        ind = compute_supertrend(candles, self.st_period, self.st_multiplier)
        symbol_label = f"NIFTY{strike}{option_type}"
        trades: list[SuperTrendTradeResult] = []

        daily_trade_count: dict[date, int] = defaultdict(int)

        in_position   = False
        entry_row     = None
        entry_price   = 0.0
        total_shares  = 0
        remaining     = 0
        peak_price    = 0.0
        breakeven_armed = False
        tp25_done     = False
        tp50_done     = False
        partials: list[PartialExit] = []

        prev_trend = 0

        def close_position(dt, last_supertrend):
            """Finalise the trade record from accumulated partial legs."""
            nonlocal in_position, entry_row
            sold = sum(p.shares for p in partials)
            total_pnl = round(sum(p.pnl for p in partials), 2)
            wavg_exit = (
                round(sum(p.price * p.shares for p in partials) / sold, 2)
                if sold else entry_price
            )
            duration = int((dt - entry_row["datetime"]).total_seconds() / 60)
            trades.append(SuperTrendTradeResult(
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
                supertrend_at_entry=entry_row["supertrend"],
                supertrend_at_exit=last_supertrend,
                atr_at_entry=entry_row["atr"],
                duration_minutes=duration,
                partials=list(partials),
            ))
            in_position = False
            entry_row   = None

        for _, row in ind.iterrows():
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            today: date  = dt.date()

            trend      = int(row["trend"])
            price      = row["close"]
            supertrend = row["supertrend"]

            if trend == 0:
                prev_trend = trend
                continue

            # ── Exit / scale-out logic ────────────────────────────────────
            if in_position:
                if price > peak_price:
                    peak_price = price

                gain_pct = (price - entry_price) / entry_price * 100.0
                tgt = self.target_pct

                # Arm breakeven once the trigger profit is reached.
                if (
                    self.breakeven_trigger_enabled
                    and not breakeven_armed
                    and gain_pct >= self.breakeven_trigger_pct
                ):
                    breakeven_armed = True

                def book(shares, reason):
                    nonlocal remaining
                    shares = min(shares, remaining)
                    if shares <= 0:
                        return
                    pnl = round(shares * (price - entry_price), 2)
                    partials.append(PartialExit(dt, price, shares, pnl, reason))
                    remaining -= shares

                # 1) Square-off at 15:20 — dump everything.
                if t >= _SQUARE_OFF:
                    book(remaining, "SQUARE_OFF")
                    close_position(dt, supertrend)
                    prev_trend = trend
                    continue

                # 2) Full target reached — sell all remaining.
                if tgt > 0 and gain_pct >= tgt:
                    book(remaining, "TARGET")
                    close_position(dt, supertrend)
                    prev_trend = trend
                    continue

                # 3) Scaled take-profit at 25% and 50% of the target move.
                if tgt > 0 and not tp25_done and gain_pct >= 0.25 * tgt:
                    book(int(round(total_shares * 0.25)), "TP_25")
                    tp25_done = True
                if tgt > 0 and not tp50_done and gain_pct >= 0.50 * tgt:
                    book(int(round(total_shares * 0.25)), "TP_50")
                    tp50_done = True

                # 4) Breakeven stop.
                if breakeven_armed and price <= entry_price:
                    book(remaining, "BREAKEVEN")
                    close_position(dt, supertrend)
                    prev_trend = trend
                    continue

                # 5) Trailing stop off the peak.
                if (
                    self.trailing_stop_enabled
                    and self.trailing_stop_pct > 0
                    and peak_price > 0
                    and price <= peak_price * (1 - self.trailing_stop_pct / 100.0)
                ):
                    book(remaining, "TRAILING_STOP")
                    close_position(dt, supertrend)
                    prev_trend = trend
                    continue

                # 6) Supertrend flip back to bearish.
                if prev_trend == 1 and trend == -1:
                    book(remaining, "SUPERTREND_FLIP")
                    close_position(dt, supertrend)
                    prev_trend = trend
                    continue

                if remaining <= 0:
                    close_position(dt, supertrend)

            # ── Entry logic ───────────────────────────────────────────────
            if (
                not in_position
                and prev_trend != 0
                and _ENTRY_START <= t <= _ENTRY_CUTOFF
                and daily_trade_count[today] < _MAX_TRADES_PER_DAY
            ):
                # Supertrend flip from bearish (-1) to bullish (+1).
                signal = prev_trend == -1 and trend == 1
                if signal and price > 0:
                    entry_price       = price
                    alloc_pct         = _capital_allocation_pct(entry_price)
                    allocated_capital = self.capital * alloc_pct
                    total_shares      = max(floor(allocated_capital / entry_price), 1)
                    remaining         = total_shares
                    in_position       = True
                    entry_row         = row
                    peak_price        = entry_price
                    breakeven_armed   = False
                    tp25_done         = False
                    tp50_done         = False
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
                partials.append(PartialExit(dt, price, remaining, pnl, "SQUARE_OFF"))
                remaining = 0
            close_position(dt, last["supertrend"])

        return trades

    # ── Debug / inspection helpers ────────────────────────────────────────────

    def _print_resampled_with_trades(
        self,
        candles: list[dict],
        trades: list[SuperTrendTradeResult],
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

        ind = compute_supertrend(candles, self.st_period, self.st_multiplier)
        merged = df.merge(
            ind[["datetime", "atr", "supertrend", "trend"]],
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
            )
            for p in t.partials:
                ps = "+" if p.pnl >= 0 else ""
                print(
                    f"        · {p.time.strftime('%H:%M:%S')}  "
                    f"sell {p.shares} @ {p.price:.2f}  pnl {ps}{p.pnl:.2f}  ({p.reason})"
                )

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[WeeklyExpiryResult]:
        expiry_results: list[WeeklyExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        effective_tf = (
            f"{self.resample_seconds}s" if self.resample_seconds > 1 else self.interval
        )
        print(f"\n{'='*70}")
        print(f"  NIFTY SUPERTREND STRATEGY — WEEKLY EXPIRY BACKTEST")
        print(f"  Period    : {self.start_date}  →  {self.end_date}")
        print(f"  Fetch TF  : {self.interval}  |  Effective TF: {effective_tf}")
        print(f"  Capital   : ₹{self.capital:,.0f}  |  ST period: {self.st_period}"
              f"  |  ST mult: {self.st_multiplier}  |  Target: {self.target_pct}%")
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
    ) -> WeeklyExpiryResult | None:
        try:
            nifty_open = self.nifty_service.get_nifty_open(monday)
        except Exception as exc:
            print(f"  [{expiry}] Could not get Nifty open for {monday}: {exc}")
            return None

        strike = NiftyOptionService.atm_strike(nifty_open)
        print(f"  Expiry {expiry}  |  Monday open {nifty_open:.2f}  |  ATM {strike}")

        week_result = WeeklyExpiryResult(
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
    ) -> WeeklyExpiryResult | None:
        print(f"  Expiry {expiry}  |  per-day ATM")

        week_result = WeeklyExpiryResult(
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
    def print_report(expiry_results: list[WeeklyExpiryResult]) -> None:
        all_trades: list[SuperTrendTradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[SuperTrendTradeResult]] = defaultdict(list)
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
