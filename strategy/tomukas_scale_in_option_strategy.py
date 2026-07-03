"""
Tomukas Scale-In v2 strategy for Nifty weekly options (seconds data).

Port of the TradingView Pine strategy "Tomukas Scale-In V2" to the
long-the-premium option backtest framework. Both legs (CE & PE) are traded
independently on the option premium itself:

Trend filter:
  EMA100 > EMA200 on the option premium → bullish premium trend. Only a
  bullish premium trend is traded, since option positions are long-only
  (buying premium). The short side of the Pine script is expressed by the
  opposite leg's own premium turning bullish.

Entry trigger (liquidity sweep):
  The bar's low sweeps below the lowest low of the previous `lookback` bars,
  but the bar closes back above that level AND closes green (close > open),
  while the trend filter holds.

Scale-in engine (pyramiding, up to 5 entries):
  The first sweep opens the position; each subsequent sweep while in a
  position adds another leg, with quantity weights q1..q5 (default
  10/10/20/40/80 — later adds are progressively larger, averaging down).

Exit rules (take-profit only, mirroring the Pine script):
  - Take-profit at (weighted-average entry price + ATR × tp_atr_mult),
    recomputed every bar with the current ATR             → close all
  - Square-off at 15:20 IST                               → close all
  - No new positions before 9:30 or after 14:45 (scale-in adds to an open
    position are allowed until square-off)
  - Max 5 positions per day per symbol
"""

from collections import defaultdict
from datetime import date, datetime, time
from math import floor

import numpy as np
import pandas as pd

from models.tomukas_scale_in_models import (
    TomukasSymbolMetrics,
    TomukasTradeResult,
    TomukasWeeklyExpiryResult,
)
from services.nifty_option_service import NiftyOptionService
from strategy.vwtf_option_strategy import resample_candles

_ENTRY_START  = time(9, 30)
_ENTRY_CUTOFF = time(14, 45)
_SQUARE_OFF   = time(15, 20)
_MAX_TRADES_PER_DAY = 5


def compute_tomukas(
    candles: list[dict],
    ema_fast_period: int,
    ema_slow_period: int,
    sweep_lookback: int,
    atr_period: int,
) -> pd.DataFrame:
    """
    Compute the trend EMAs, the previous-bars lowest low used by the
    liquidity-sweep trigger, and the ATR from a list of OHLC dicts.

    Returns a DataFrame with columns: datetime, open, high, low, close,
    ema_fast, ema_slow, prev_low (lowest low of the prior `sweep_lookback`
    bars, excluding the current bar) and atr.
    """
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    ema_fast = df["close"].ewm(span=ema_fast_period, adjust=False).mean()
    ema_slow = df["close"].ewm(span=ema_slow_period, adjust=False).mean()
    # Mask the warm-up region so the trend filter only fires once the slower
    # EMA has seen a full period of data.
    warmup = min(ema_slow_period, len(df))
    ema_fast.iloc[: max(warmup - 1, 0)] = np.nan
    ema_slow.iloc[: max(warmup - 1, 0)] = np.nan

    # ta.lowest(low[1], lookback): lowest low of the previous `lookback` bars,
    # not including the current bar.
    prev_low = df["low"].shift(1).rolling(
        window=sweep_lookback, min_periods=sweep_lookback
    ).min()

    # Wilder ATR (matches Pine's ta.atr): RMA of the true range.
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()
    atr.iloc[: min(atr_period, len(df)) - 1] = np.nan

    result = df[["datetime", "open", "high", "low", "close"]].copy()
    result["ema_fast"] = ema_fast.round(4)
    result["ema_slow"] = ema_slow.round(4)
    result["prev_low"] = prev_low
    result["atr"]      = atr.round(4)
    return result


def _compute_metrics(symbol: str, trades: list[TomukasTradeResult]) -> TomukasSymbolMetrics:
    if not trades:
        return TomukasSymbolMetrics(
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

    return TomukasSymbolMetrics(
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


class TomukasScaleInOptionStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        ema_fast_period: int = 100,
        ema_slow_period: int = 200,
        sweep_lookback: int = 20,
        entry_weights: tuple[float, ...] = (10, 10, 20, 40, 80),
        atr_period: int = 14,
        tp_atr_mult: float = 1.5,
        start_date: str = "",
        end_date: str = "",
        interval: str = "1minute",
        resample_seconds: int = 1,
        print_resampled: bool = False,
        cache_only: bool = False,
        market_holidays: set[date] | None = None,
        per_day_atm: bool = False,
    ):
        self.nifty_service   = nifty_service
        self.capital         = capital
        self.ema_fast_period = ema_fast_period
        self.ema_slow_period = ema_slow_period
        # Lookback used by the liquidity-sweep trigger: the bar must sweep the
        # lowest low of the previous `sweep_lookback` bars and close back
        # above it.
        self.sweep_lookback  = sweep_lookback
        # Quantity weights for the scale-in legs (Pine q1..q5). Leg i is
        # allocated capital × weight_i / sum(weights); at most len(weights)
        # entries are pyramided into a single position.
        self.entry_weights   = tuple(float(w) for w in entry_weights)
        if not self.entry_weights or any(w <= 0 for w in self.entry_weights):
            raise ValueError("entry_weights must be a non-empty tuple of positive numbers")
        self.atr_period      = atr_period
        # Take-profit distance in ATRs above the weighted-average entry price.
        self.tp_atr_mult     = tp_atr_mult
        self.start_date      = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date        = datetime.strptime(end_date,   "%d-%b-%Y").date()
        self.interval        = interval
        self.resample_seconds = resample_seconds
        self.print_resampled = print_resampled
        self.cache_only      = cache_only
        self.market_holidays = set(market_holidays) if market_holidays else set()
        self.per_day_atm     = per_day_atm

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _run_symbol(
        self,
        candles: list[dict],
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[TomukasTradeResult]:
        """
        Run the Tomukas Scale-In v2 strategy on a single option contract's
        candle data. Returns a list of completed trades.
        """
        if not candles:
            return []

        indicators = compute_tomukas(
            candles,
            self.ema_fast_period,
            self.ema_slow_period,
            self.sweep_lookback,
            self.atr_period,
        )
        symbol_label = f"NIFTY{strike}{option_type}"
        trades: list[TomukasTradeResult] = []

        daily_trade_count: dict[date, int] = defaultdict(int)
        total_weight = sum(self.entry_weights)
        max_entries  = len(self.entry_weights)

        in_position  = False
        entry_row    = None      # row of the FIRST scale-in leg
        num_entries  = 0
        total_shares = 0
        total_cost   = 0.0       # Σ qty × price across scale-in legs
        atr_at_entry = 0.0
        target_price = 0.0
        leg_descriptions: list[str] = []

        def _avg_price() -> float:
            return total_cost / total_shares if total_shares else 0.0

        def _record_trade(dt: datetime, exit_price: float, reason: str) -> None:
            avg_entry = _avg_price()
            duration  = int((dt - entry_row["datetime"]).total_seconds() / 60)
            trades.append(TomukasTradeResult(
                symbol=symbol_label,
                option_type=option_type,
                strike=strike,
                expiry_date=expiry_date,
                entry_time=entry_row["datetime"],
                exit_time=dt,
                entry_price=round(avg_entry, 2),
                exit_price=round(exit_price, 2),
                shares=total_shares,
                pnl=round(total_shares * (exit_price - avg_entry), 2),
                exit_reason=reason,
                ema_fast_at_entry=entry_row["ema_fast"],
                ema_slow_at_entry=entry_row["ema_slow"],
                atr_at_entry=atr_at_entry,
                target_price=round(target_price, 2),
                num_entries=num_entries,
                duration_minutes=duration,
                scale_in_legs=" | ".join(leg_descriptions),
            ))

        for _, row in indicators.iterrows():
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            today: date  = dt.date()

            ema_fast = row["ema_fast"]
            ema_slow = row["ema_slow"]
            prev_low = row["prev_low"]
            atr      = row["atr"]
            price    = row["close"]

            if pd.isna(ema_fast) or pd.isna(ema_slow) or pd.isna(atr):
                continue

            # ── Exit logic (checked every bar while in position) ──────────
            if in_position:
                # TP recomputed each bar with the live ATR, as in the Pine
                # script's strategy.exit(limit = avg + atr × mult).
                target_price = _avg_price() + atr * self.tp_atr_mult

                exit_reason = None
                if t >= _SQUARE_OFF:
                    exit_reason = "SQUARE_OFF"
                elif row["high"] >= target_price:
                    exit_reason = "TARGET"

                if exit_reason:
                    exit_price = (
                        target_price if exit_reason == "TARGET" else price
                    )
                    _record_trade(dt, exit_price, exit_reason)
                    in_position  = False
                    entry_row    = None
                    num_entries  = 0
                    total_shares = 0
                    total_cost   = 0.0
                    leg_descriptions = []
                    continue

            # ── Liquidity-sweep signal ────────────────────────────────────
            bull_trend = ema_fast > ema_slow
            long_sweep = (
                bull_trend
                and not pd.isna(prev_low)
                and row["low"] < prev_low
                and price > prev_low
                and price > row["open"]
            )

            if not long_sweep or price <= 0:
                continue

            # ── First entry ───────────────────────────────────────────────
            if (
                not in_position
                and _ENTRY_START <= t <= _ENTRY_CUTOFF
                and daily_trade_count[today] < _MAX_TRADES_PER_DAY
            ):
                weight = self.entry_weights[0]
                qty = max(floor(self.capital * (weight / total_weight) / price), 1)

                in_position  = True
                entry_row    = row
                num_entries  = 1
                total_shares = qty
                total_cost   = qty * price
                atr_at_entry = atr
                target_price = price + atr * self.tp_atr_mult
                leg_descriptions = [f"L1@{price:.2f}x{qty}"]
                daily_trade_count[today] += 1

            # ── Scale-in adds (pyramiding) ────────────────────────────────
            elif in_position and num_entries < max_entries and t < _SQUARE_OFF:
                weight = self.entry_weights[num_entries]
                qty = max(floor(self.capital * (weight / total_weight) / price), 1)

                num_entries  += 1
                total_shares += qty
                total_cost   += qty * price
                leg_descriptions.append(f"L{num_entries}@{price:.2f}x{qty}")

        # Force-close any open position at end of data
        if in_position and entry_row is not None and total_shares > 0:
            last = indicators.iloc[-1]
            _record_trade(last["datetime"], last["close"], "SQUARE_OFF")

        return trades

    # ── Debug / inspection helpers ────────────────────────────────────────────

    def _print_resampled_with_trades(
        self,
        candles: list[dict],
        trades: list[TomukasTradeResult],
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

        indicators = compute_tomukas(
            candles,
            self.ema_fast_period,
            self.ema_slow_period,
            self.sweep_lookback,
            self.atr_period,
        )

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
                f"  avg entry {t.entry_price:.2f}  exit {t.exit_price:.2f}"
                f"  qty {t.shares}  entries {t.num_entries}"
                f"  pnl {pnl_sign}{t.pnl:.2f}  ({t.exit_reason})"
                f"  [{t.scale_in_legs}]"
            )

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[TomukasWeeklyExpiryResult]:
        expiry_results: list[TomukasWeeklyExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        effective_tf = (
            f"{self.resample_seconds}s" if self.resample_seconds > 1 else self.interval
        )
        print(f"\n{'='*70}")
        print(f"  NIFTY TOMUKAS SCALE-IN V2 — WEEKLY EXPIRY BACKTEST")
        print(f"  Period    : {self.start_date}  →  {self.end_date}")
        print(f"  Fetch TF  : {self.interval}  |  Effective TF: {effective_tf}")
        print(f"  Capital   : ₹{self.capital:,.0f}"
              f"  |  EMAs: {self.ema_fast_period}/{self.ema_slow_period}"
              f"  |  Sweep lookback: {self.sweep_lookback}")
        print(f"  Weights   : {self.entry_weights}"
              f"  |  ATR: {self.atr_period}  |  TP mult: {self.tp_atr_mult}")
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
    ) -> TomukasWeeklyExpiryResult | None:
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

        week_result = TomukasWeeklyExpiryResult(
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
    ) -> TomukasWeeklyExpiryResult | None:
        """
        Per-day ATM mode: for each trading day in the expiry window, choose a
        fresh ATM strike from that day's Nifty open and trade only that day.
        """
        print(f"  Expiry {expiry}  |  per-day ATM")

        week_result = TomukasWeeklyExpiryResult(
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
    def print_report(expiry_results: list[TomukasWeeklyExpiryResult]) -> None:
        all_trades: list[TomukasTradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[TomukasTradeResult]] = defaultdict(list)
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
                f" {'AvgEnt₹':>8} {'Exit₹':>8} {'Tgt₹':>8} {'Qty':>6}"
                f" {'Legs':>5} {'PnL':>10} {'Reason':<12}"
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
                    f" {t.target_price:>8.2f}"
                    f" {t.shares:>6}"
                    f" {t.num_entries:>5}"
                    f" {pnl_sign+f'{t.pnl:.2f}':>10}"
                    f" {t.exit_reason:<12}"
                )
                if t.scale_in_legs:
                    print(f"      legs: {t.scale_in_legs}")

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
