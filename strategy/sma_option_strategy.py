"""
SMA fast/slow crossover strategy (with volume confirmation) for Nifty weekly options.

Entry rules:
  CE — buy when the fast SMA crosses above the slow SMA and volume is good
  PE — buy when the fast SMA crosses above the slow SMA and volume is good

  "Good volume" means the entry bar's volume is at least `volume_factor` times
  the rolling-average volume over `sma_slow` bars.

Exit rules:
  - SMA crossover reversal (fast SMA crosses back below the slow SMA)
  - Trailing stop-loss (optional, percentage-based)
  - Break-even stop: optional; once price moves `breakeven_trigger_pct` percent
    above entry, the stop-loss is moved up to the entry price
  - Square-off at 15:20 IST
  - Max 5 trades per day per symbol
  - No new entries before 9:30 or after 14:45
"""

from collections import defaultdict
from datetime import date, datetime, time
from math import floor

import numpy as np
import pandas as pd

from models.sma_models import (
    SMASymbolMetrics,
    SMATradeResult,
    SMAWeeklyExpiryResult,
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
    Mirrors the allocation tiers used by the ADX strategy.
    """
    if price <= 20:
        return 0.30
    elif price <= 60:
        return 1.00
    elif price <= 100:
        return 1.00
    else:
        return 1.00


def compute_sma(candles: list[dict], fast: int, slow: int) -> pd.DataFrame:
    """
    Compute fast/slow simple moving averages from a list of OHLC dicts.
    Returns a DataFrame with columns: datetime, sma_fast, sma_slow, close,
    volume and vol_avg (rolling average volume over `slow` bars).
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

    sma_fast = df["close"].rolling(window=fast, min_periods=fast).mean()
    sma_slow = df["close"].rolling(window=slow, min_periods=slow).mean()

    # Rolling average volume used as the "good volume" reference. Shift by one
    # so the current bar's own volume does not inflate its own benchmark.
    vol_avg = df["volume"].rolling(window=slow, min_periods=1).mean().shift(1)

    result = df[["datetime"]].copy()
    result["sma_fast"] = sma_fast.round(4)
    result["sma_slow"] = sma_slow.round(4)
    result["close"]    = df["close"]
    result["volume"]   = df["volume"]
    result["vol_avg"]  = vol_avg
    return result


def _compute_metrics(symbol: str, trades: list[SMATradeResult]) -> SMASymbolMetrics:
    if not trades:
        return SMASymbolMetrics(
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

    return SMASymbolMetrics(
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


class SMAOptionStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        sma_fast: int = 9,
        sma_slow: int = 21,
        volume_factor: float = 1.0,
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
        self.sma_fast         = sma_fast
        self.sma_slow         = sma_slow
        # Minimum ratio of the current bar's volume to the rolling average
        # volume required to confirm an entry. 1.0 means at least average
        # volume; >1.0 demands an above-average ("good") volume surge. Set to 0
        # to disable the volume filter entirely.
        self.volume_factor    = volume_factor
        self.start_date       = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date         = datetime.strptime(end_date,   "%d-%b-%Y").date()
        self.interval         = interval
        self.resample_seconds = resample_seconds
        self.print_resampled  = print_resampled
        self.cache_only       = cache_only
        self.market_holidays  = set(market_holidays) if market_holidays else set()
        self.per_day_atm      = per_day_atm
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_pct     = trailing_stop_pct
        # Break-even stop. Once the option price moves `breakeven_trigger_pct`
        # percent above entry, the stop-loss is moved up to the entry price so the
        # trade can no longer turn into a loss (locks in break-even).
        self.breakeven_enabled     = breakeven_enabled
        self.breakeven_trigger_pct = breakeven_trigger_pct

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _run_symbol(
        self,
        candles: list[dict],
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[SMATradeResult]:
        """
        Run the SMA crossover strategy on a single option contract's candle data.
        Returns a list of completed trades.
        """
        if not candles:
            return []

        indicators = compute_sma(candles, self.sma_fast, self.sma_slow)
        symbol_label = f"NIFTY{strike}{option_type}"
        trades: list[SMATradeResult] = []

        daily_trade_count: dict[date, int] = defaultdict(int)

        in_position   = False
        entry_row     = None
        entry_price   = 0.0
        shares        = 0
        peak_price    = 0.0   # highest price seen since entry (for trailing stop)
        breakeven_stop = 0.0  # stop-loss level once moved to break-even
        moved_to_breakeven = False

        prev_fast = None
        prev_slow = None

        for _, row in indicators.iterrows():
            dt: datetime  = row["datetime"]
            t: time       = dt.time()
            today: date   = dt.date()

            sma_fast = row["sma_fast"]
            sma_slow = row["sma_slow"]
            price    = row["close"]
            volume   = row["volume"]
            vol_avg  = row["vol_avg"]

            if pd.isna(sma_fast) or pd.isna(sma_slow):
                prev_fast = sma_fast
                prev_slow = sma_slow
                continue

            # ── Exit logic (checked every bar while in position) ──────────
            if in_position:
                exit_reason = None

                if price > peak_price:
                    peak_price = price

                # Break-even: once price moves `breakeven_trigger_pct` above entry,
                # ratchet the stop-loss up to the entry price (lock in break-even).
                if (
                    self.breakeven_enabled
                    and not moved_to_breakeven
                    and price >= entry_price * (1 + self.breakeven_trigger_pct / 100.0)
                ):
                    breakeven_stop = entry_price
                    moved_to_breakeven = True

                # Square-off at 15:20
                if t >= _SQUARE_OFF:
                    exit_reason = "SQUARE_OFF"

                # Break-even stop: after the stop has been moved to entry, a fall
                # back to the entry price exits the trade flat instead of at a loss.
                elif moved_to_breakeven and price <= breakeven_stop:
                    exit_reason = "BREAKEVEN"

                # Trailing stop-loss
                elif (
                    self.trailing_stop_enabled
                    and self.trailing_stop_pct > 0
                    and peak_price > 0
                    and price <= peak_price * (1 - self.trailing_stop_pct / 100.0)
                ):
                    exit_reason = "TRAILING_STOP"

                # SMA crossover reversal: fast crosses back below slow
                elif prev_fast is not None and not pd.isna(prev_fast):
                    if prev_fast >= prev_slow and sma_fast < sma_slow:
                        exit_reason = "SMA_CROSSOVER"

                if exit_reason:
                    exit_price = price
                    duration   = int((dt - entry_row["datetime"]).total_seconds() / 60)
                    pnl        = round(shares * (exit_price - entry_price), 2)

                    trades.append(SMATradeResult(
                        symbol=symbol_label,
                        option_type=option_type,
                        strike=strike,
                        expiry_date=expiry_date,
                        entry_time=entry_row["datetime"],
                        exit_time=dt,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        shares=shares,
                        pnl=pnl,
                        exit_reason=exit_reason,
                        sma_fast_at_entry=entry_row["sma_fast"],
                        sma_slow_at_entry=entry_row["sma_slow"],
                        sma_fast_at_exit=sma_fast,
                        sma_slow_at_exit=sma_slow,
                        duration_minutes=duration,
                    ))

                    in_position = False
                    entry_row   = None

            # ── Entry logic ───────────────────────────────────────────────
            if (
                not in_position
                and prev_fast is not None
                and not pd.isna(prev_fast)
                and not pd.isna(prev_slow)
                and _ENTRY_START <= t <= _ENTRY_CUTOFF
                and daily_trade_count[today] < _MAX_TRADES_PER_DAY
            ):
                # Filter: confirm the move with good volume.
                if self.volume_factor <= 0 or pd.isna(vol_avg) or vol_avg <= 0:
                    volume_ok = False
                else:
                    volume_ok = volume >= self.volume_factor * vol_avg

                # Trigger: fast SMA crosses above slow SMA. Both CE and PE legs
                # are bought on the same long-side crossover of their own option
                # price (a rising option premium).
                signal = (prev_fast <= prev_slow) and (sma_fast > sma_slow)

                if signal and volume_ok and price > 0:
                    entry_price       = price
                    alloc_pct         = _capital_allocation_pct(entry_price)
                    allocated_capital = self.capital * alloc_pct
                    shares            = max(floor(allocated_capital / entry_price), 1)
                    in_position       = True
                    entry_row         = row
                    peak_price        = entry_price
                    breakeven_stop    = 0.0
                    moved_to_breakeven = False
                    daily_trade_count[today] += 1

            prev_fast = sma_fast
            prev_slow = sma_slow

        # Force-close any open position at end of data
        if in_position and entry_row is not None:
            last  = indicators.iloc[-1]
            price = last["close"]
            dt    = last["datetime"]
            duration = int((dt - entry_row["datetime"]).total_seconds() / 60)
            pnl      = round(shares * (price - entry_price), 2)

            trades.append(SMATradeResult(
                symbol=symbol_label,
                option_type=option_type,
                strike=strike,
                expiry_date=expiry_date,
                entry_time=entry_row["datetime"],
                exit_time=dt,
                entry_price=entry_price,
                exit_price=price,
                shares=shares,
                pnl=pnl,
                exit_reason="SQUARE_OFF",
                sma_fast_at_entry=entry_row["sma_fast"],
                sma_slow_at_entry=entry_row["sma_slow"],
                sma_fast_at_exit=last["sma_fast"],
                sma_slow_at_exit=last["sma_slow"],
                duration_minutes=duration,
            ))

        return trades

    # ── Debug / inspection helpers ────────────────────────────────────────────

    def _print_resampled_with_trades(
        self,
        candles: list[dict],
        trades: list[SMATradeResult],
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

        indicators = compute_sma(candles, self.sma_fast, self.sma_slow)
        merged = df.merge(
            indicators[["datetime", "sma_fast", "sma_slow"]],
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

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[SMAWeeklyExpiryResult]:
        expiry_results: list[SMAWeeklyExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        effective_tf = (
            f"{self.resample_seconds}s" if self.resample_seconds > 1 else self.interval
        )
        print(f"\n{'='*70}")
        print(f"  NIFTY SMA CROSSOVER STRATEGY — WEEKLY EXPIRY BACKTEST")
        print(f"  Period    : {self.start_date}  →  {self.end_date}")
        print(f"  Fetch TF  : {self.interval}  |  Effective TF: {effective_tf}")
        print(f"  Capital   : ₹{self.capital:,.0f}  |  SMA fast: {self.sma_fast}"
              f"  |  SMA slow: {self.sma_slow}  |  Vol factor: {self.volume_factor}")
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
    ) -> SMAWeeklyExpiryResult | None:
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

        week_result = SMAWeeklyExpiryResult(
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
    ) -> SMAWeeklyExpiryResult | None:
        """
        Per-day ATM mode: for each trading day in the expiry window, choose a
        fresh ATM strike from that day's Nifty open and trade only that day.
        """
        print(f"  Expiry {expiry}  |  per-day ATM")

        week_result = SMAWeeklyExpiryResult(
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
    def print_report(expiry_results: list[SMAWeeklyExpiryResult]) -> None:
        all_trades: list[SMATradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[SMATradeResult]] = defaultdict(list)
        for t in all_trades:
            by_symbol[t.symbol].append(t)

        sep = "─" * 90
        print(f"\n{'='*90}")
        print("  RESULTS BY EXPIRY")
        print(f"{'='*90}")

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
                f"  {'Symbol':<22} {'Entry':>19} {'Exit':>19}"
                f" {'Entry₹':>8} {'Exit₹':>8} {'Qty':>5}"
                f" {'PnL':>10} {'Reason':<14}"
                f" {'SMAf':>8} {'SMAs':>8}"
            )
            print(header)
            print(f"  {sep}")

            for t in sorted(er.all_trades, key=lambda x: x.entry_time):
                pnl_sign = "+" if t.pnl >= 0 else ""
                print(
                    f"  {t.symbol:<22}"
                    f" {t.entry_time.strftime('%d-%b %H:%M'):>19}"
                    f" {t.exit_time.strftime('%d-%b %H:%M'):>19}"
                    f" {t.entry_price:>8.2f}"
                    f" {t.exit_price:>8.2f}"
                    f" {t.shares:>5}"
                    f" {pnl_sign+f'{t.pnl:.2f}':>10}"
                    f" {t.exit_reason:<14}"
                    f" {t.sma_fast_at_exit:>8.2f}"
                    f" {t.sma_slow_at_exit:>8.2f}"
                )

        print(f"\n{'='*90}")
        print("  SYMBOL-WISE METRICS")
        print(f"{'='*90}")

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
        print(f"  {'─'*88}")

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
        print(f"  {'─'*88}")
        print(
            f"  {'OVERALL':<{col_w['sym']}} {overall_trades:>{col_w['trades']}}"
            f" {overall_wins:>{col_w['wins']}} {overall_losses:>{col_w['loss']}}"
            f" {wr_overall:>{col_w['wr']}.1f} {pnl_s:>{col_w['pnl']}}"
        )
        print(f"{'='*90}\n")
