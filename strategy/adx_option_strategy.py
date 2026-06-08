"""
ADX DI+/DI- crossover strategy for Nifty weekly WeeklyOptions.

Entry rules:
  CE — buy when DI+ crosses above DI- and ADX >= threshold
  PE — buy when DI- crosses above DI+ and ADX >= threshold

Exit rules:
  - DI direction reversal (crossover flips)
  - Square-off at 15:20 IST
  - Max 5 trades per day per symbol
  - No new entries before 9:30 or after 14:45
"""

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from math import floor

import numpy as np
import pandas as pd

from models.adx_models import ADXTradeResult, SymbolMetrics, WeeklyExpiryResult
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

    # Build a bucket key: floor each timestamp to the nearest N-second boundary
    # relative to midnight so that bucket boundaries are consistent across days.
    epoch = pd.Timestamp("1970-01-01")
    df["_bucket"] = df["datetime"].apply(
        lambda ts: ts.floor(f"{seconds}s")
    )

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
    resampled["datetime"] = resampled["datetime"].dt.to_pydatetime()
    return resampled.to_dict("records")


_ENTRY_START  = time(9, 30)
_ENTRY_CUTOFF = time(14, 45)
_SQUARE_OFF   = time(15, 20)
_MAX_TRADES_PER_DAY = 5


def _capital_allocation_pct(price: float) -> float:
    """
    Return the fraction of capital to allocate based on option price:
      price <= 30          →  10%
      31 <= price <= 60    →  30%
      61 <= price <= 100   →  50%
      price > 100          → 100%
    """
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


def compute_adx(candles: list[dict], period: int = 14) -> pd.DataFrame:
    """
    Compute ADX, DI+, DI- from a list of OHLC dicts.
    Returns a DataFrame with columns: datetime, adx, di_plus, di_minus.
    """
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    prev_high  = df["high"].shift(1)
    prev_low   = df["low"].shift(1)
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move   = df["high"] - prev_high
    down_move = prev_low   - df["low"]

    plus_dm  = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move

    smooth_tr       = _wilder_ewm(tr,       period)
    smooth_plus_dm  = _wilder_ewm(plus_dm,  period)
    smooth_minus_dm = _wilder_ewm(minus_dm, period)

    di_plus  = 100 * smooth_plus_dm  / smooth_tr
    di_minus = 100 * smooth_minus_dm / smooth_tr

    dx_denom = di_plus + di_minus
    dx = (100 * (di_plus - di_minus).abs() / dx_denom).where(dx_denom != 0, 0.0)
    adx = _wilder_ewm(dx, period)

    # Warm-up masking. Wilder's ADX is only meaningful once the smoothing has
    # seen enough bars: DI+/DI- need ~`period` bars, and ADX (a smoothed DX)
    # needs ~`2 * period` bars before it reflects genuine trend strength rather
    # than the seed value. Without this, the ADX gate would fire on noisy
    # warm-up values right at the start of each contract's data.
    n = len(df)
    di_warmup  = min(period, n)
    adx_warmup = min(2 * period - 1, n)
    di_plus.iloc[:di_warmup]   = np.nan
    di_minus.iloc[:di_warmup]  = np.nan
    adx.iloc[:adx_warmup]      = np.nan

    result = df[["datetime"]].copy()
    result["adx"]      = adx.round(4)
    result["di_plus"]  = di_plus.round(4)
    result["di_minus"] = di_minus.round(4)
    result["close"]    = df["close"]
    return result


def _compute_metrics(symbol: str, trades: list[ADXTradeResult]) -> SymbolMetrics:
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


class ADXOptionStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        start_date: str = "",
        end_date: str = "",
        interval: str = "1minute",
        resample_seconds: int = 1,
        print_resampled: bool = False,
    ):
        self.nifty_service    = nifty_service
        self.capital          = capital
        self.adx_period       = adx_period
        self.adx_threshold    = adx_threshold
        self.start_date       = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date         = datetime.strptime(end_date,   "%d-%b-%Y").date()
        self.interval         = interval
        self.resample_seconds = resample_seconds
        self.print_resampled  = print_resampled

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _run_symbol(
        self,
        candles: list[dict],
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[ADXTradeResult]:
        """
        Run the ADX crossover strategy on a single option contract's candle data.
        Returns a list of completed trades.
        """
        if not candles:
            return []

        indicators = compute_adx(candles, self.adx_period)
        symbol_label = f"NIFTY{strike}{option_type}"
        trades: list[ADXTradeResult] = []

        # Group indicator rows by date for per-day trade counting
        daily_trade_count: dict[date, int] = defaultdict(int)

        in_position   = False
        entry_row     = None
        entry_price   = 0.0
        shares        = 0

        prev_di_plus  = None
        prev_di_minus = None

        for _, row in indicators.iterrows():
            dt: datetime  = row["datetime"]
            t: time       = dt.time()
            today: date   = dt.date()

            adx      = row["adx"]
            di_plus  = row["di_plus"]
            di_minus = row["di_minus"]
            price    = row["close"]

            if pd.isna(adx) or pd.isna(di_plus) or pd.isna(di_minus):
                prev_di_plus  = di_plus
                prev_di_minus = di_minus
                continue

            # ── Exit logic (checked every bar while in position) ──────────
            if in_position:
                exit_reason = None

                # Square-off at 15:20
                if t >= _SQUARE_OFF:
                    exit_reason = "SQUARE_OFF"

                # ADX crossover reversal
                elif option_type == "CE" and prev_di_plus is not None:
                    # CE was entered on DI+ cross above DI-; exit when DI+ crosses below DI-
                    if prev_di_plus >= prev_di_minus and di_plus < di_minus:
                        exit_reason = "ADX_CROSSOVER"
                elif option_type == "PE" and prev_di_plus is not None:
                    # PE was entered on DI- cross above DI+; exit when DI- crosses below DI+
                    if prev_di_minus >= prev_di_plus and di_minus < di_plus:
                        exit_reason = "ADX_CROSSOVER"

                if exit_reason:
                    exit_price = price
                    duration   = int((dt - entry_row["datetime"]).total_seconds() / 60)
                    pnl        = round(shares * (exit_price - entry_price), 2)

                    trades.append(ADXTradeResult(
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
                        adx_at_entry=entry_row["adx"],
                        di_plus_at_entry=entry_row["di_plus"],
                        di_minus_at_entry=entry_row["di_minus"],
                        adx_at_exit=adx,
                        di_plus_at_exit=di_plus,
                        di_minus_at_exit=di_minus,
                        duration_minutes=duration,
                    ))

                    in_position = False
                    entry_row   = None

            # ── Entry logic ───────────────────────────────────────────────
            # Standard ADX/DI system:
            #   • ADX is the trend-strength FILTER — it must already be elevated
            #     (adx >= adx_threshold) for the regime to be considered trending.
            #     ADX is a smoothed average of DX over `adx_period` bars, so it
            #     reflects the strength of the trend that has been building, not
            #     the instantaneous DI gap at the crossover bar.
            #   • The DI crossover is the TRIGGER — it times the entry within that
            #     already-trending regime.
            # Both conditions must hold on the same bar to enter.
            if (
                not in_position
                and prev_di_plus is not None
                and _ENTRY_START <= t <= _ENTRY_CUTOFF
                and daily_trade_count[today] < _MAX_TRADES_PER_DAY
            ):
                # Filter: trend strength must already be elevated.
                adx_ok = adx >= self.adx_threshold

                # Trigger: DI crossover in the option's direction.
                signal = False
                if option_type == "CE":
                    # DI+ crosses above DI-
                    signal = (prev_di_plus <= prev_di_minus) and (di_plus > di_minus)
                elif option_type == "PE":
                    # DI- crosses above DI+
                    signal = (prev_di_minus <= prev_di_plus) and (di_minus > di_plus)

                if adx_ok and signal and price > 0:
                    entry_price       = price
                    alloc_pct         = _capital_allocation_pct(entry_price)
                    allocated_capital = self.capital * alloc_pct
                    shares            = max(floor(allocated_capital / entry_price), 1)
                    in_position       = True
                    entry_row         = row
                    daily_trade_count[today] += 1

            prev_di_plus  = di_plus
            prev_di_minus = di_minus

        # Force-close any open position at end of data
        if in_position and entry_row is not None:
            last  = indicators.iloc[-1]
            price = last["close"]
            dt    = last["datetime"]
            duration = int((dt - entry_row["datetime"]).total_seconds() / 60)
            pnl      = round(shares * (price - entry_price), 2)

            trades.append(ADXTradeResult(
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
                adx_at_entry=entry_row["adx"],
                di_plus_at_entry=entry_row["di_plus"],
                di_minus_at_entry=entry_row["di_minus"],
                adx_at_exit=last["adx"],
                di_plus_at_exit=last["di_plus"],
                di_minus_at_exit=last["di_minus"],
                duration_minutes=duration,
            ))

        return trades

    # ── Debug / inspection helpers ────────────────────────────────────────────

    def _print_resampled_with_trades(
        self,
        candles: list[dict],
        trades: list[ADXTradeResult],
        strike: int,
        option_type: str,
        expiry_date: date,
    ) -> None:
        """
        Print the final resampled candle DataFrame (with ADX/DI+/DI- indicators
        merged in) followed by the trades generated for the same contract.
        """
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

        indicators = compute_adx(candles, self.adx_period)
        merged = df.merge(
            indicators[["datetime", "adx", "di_plus", "di_minus"]],
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

    def run_weekly_backtest(self) -> list[WeeklyExpiryResult]:
        expiry_results: list[WeeklyExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        effective_tf = (
            f"{self.resample_seconds}s" if self.resample_seconds > 1 else self.interval
        )
        print(f"\n{'='*70}")
        print(f"  NIFTY ADX STRATEGY — WEEKLY EXPIRY BACKTEST")
        print(f"  Period    : {self.start_date}  →  {self.end_date}")
        print(f"  Fetch TF  : {self.interval}  |  Effective TF: {effective_tf}")
        print(f"  Capital   : ₹{self.capital:,.0f}  |  ADX period: {self.adx_period}"
              f"  |  ADX threshold: {self.adx_threshold}")
        print(f"  Expiries found: {len(wednesdays)}")
        print(f"{'='*70}\n")

        for expiry in wednesdays:
            monday     = NiftyOptionService.monday_of_week(expiry)
            win_start, win_end = NiftyOptionService.week_window(expiry)

            try:
                nifty_open = self.nifty_service.get_nifty_open(monday)
            except Exception as exc:
                print(f"  [{expiry}] Could not get Nifty open for {monday}: {exc}")
                continue

            strike = NiftyOptionService.atm_strike(nifty_open)

            print(f"  Expiry {expiry}  |  Monday open {nifty_open:.2f}  |  ATM {strike}")

            week_result = WeeklyExpiryResult(
                expiry_date=expiry,
                atm_strike=strike,
                nifty_open=nifty_open,
            )

            from_dt = datetime(win_start.year, win_start.month, win_start.day, 9, 15, 0)
            to_dt   = datetime(win_end.year,   win_end.month,   win_end.day,   15, 30, 0)

            for opt_type in ("CE", "PE"):
                try:
                    candles = self.nifty_service.get_option_candles(
                        strike=strike,
                        expiry_date=expiry,
                        option_type=opt_type,
                        start=from_dt,
                        end=to_dt,
                        interval=self.interval,
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

            expiry_results.append(week_result)

        return expiry_results

    # ── Reporting ─────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(expiry_results: list[WeeklyExpiryResult]) -> None:
        all_trades: list[ADXTradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        # Group by symbol
        by_symbol: dict[str, list[ADXTradeResult]] = defaultdict(list)
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
                f" {'ADX':>6} {'DI+':>6} {'DI-':>6}"
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
                    f" {t.adx_at_exit:>6.2f}"
                    f" {t.di_plus_at_exit:>6.2f}"
                    f" {t.di_minus_at_exit:>6.2f}"
                )

        # Per-symbol metrics
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
