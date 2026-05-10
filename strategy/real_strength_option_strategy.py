"""
Real Strength Histogram — Nifty Weekly Options Backtest
=======================================================

Signals are generated from Nifty 50 spot (5-minute candles).
Trades are executed on the ATM CE or PE option contract for the expiry week.

Entry (Long → CE):
  - Real Strength histogram > strength_threshold
  - Histogram rising vs previous bar
  - ADX >= min_adx
  - DI+ > DI-
  - Volume ratio >= vol_ratio_min
  - Optional: SMA(fast) > SMA(slow)

Entry (Short → PE): mirror conditions on bearish side.

Exit — Regime-dependent:
  - Hard stop loss (static %) always active
  - Min hold of min_bars_hold bars before peak/flip exits fire
  - SMA still confirms → exit when histogram crosses into opposite zone
    past flip_threshold (±flip_threshold)
  - SMA reversed against position → exit when histogram falls peak_drop_pct%
    from its peak value reached during the trade
  - Square-off at 15:20 IST

Re-entry lock: after a stop loss, no re-entry in same direction until
histogram returns to zero.
"""

from collections import defaultdict
from datetime import date, datetime, time
from math import floor

import numpy as np
import pandas as pd

from models.real_strength_models import (
    RSSymbolMetrics,
    RSTradeResult,
    RSWeeklyExpiryResult,
)
from services.nifty_option_service import NiftyOptionService
from strategy.adx_option_strategy import compute_adx

_ENTRY_START  = time(9, 30)
_ENTRY_CUTOFF = time(14, 45)
_SQUARE_OFF   = time(15, 20)
_RS_FLOOR     = 0.1   # prevents amplifiers from zeroing out on quiet bars


def _wilder_ewm(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_real_strength(
    candles: list[dict],
    adx_period: int = 14,
    roc_period: int = 10,
    vol_ma_period: int = 20,
    smooth_period: int = 3,
    sma_fast: int = 30,
    sma_slow: int = 60,
) -> pd.DataFrame:
    """
    Compute the Real Strength histogram and supporting indicators.

    Real Strength = EMA( ROC × max(vol/vol_MA, floor) × max(ADX/20, floor) )

    Returns a DataFrame with columns:
      datetime, close, volume, adx, di_plus, di_minus,
      roc, vol_ratio, histogram, sma_fast, sma_slow
    """
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    # ADX / DI± via existing helper (returns separate df; merge by position)
    adx_df = compute_adx(candles, adx_period)
    df["adx"]      = adx_df["adx"].values
    df["di_plus"]  = adx_df["di_plus"].values
    df["di_minus"] = adx_df["di_minus"].values

    # Rate of Change (%)
    df["roc"] = df["close"].pct_change(periods=roc_period) * 100

    # Volume ratio
    df["vol_ma"]    = df["volume"].rolling(vol_ma_period, min_periods=1).mean()
    df["vol_ratio"] = (df["volume"] / df["vol_ma"]).clip(lower=0)

    # ADX scaled around 20 reference — at ADX=20 amplifier = 1.0
    df["adx_scaled"] = df["adx"] / 20.0

    # Real Strength raw: sign from ROC, magnitude from vol & ADX amplifiers
    vol_amp = df["vol_ratio"].clip(lower=_RS_FLOOR)
    adx_amp = df["adx_scaled"].clip(lower=_RS_FLOOR)
    df["rs_raw"] = df["roc"] * vol_amp * adx_amp

    # Smooth with EMA
    df["histogram"] = _ema(df["rs_raw"], smooth_period)

    # Dual SMA trend filter
    df["sma_fast"] = df["close"].rolling(sma_fast, min_periods=1).mean()
    df["sma_slow"] = df["close"].rolling(sma_slow, min_periods=1).mean()

    return df[[
        "datetime", "close", "volume", "adx", "di_plus", "di_minus",
        "roc", "vol_ratio", "histogram", "sma_fast", "sma_slow",
    ]]


def _build_price_map(candles: list[dict]) -> dict[datetime, float]:
    """Map datetime → close price for an option contract."""
    price_map: dict[datetime, float] = {}
    for c in candles:
        dt = pd.to_datetime(c["datetime"]).to_pydatetime()
        try:
            price_map[dt] = float(c["close"])
        except (ValueError, TypeError):
            pass
    return price_map


def _compute_metrics(symbol: str, trades: list[RSTradeResult]) -> RSSymbolMetrics:
    if not trades:
        return RSSymbolMetrics(
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

    return RSSymbolMetrics(
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


class RealStrengthOptionStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        adx_period: int = 14,
        roc_period: int = 10,
        vol_ma_period: int = 20,
        smooth_period: int = 3,
        strength_threshold: float = 1.0,
        min_adx: float = 14.0,
        vol_ratio_min: float = 1.2,
        sma_fast: int = 30,
        sma_slow: int = 60,
        use_sma_filter: bool = True,
        stop_loss_pct: float = 1.0,
        peak_drop_pct: float = 25.0,
        flip_threshold: float = 0.8,
        min_bars_hold: int = 3,
        start_date: str = "",
        end_date: str = "",
        interval: str = "5minute",
    ):
        self.nifty_service      = nifty_service
        self.capital            = capital
        self.adx_period         = adx_period
        self.roc_period         = roc_period
        self.vol_ma_period      = vol_ma_period
        self.smooth_period      = smooth_period
        self.strength_threshold = strength_threshold
        self.min_adx            = min_adx
        self.vol_ratio_min      = vol_ratio_min
        self.sma_fast           = sma_fast
        self.sma_slow           = sma_slow
        self.use_sma_filter     = use_sma_filter
        self.stop_loss_pct      = stop_loss_pct / 100.0
        self.peak_drop_pct      = peak_drop_pct / 100.0
        self.flip_threshold     = flip_threshold
        self.min_bars_hold      = min_bars_hold
        self.start_date         = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date           = datetime.strptime(end_date,   "%d-%b-%Y").date()
        self.interval           = interval

    # ── Core per-expiry strategy ──────────────────────────────────────────────

    def _run_expiry(
        self,
        signal_df: pd.DataFrame,
        ce_prices: dict[datetime, float],
        pe_prices: dict[datetime, float],
        strike: int,
        expiry: date,
    ) -> tuple[list[RSTradeResult], list[RSTradeResult]]:
        """
        Simulate one direction at a time is NOT how this strategy works.
        We run a unified state machine: one position at a time (CE or PE),
        driven by spot signals but priced off the option contracts.
        """
        ce_trades: list[RSTradeResult] = []
        pe_trades: list[RSTradeResult] = []

        # Position state
        in_position   = False
        opt_type      = ""          # "CE" or "PE"
        entry_dt: datetime | None = None
        entry_price   = 0.0
        shares        = 0
        bars_in_trade = 0
        peak_hist     = 0.0         # max (CE) or min (PE) histogram during trade
        stop_price    = 0.0
        entry_row: pd.Series | None = None

        # Re-entry lock: key = "CE"/"PE", cleared once histogram touches zero
        stop_locked: dict[str, bool] = {"CE": False, "PE": False}

        prev_hist = None

        rows = list(signal_df.itertuples(index=False))

        for i, row in enumerate(rows):
            dt: datetime = row.datetime
            t: time      = dt.time()

            hist      = row.histogram
            adx       = row.adx
            di_plus   = row.di_plus
            di_minus  = row.di_minus
            vol_ratio = row.vol_ratio
            sma_f     = row.sma_fast
            sma_s     = row.sma_slow

            if pd.isna(hist) or pd.isna(adx):
                prev_hist = hist
                continue

            # Update re-entry lock: unlock when histogram crosses through zero
            if hist <= 0:
                stop_locked["CE"] = False
            if hist >= 0:
                stop_locked["PE"] = False

            # ── Exit logic ────────────────────────────────────────────────
            if in_position:
                bars_in_trade += 1

                # Look up current option price
                opt_prices = ce_prices if opt_type == "CE" else pe_prices
                cur_price  = opt_prices.get(dt)

                if cur_price is None:
                    # No option tick at this bar — keep position open
                    prev_hist = hist
                    continue

                # Track histogram peak during the trade
                if opt_type == "CE":
                    peak_hist = max(peak_hist, hist)
                else:
                    peak_hist = min(peak_hist, hist)

                exit_reason = None

                # 1. Hard stop loss
                if opt_type == "CE" and cur_price <= stop_price:
                    exit_reason = "STOP_LOSS"
                    stop_locked["CE"] = True
                elif opt_type == "PE" and cur_price <= stop_price:
                    exit_reason = "STOP_LOSS"
                    stop_locked["PE"] = True

                # 2. Square-off
                elif t >= _SQUARE_OFF:
                    exit_reason = "SQUARE_OFF"

                # 3. Regime-dependent exits (only after min hold)
                elif bars_in_trade >= self.min_bars_hold:
                    sma_confirms = (
                        (opt_type == "CE" and sma_f > sma_s) or
                        (opt_type == "PE" and sma_f < sma_s)
                    )
                    if sma_confirms:
                        # Flip exit: histogram crosses into opposite zone
                        if opt_type == "CE" and hist < -self.flip_threshold:
                            exit_reason = "FLIP_EXIT"
                        elif opt_type == "PE" and hist > self.flip_threshold:
                            exit_reason = "FLIP_EXIT"
                    else:
                        # Peak-drop exit: histogram fell peak_drop_pct from trade peak
                        if opt_type == "CE" and peak_hist > 0:
                            if hist <= peak_hist * (1.0 - self.peak_drop_pct):
                                exit_reason = "PEAK_DROP_EXIT"
                        elif opt_type == "PE" and peak_hist < 0:
                            if hist >= peak_hist * (1.0 - self.peak_drop_pct):
                                exit_reason = "PEAK_DROP_EXIT"

                if exit_reason:
                    exit_price   = cur_price
                    duration_min = int((dt - entry_dt).total_seconds() / 60)
                    pnl          = round(shares * (exit_price - entry_price), 2)
                    symbol_lbl   = f"NIFTY{strike}{opt_type}"

                    result = RSTradeResult(
                        symbol=symbol_lbl,
                        option_type=opt_type,
                        strike=strike,
                        expiry_date=expiry,
                        entry_time=entry_dt,
                        exit_time=dt,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        shares=shares,
                        pnl=pnl,
                        exit_reason=exit_reason,
                        histogram_at_entry=float(entry_row.histogram),
                        adx_at_entry=float(entry_row.adx),
                        di_plus_at_entry=float(entry_row.di_plus),
                        di_minus_at_entry=float(entry_row.di_minus),
                        histogram_at_exit=float(hist),
                        adx_at_exit=float(adx),
                        duration_minutes=duration_min,
                    )
                    if opt_type == "CE":
                        ce_trades.append(result)
                    else:
                        pe_trades.append(result)

                    in_position   = False
                    opt_type      = ""
                    bars_in_trade = 0
                    peak_hist     = 0.0

            # ── Entry logic ───────────────────────────────────────────────
            if (
                not in_position
                and prev_hist is not None
                and _ENTRY_START <= t <= _ENTRY_CUTOFF
                and adx >= self.min_adx
                and vol_ratio >= self.vol_ratio_min
            ):
                # Long → CE
                long_signal = (
                    hist > self.strength_threshold
                    and hist > prev_hist          # histogram rising
                    and di_plus > di_minus
                    and not stop_locked["CE"]
                    and (not self.use_sma_filter or sma_f > sma_s)
                )
                # Short → PE
                short_signal = (
                    hist < -self.strength_threshold
                    and hist < prev_hist          # histogram falling
                    and di_minus > di_plus
                    and not stop_locked["PE"]
                    and (not self.use_sma_filter or sma_f < sma_s)
                )

                if long_signal or short_signal:
                    candidate_type  = "CE" if long_signal else "PE"
                    opt_prices      = ce_prices if candidate_type == "CE" else pe_prices
                    option_price    = opt_prices.get(dt)

                    if option_price and option_price > 0:
                        opt_type      = candidate_type
                        entry_dt      = dt
                        entry_price   = option_price
                        shares        = max(floor(self.capital / entry_price), 1)
                        in_position   = True
                        bars_in_trade = 0
                        entry_row     = row
                        peak_hist     = hist
                        # Stop loss is on option price
                        stop_price    = round(entry_price * (1.0 - self.stop_loss_pct), 2)

            prev_hist = hist

        # Force-close any open position at end of data
        if in_position and entry_dt is not None:
            last_row  = rows[-1]
            dt        = last_row.datetime
            opt_prices = ce_prices if opt_type == "CE" else pe_prices
            cur_price  = opt_prices.get(dt, entry_price)
            duration_min = int((dt - entry_dt).total_seconds() / 60)
            pnl          = round(shares * (cur_price - entry_price), 2)
            symbol_lbl   = f"NIFTY{strike}{opt_type}"

            result = RSTradeResult(
                symbol=symbol_lbl,
                option_type=opt_type,
                strike=strike,
                expiry_date=expiry,
                entry_time=entry_dt,
                exit_time=dt,
                entry_price=entry_price,
                exit_price=cur_price,
                shares=shares,
                pnl=pnl,
                exit_reason="SQUARE_OFF",
                histogram_at_entry=float(entry_row.histogram),
                adx_at_entry=float(entry_row.adx),
                di_plus_at_entry=float(entry_row.di_plus),
                di_minus_at_entry=float(entry_row.di_minus),
                histogram_at_exit=float(last_row.histogram),
                adx_at_exit=float(last_row.adx),
                duration_minutes=duration_min,
            )
            if opt_type == "CE":
                ce_trades.append(result)
            else:
                pe_trades.append(result)

        return ce_trades, pe_trades

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[RSWeeklyExpiryResult]:
        expiry_results: list[RSWeeklyExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        print(f"\n{'='*70}")
        print(f"  NIFTY REAL STRENGTH STRATEGY — WEEKLY EXPIRY BACKTEST")
        print(f"  Period   : {self.start_date}  →  {self.end_date}")
        print(f"  Capital  : ₹{self.capital:,.0f}  |  Interval: {self.interval}")
        print(f"  Threshold: {self.strength_threshold}  |  Min ADX: {self.min_adx}"
              f"  |  Vol Ratio: {self.vol_ratio_min}")
        print(f"  SMA      : {self.sma_fast}/{self.sma_slow}"
              f"  (filter {'ON' if self.use_sma_filter else 'OFF'})")
        print(f"  Stop Loss: {self.stop_loss_pct*100:.1f}%"
              f"  |  Peak Drop: {self.peak_drop_pct*100:.0f}%"
              f"  |  Flip: ±{self.flip_threshold}")
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

            from_dt = datetime(win_start.year, win_start.month, win_start.day, 9, 15, 0)
            to_dt   = datetime(win_end.year,   win_end.month,   win_end.day,   15, 30, 0)

            # 1. Fetch spot candles for signal generation
            try:
                spot_candles = self.nifty_service.get_nifty_spot_candles(
                    start=from_dt, end=to_dt, interval=self.interval
                )
                if not spot_candles:
                    print(f"    No spot candles — skipping")
                    continue
            except Exception as exc:
                print(f"    Spot fetch error: {exc}")
                continue

            # 2. Compute indicators on spot
            signal_df = compute_real_strength(
                candles=spot_candles,
                adx_period=self.adx_period,
                roc_period=self.roc_period,
                vol_ma_period=self.vol_ma_period,
                smooth_period=self.smooth_period,
                sma_fast=self.sma_fast,
                sma_slow=self.sma_slow,
            )

            # 3. Fetch option candles for execution pricing
            ce_prices: dict[datetime, float] = {}
            pe_prices: dict[datetime, float] = {}

            for opt_type in ("CE", "PE"):
                try:
                    opt_candles = self.nifty_service.get_option_candles(
                        strike=strike,
                        expiry_date=expiry,
                        option_type=opt_type,
                        start=from_dt,
                        end=to_dt,
                        interval=self.interval,
                    )
                    price_map = _build_price_map(opt_candles)
                    if opt_type == "CE":
                        ce_prices = price_map
                    else:
                        pe_prices = price_map
                    print(f"    {opt_type}: {len(opt_candles)} candles fetched")
                except Exception as exc:
                    print(f"    [{opt_type}] Fetch error: {exc}")

            # 4. Run unified state machine
            week_result = RSWeeklyExpiryResult(
                expiry_date=expiry,
                atm_strike=strike,
                nifty_open=nifty_open,
            )

            ce_trades, pe_trades = self._run_expiry(
                signal_df=signal_df,
                ce_prices=ce_prices,
                pe_prices=pe_prices,
                strike=strike,
                expiry=expiry,
            )

            week_result.ce_trades = ce_trades
            week_result.pe_trades = pe_trades

            total = len(ce_trades) + len(pe_trades)
            week_pnl = sum(t.pnl for t in week_result.all_trades)
            pnl_sign = "+" if week_pnl >= 0 else ""
            print(f"    Trades: {total} (CE:{len(ce_trades)} PE:{len(pe_trades)})"
                  f"  |  PnL: {pnl_sign}{week_pnl:.2f}")

            expiry_results.append(week_result)

        return expiry_results

    # ── Reporting ─────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(expiry_results: list[RSWeeklyExpiryResult]) -> None:
        all_trades: list[RSTradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[RSTradeResult]] = defaultdict(list)
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
                f"  {'Symbol':<22} {'Entry':>19} {'Exit':>19}"
                f" {'Entry₹':>8} {'Exit₹':>8} {'Qty':>5}"
                f" {'PnL':>10} {'Reason':<16}"
                f" {'Hist@Entry':>10} {'ADX@Exit':>8}"
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
                    f" {t.exit_reason:<16}"
                    f" {t.histogram_at_entry:>10.3f}"
                    f" {t.adx_at_exit:>8.2f}"
                )

        # Per-symbol metrics
        print(f"\n{'='*100}")
        print("  SYMBOL-WISE METRICS")
        print(f"{'='*100}")

        col_w = {"sym": 22, "trades": 7, "wins": 5, "loss": 6,
                 "wr": 6, "pnl": 12, "avg": 10, "pf": 7,
                 "best": 10, "worst": 10, "dur": 7, "cons": 5}

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
                f" {m.avg_duration_minutes:>{col_w['dur']}.1f}"
                f" {m.max_consecutive_losses:>{col_w['cons']}}"
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

        # Exit reason breakdown
        if all_trades:
            print("  EXIT REASON BREAKDOWN")
            print(f"  {'─'*50}")
            reason_counts: dict[str, int] = defaultdict(int)
            reason_pnl: dict[str, float] = defaultdict(float)
            for t in all_trades:
                reason_counts[t.exit_reason] += 1
                reason_pnl[t.exit_reason] += t.pnl
            for reason in sorted(reason_counts):
                n   = reason_counts[reason]
                pnl = reason_pnl[reason]
                print(f"  {reason:<20} {n:>4} trades   PnL: "
                      f"{'+' if pnl >= 0 else ''}{pnl:.2f}")
            print()
