"""
RSI + Bollinger Bands combined strategy for Nifty weekly options.

Entry rules
-----------
  Long  (CE) — both indicators oversold simultaneously:
    • Price is BELOW the lower Bollinger Band
    • RSI is BELOW the oversold threshold (default 30)

  Short (PE) — both indicators overbought simultaneously:
    • Price is ABOVE the upper Bollinger Band
    • RSI is ABOVE the overbought threshold (default 70)

Exit rules
----------
  Long  (CE) — price crosses ABOVE the upper Bollinger Band
  Short (PE) — price crosses BELOW the lower Bollinger Band
  Square-off at 15:20 IST regardless of signal
  Max 5 trades per day per symbol
  No new entries before 9:30 or after 14:45

Mode toggle
-----------
  long_only  = True  (default) — only CE trades
  short_only = True             — only PE trades
  Both False                    — CE and PE trades
"""

from collections import defaultdict
from datetime import date, datetime, time
from math import floor

import pandas as pd

from models.rsi_bb_models import (
    RSIBBSymbolMetrics,
    RSIBBTradeResult,
    RSIBBWeeklyExpiryResult,
)
from services.nifty_option_service import NiftyOptionService


_ENTRY_START        = time(9, 30)
_ENTRY_CUTOFF       = time(14, 45)
_SQUARE_OFF         = time(15, 20)
_MAX_TRADES_PER_DAY = 5


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    return (100 - (100 / (1 + rs))).round(4)


def compute_indicators(
    candles: list[dict],
    rsi_period: int,
    bb_period: int,
    bb_std_dev: float,
) -> pd.DataFrame:
    """
    Returns DataFrame with: datetime, close, rsi, upper, middle, lower.
    """
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    middle = df["close"].rolling(window=bb_period).mean()
    std    = df["close"].rolling(window=bb_period).std(ddof=0)

    result = df[["datetime", "close"]].copy()
    result["rsi"]    = compute_rsi(df["close"], rsi_period)
    result["middle"] = middle.round(4)
    result["upper"]  = (middle + bb_std_dev * std).round(4)
    result["lower"]  = (middle - bb_std_dev * std).round(4)
    return result


def _compute_metrics(symbol: str, trades: list[RSIBBTradeResult]) -> RSIBBSymbolMetrics:
    if not trades:
        return RSIBBSymbolMetrics(
            symbol=symbol, total_trades=0, wins=0, losses=0,
            win_rate=0.0, total_pnl=0.0, avg_pnl=0.0,
            profit_factor=0.0, best_trade=0.0, worst_trade=0.0,
            avg_duration_minutes=0.0, max_consecutive_losses=0,
        )

    pnls   = [t.pnl for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit  = sum(wins)
    gross_loss    = abs(sum(losses))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss else float("inf")

    max_consec = cur_consec = 0
    for p in pnls:
        if p <= 0:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    return RSIBBSymbolMetrics(
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


class RSIBBOptionStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        long_only: bool = True,
        short_only: bool = False,
        start_date: str = "",
        end_date: str = "",
        interval: str = "1minute",
    ):
        if long_only and short_only:
            raise ValueError("long_only and short_only cannot both be True.")

        self.nifty_service  = nifty_service
        self.capital        = capital
        self.rsi_period     = rsi_period
        self.rsi_oversold   = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period      = bb_period
        self.bb_std_dev     = bb_std_dev
        self.long_only      = long_only
        self.short_only     = short_only
        self.start_date     = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date       = datetime.strptime(end_date,   "%d-%b-%Y").date()
        self.interval       = interval

    @property
    def _active_option_types(self) -> list[str]:
        if self.long_only:
            return ["CE"]
        if self.short_only:
            return ["PE"]
        return ["CE", "PE"]

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _run_symbol(
        self,
        candles: list[dict],
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[RSIBBTradeResult]:
        if not candles:
            return []

        df           = compute_indicators(candles, self.rsi_period, self.bb_period, self.bb_std_dev)
        symbol_label = f"NIFTY{strike}{option_type}"
        trades: list[RSIBBTradeResult] = []

        daily_trade_count: dict[date, int] = defaultdict(int)

        in_position = False
        entry_row   = None
        entry_price = 0.0
        shares      = 0

        prev_close = None
        prev_upper = None
        prev_lower = None

        for _, row in df.iterrows():
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            today: date  = dt.date()

            price  = row["close"]
            rsi    = row["rsi"]
            upper  = row["upper"]
            middle = row["middle"]
            lower  = row["lower"]

            if pd.isna(rsi) or pd.isna(upper) or pd.isna(middle) or pd.isna(lower):
                prev_close = price
                prev_upper = upper
                prev_lower = lower
                continue

            # ── Exit logic ────────────────────────────────────────────────
            if in_position:
                exit_reason = None

                if t >= _SQUARE_OFF:
                    exit_reason = "SQUARE_OFF"
                elif option_type == "CE" and prev_close is not None:
                    # Long exit: price crosses above upper band
                    if prev_close <= prev_upper and price > upper:
                        exit_reason = "BB_UPPER_EXIT"
                elif option_type == "PE" and prev_close is not None:
                    # Short exit: price crosses below lower band
                    if prev_close >= prev_lower and price < lower:
                        exit_reason = "BB_LOWER_EXIT"

                if exit_reason:
                    exit_price = price
                    duration   = int((dt - entry_row["datetime"]).total_seconds() / 60)
                    pnl        = round(shares * (exit_price - entry_price), 2)

                    trades.append(RSIBBTradeResult(
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
                        rsi_at_entry=entry_row["rsi"],
                        bb_upper_at_entry=entry_row["upper"],
                        bb_middle_at_entry=entry_row["middle"],
                        bb_lower_at_entry=entry_row["lower"],
                        rsi_at_exit=round(float(rsi), 4),
                        bb_upper_at_exit=round(float(upper), 4),
                        bb_middle_at_exit=round(float(middle), 4),
                        bb_lower_at_exit=round(float(lower), 4),
                        duration_minutes=duration,
                    ))

                    in_position = False
                    entry_row   = None

            # ── Entry logic ───────────────────────────────────────────────
            if (
                not in_position
                and prev_close is not None
                and _ENTRY_START <= t <= _ENTRY_CUTOFF
                and daily_trade_count[today] < _MAX_TRADES_PER_DAY
                and price > 0
            ):
                signal = False

                if option_type == "CE":
                    # Both RSI and price confirm oversold: price below lower BB and RSI below oversold
                    signal = (price < lower) and (rsi < self.rsi_oversold)
                elif option_type == "PE":
                    # Both RSI and price confirm overbought: price above upper BB and RSI above overbought
                    signal = (price > upper) and (rsi > self.rsi_overbought)

                if signal:
                    entry_price = price
                    shares      = max(floor(self.capital / entry_price), 1)
                    in_position = True
                    entry_row   = row
                    daily_trade_count[today] += 1

            prev_close = price
            prev_upper = upper
            prev_lower = lower

        # Force-close any open position at end of data
        if in_position and entry_row is not None:
            last      = df.iloc[-1]
            price     = last["close"]
            dt        = last["datetime"]
            duration  = int((dt - entry_row["datetime"]).total_seconds() / 60)
            pnl       = round(shares * (price - entry_price), 2)

            trades.append(RSIBBTradeResult(
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
                rsi_at_entry=entry_row["rsi"],
                bb_upper_at_entry=entry_row["upper"],
                bb_middle_at_entry=entry_row["middle"],
                bb_lower_at_entry=entry_row["lower"],
                rsi_at_exit=round(float(last["rsi"]), 4) if not pd.isna(last["rsi"]) else 0.0,
                bb_upper_at_exit=round(float(last["upper"]), 4) if not pd.isna(last["upper"]) else 0.0,
                bb_middle_at_exit=round(float(last["middle"]), 4) if not pd.isna(last["middle"]) else 0.0,
                bb_lower_at_exit=round(float(last["lower"]), 4) if not pd.isna(last["lower"]) else 0.0,
                duration_minutes=duration,
            ))

        return trades

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[RSIBBWeeklyExpiryResult]:
        expiry_results: list[RSIBBWeeklyExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)

        mode_label = "LONG-ONLY" if self.long_only else ("SHORT-ONLY" if self.short_only else "LONG & SHORT")

        print(f"\n{'='*75}")
        print(f"  NIFTY RSI + BOLLINGER BANDS — WEEKLY EXPIRY BACKTEST")
        print(f"  Period    : {self.start_date}  →  {self.end_date}")
        print(f"  Mode      : {mode_label}")
        print(f"  Capital   : ₹{self.capital:,.0f}")
        print(f"  RSI period: {self.rsi_period}  |  Oversold: {self.rsi_oversold}"
              f"  |  Overbought: {self.rsi_overbought}")
        print(f"  BB period : {self.bb_period}  |  Std dev: {self.bb_std_dev}")
        print(f"  Expiries found: {len(wednesdays)}")
        print(f"{'='*75}\n")

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

            week_result = RSIBBWeeklyExpiryResult(
                expiry_date=expiry,
                atm_strike=strike,
                nifty_open=nifty_open,
            )

            from_dt = datetime(win_start.year, win_start.month, win_start.day, 9, 15, 0)
            to_dt   = datetime(win_end.year,   win_end.month,   win_end.day,   15, 30, 0)

            for opt_type in self._active_option_types:
                try:
                    candles = self.nifty_service.get_option_candles(
                        strike=strike,
                        expiry_date=expiry,
                        option_type=opt_type,
                        start=from_dt,
                        end=to_dt,
                        interval=self.interval,
                    )
                    trades = self._run_symbol(candles, opt_type, strike, expiry)

                    if opt_type == "CE":
                        week_result.ce_trades = trades
                    else:
                        week_result.pe_trades = trades

                    print(f"    {opt_type}: {len(trades)} trades")
                except Exception as exc:
                    print(f"    [{opt_type}] Error: {exc}")

            expiry_results.append(week_result)

        return expiry_results

    # ── Reporting ─────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(expiry_results: list[RSIBBWeeklyExpiryResult]) -> None:
        all_trades: list[RSIBBTradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[RSIBBTradeResult]] = defaultdict(list)
        for t in all_trades:
            by_symbol[t.symbol].append(t)

        sep = "─" * 110
        print(f"\n{'='*110}")
        print("  RESULTS BY EXPIRY")
        print(f"{'='*110}")

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
                f" {'RSI-In':>7} {'RSI-Out':>7}"
                f" {'BB-Lo':>8} {'BB-Up':>8}"
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
                    f" {t.rsi_at_entry:>7.2f}"
                    f" {t.rsi_at_exit:>7.2f}"
                    f" {t.bb_lower_at_entry:>8.2f}"
                    f" {t.bb_upper_at_entry:>8.2f}"
                )

        # Per-symbol metrics
        print(f"\n{'='*110}")
        print("  SYMBOL-WISE METRICS")
        print(f"{'='*110}")

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
        print(f"  {'─'*108}")

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
        print(f"  {'─'*108}")
        print(
            f"  {'OVERALL':<{col_w['sym']}} {overall_trades:>{col_w['trades']}}"
            f" {overall_wins:>{col_w['wins']}} {overall_losses:>{col_w['loss']}}"
            f" {wr_overall:>{col_w['wr']}.1f} {pnl_s:>{col_w['pnl']}}"
        )
        print(f"{'='*110}\n")
