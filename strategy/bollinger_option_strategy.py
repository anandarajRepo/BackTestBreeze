"""
Bollinger Band mean-reversion strategy for Nifty weekly WeeklyOptions.

Entry rules:
  CE — buy when close crosses above the lower band (oversold bounce)
  PE — buy when close crosses below the upper band (overbought reversal)

Exit rules:
  - Price reaches the middle band (mean reversion complete)
  - Square-off at 15:20 IST
  - Max 5 trades per day per symbol
  - No new entries before 9:30 or after 14:45
"""

from collections import defaultdict
from datetime import date, datetime, time
from math import floor

import pandas as pd

from models.bollinger_models import BBSymbolMetrics, BBTradeResult, BBWeeklyExpiryResult
from services.nifty_option_service import NiftyOptionService


_ENTRY_START  = time(9, 30)
_ENTRY_CUTOFF = time(14, 45)
_SQUARE_OFF   = time(15, 20)
_MAX_TRADES_PER_DAY = 5


def compute_bollinger_bands(
    candles: list[dict],
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Compute Bollinger Bands from a list of OHLC dicts.
    Returns a DataFrame with columns: datetime, close, upper, middle, lower.
    """
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    middle = df["close"].rolling(window=period).mean()
    std    = df["close"].rolling(window=period).std(ddof=0)

    result = df[["datetime", "close"]].copy()
    result["middle"] = middle.round(4)
    result["upper"]  = (middle + std_dev * std).round(4)
    result["lower"]  = (middle - std_dev * std).round(4)
    return result


def _compute_metrics(symbol: str, trades: list[BBTradeResult]) -> BBSymbolMetrics:
    if not trades:
        return BBSymbolMetrics(
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

    return BBSymbolMetrics(
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


class BollingerOptionStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        start_date: str = "",
        end_date: str = "",
        interval: str = "1minute",
    ):
        self.nifty_service = nifty_service
        self.capital       = capital
        self.bb_period     = bb_period
        self.bb_std_dev    = bb_std_dev
        self.start_date    = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date      = datetime.strptime(end_date,   "%d-%b-%Y").date()
        self.interval      = interval

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _run_symbol(
        self,
        candles: list[dict],
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[BBTradeResult]:
        """
        Run the Bollinger Band strategy on a single option contract's candle data.
        Returns a list of completed trades.
        """
        if not candles:
            return []

        indicators   = compute_bollinger_bands(candles, self.bb_period, self.bb_std_dev)
        symbol_label = f"NIFTY{strike}{option_type}"
        trades: list[BBTradeResult] = []

        daily_trade_count: dict[date, int] = defaultdict(int)

        in_position = False
        entry_row   = None
        entry_price = 0.0
        shares      = 0

        prev_close  = None
        prev_upper  = None
        prev_lower  = None

        for _, row in indicators.iterrows():
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            today: date  = dt.date()

            price  = row["close"]
            upper  = row["upper"]
            middle = row["middle"]
            lower  = row["lower"]

            if pd.isna(upper) or pd.isna(middle) or pd.isna(lower):
                prev_close = price
                prev_upper = upper
                prev_lower = lower
                continue

            # ── Exit logic ────────────────────────────────────────────────
            if in_position:
                exit_reason = None

                if t >= _SQUARE_OFF:
                    exit_reason = "SQUARE_OFF"
                elif option_type == "CE" and price >= middle:
                    # CE entered on lower-band bounce; exit when price reaches middle
                    exit_reason = "BB_REVERSAL"
                elif option_type == "PE" and price <= middle:
                    # PE entered on upper-band touch; exit when price reaches middle
                    exit_reason = "BB_REVERSAL"

                if exit_reason:
                    exit_price = price
                    duration   = int((dt - entry_row["datetime"]).total_seconds() / 60)
                    pnl        = round(shares * (exit_price - entry_price), 2)

                    trades.append(BBTradeResult(
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
                        upper_band_at_entry=entry_row["upper"],
                        middle_band_at_entry=entry_row["middle"],
                        lower_band_at_entry=entry_row["lower"],
                        upper_band_at_exit=upper,
                        middle_band_at_exit=middle,
                        lower_band_at_exit=lower,
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
                    # Price crosses above lower band from below
                    signal = (prev_close <= prev_lower) and (price > lower)
                elif option_type == "PE":
                    # Price crosses below upper band from above
                    signal = (prev_close >= prev_upper) and (price < upper)

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
            last   = indicators.iloc[-1]
            price  = last["close"]
            dt     = last["datetime"]
            duration = int((dt - entry_row["datetime"]).total_seconds() / 60)
            pnl      = round(shares * (price - entry_price), 2)

            trades.append(BBTradeResult(
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
                upper_band_at_entry=entry_row["upper"],
                middle_band_at_entry=entry_row["middle"],
                lower_band_at_entry=entry_row["lower"],
                upper_band_at_exit=last["upper"],
                middle_band_at_exit=last["middle"],
                lower_band_at_exit=last["lower"],
                duration_minutes=duration,
            ))

        return trades

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[BBWeeklyExpiryResult]:
        expiry_results: list[BBWeeklyExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        print(f"\n{'='*70}")
        print(f"  NIFTY BOLLINGER BAND STRATEGY — WEEKLY EXPIRY BACKTEST")
        print(f"  Period  : {self.start_date}  →  {self.end_date}")
        print(f"  Capital : ₹{self.capital:,.0f}  |  BB period: {self.bb_period}"
              f"  |  Std dev: {self.bb_std_dev}")
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

            week_result = BBWeeklyExpiryResult(
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
    def print_report(expiry_results: list[BBWeeklyExpiryResult]) -> None:
        all_trades: list[BBTradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[BBTradeResult]] = defaultdict(list)
        for t in all_trades:
            by_symbol[t.symbol].append(t)

        sep = "─" * 95
        print(f"\n{'='*95}")
        print("  RESULTS BY EXPIRY")
        print(f"{'='*95}")

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
                f" {'Upper':>8} {'Mid':>8} {'Lower':>8}"
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
                    f" {t.upper_band_at_exit:>8.2f}"
                    f" {t.middle_band_at_exit:>8.2f}"
                    f" {t.lower_band_at_exit:>8.2f}"
                )

        # Per-symbol metrics
        print(f"\n{'='*95}")
        print("  SYMBOL-WISE METRICS")
        print(f"{'='*95}")

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
        print(f"  {'─'*93}")

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
        print(f"  {'─'*93}")
        print(
            f"  {'OVERALL':<{col_w['sym']}} {overall_trades:>{col_w['trades']}}"
            f" {overall_wins:>{col_w['wins']}} {overall_losses:>{col_w['loss']}}"
            f" {wr_overall:>{col_w['wr']}.1f} {pnl_s:>{col_w['pnl']}}"
        )
        print(f"{'='*95}\n")
