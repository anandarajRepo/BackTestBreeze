"""
ADX DI+/DI- crossover strategy for MCX commodity monthly options.

Entry rules:
  CE — buy when DI+ crosses above DI- and ADX >= threshold
  PE — buy when DI- crosses above DI+ and ADX >= threshold

Exit rules:
  - DI direction reversal (crossover flips)
  - Square-off at 23:25 IST (MCX evening session close)
  - Max 5 trades per day per symbol
  - No new entries before 09:00 or after 22:45
"""

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from math import floor

import pandas as pd

from models.commodity_models import (
    CommoditySymbolMetrics,
    CommodityTradeResult,
    MonthlyExpiryResult,
)
from services.commodity_option_service import COMMODITY_CONFIG, CommodityOptionService


_ENTRY_START  = time(9, 0)
_ENTRY_CUTOFF = time(22, 45)
_SQUARE_OFF   = time(23, 25)
_MAX_TRADES_PER_DAY = 5


def _wilder_ewm(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_adx(candles: list[dict], period: int = 14) -> pd.DataFrame:
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

    result = df[["datetime"]].copy()
    result["adx"]      = adx.round(4)
    result["di_plus"]  = di_plus.round(4)
    result["di_minus"] = di_minus.round(4)
    result["close"]    = df["close"]
    return result


def _compute_metrics(
    symbol: str, trades: list[CommodityTradeResult]
) -> CommoditySymbolMetrics:
    if not trades:
        return CommoditySymbolMetrics(
            symbol=symbol, total_trades=0, wins=0, losses=0,
            win_rate=0.0, total_pnl=0.0, avg_pnl=0.0,
            profit_factor=0.0, best_trade=0.0, worst_trade=0.0,
            avg_duration_minutes=0.0, max_consecutive_losses=0,
        )

    pnls = [t.pnl for t in trades]
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

    return CommoditySymbolMetrics(
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


class ADXCommodityStrategy:
    def __init__(
        self,
        commodity_service: CommodityOptionService,
        commodities: list[str],
        capital: float = 100_000.0,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        start_date: str = "",
        end_date: str = "",
        interval: str = "1minute",
    ):
        self.commodity_service = commodity_service
        self.commodities       = commodities
        self.capital           = capital
        self.adx_period        = adx_period
        self.adx_threshold     = adx_threshold
        self.start_date        = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date          = datetime.strptime(end_date,   "%d-%b-%Y").date()
        self.interval          = interval

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _run_symbol(
        self,
        candles: list[dict],
        commodity: str,
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[CommodityTradeResult]:
        if not candles:
            return []

        indicators   = compute_adx(candles, self.adx_period)
        symbol_label = f"{commodity}{strike}{option_type}"
        trades: list[CommodityTradeResult] = []

        daily_trade_count: dict[date, int] = defaultdict(int)

        in_position   = False
        entry_row     = None
        entry_price   = 0.0
        shares        = 0

        prev_di_plus  = None
        prev_di_minus = None

        for _, row in indicators.iterrows():
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            today: date  = dt.date()

            adx      = row["adx"]
            di_plus  = row["di_plus"]
            di_minus = row["di_minus"]
            price    = row["close"]

            if pd.isna(adx) or pd.isna(di_plus) or pd.isna(di_minus):
                prev_di_plus  = di_plus
                prev_di_minus = di_minus
                continue

            # ── Exit logic ────────────────────────────────────────────────
            if in_position:
                exit_reason = None

                if t >= _SQUARE_OFF:
                    exit_reason = "SQUARE_OFF"
                elif option_type == "CE" and prev_di_plus is not None:
                    if prev_di_plus >= prev_di_minus and di_plus < di_minus:
                        exit_reason = "ADX_CROSSOVER"
                elif option_type == "PE" and prev_di_plus is not None:
                    if prev_di_minus >= prev_di_plus and di_minus < di_plus:
                        exit_reason = "ADX_CROSSOVER"

                if exit_reason:
                    exit_price = price
                    duration   = int((dt - entry_row["datetime"]).total_seconds() / 60)
                    pnl        = round(shares * (exit_price - entry_price), 2)

                    trades.append(CommodityTradeResult(
                        symbol=symbol_label,
                        commodity=commodity,
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
            if (
                not in_position
                and prev_di_plus is not None
                and _ENTRY_START <= t <= _ENTRY_CUTOFF
                and adx >= self.adx_threshold
                and daily_trade_count[today] < _MAX_TRADES_PER_DAY
            ):
                signal = False
                if option_type == "CE":
                    signal = (prev_di_plus <= prev_di_minus) and (di_plus > di_minus)
                elif option_type == "PE":
                    signal = (prev_di_minus <= prev_di_plus) and (di_minus > di_plus)

                if signal and price > 0:
                    entry_price = price
                    shares      = max(floor(self.capital / entry_price), 1)
                    in_position = True
                    entry_row   = row
                    daily_trade_count[today] += 1

            prev_di_plus  = di_plus
            prev_di_minus = di_minus

        # Force-close any open position at end of data
        if in_position and entry_row is not None:
            last     = indicators.iloc[-1]
            price    = last["close"]
            dt       = last["datetime"]
            duration = int((dt - entry_row["datetime"]).total_seconds() / 60)
            pnl      = round(shares * (price - entry_price), 2)

            trades.append(CommodityTradeResult(
                symbol=symbol_label,
                commodity=commodity,
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

    # ── Monthly expiry orchestration ──────────────────────────────────────────

    def run_monthly_backtest(self) -> list[MonthlyExpiryResult]:
        expiry_results: list[MonthlyExpiryResult] = []

        expiries = CommodityOptionService.monthly_expiries(self.start_date, self.end_date)

        print(f"\n{'='*75}")
        print(f"  COMMODITY ADX STRATEGY — MCX MONTHLY EXPIRY BACKTEST")
        print(f"  Period     : {self.start_date}  →  {self.end_date}")
        print(f"  Commodities: {', '.join(self.commodities)}")
        print(f"  Capital    : ₹{self.capital:,.0f}  |  ADX period: {self.adx_period}"
              f"  |  ADX threshold: {self.adx_threshold}")
        print(f"  Expiries found: {len(expiries)}")
        print(f"{'='*75}\n")

        for expiry in expiries:
            win_start, win_end = CommodityOptionService.month_window(expiry)

            for commodity in self.commodities:
                _, strike_interval = COMMODITY_CONFIG[commodity]

                # Use first trading day of the month to get opening price for ATM
                try:
                    commodity_open = self.commodity_service.get_commodity_open(
                        commodity, win_start
                    )
                except Exception as exc:
                    print(f"  [{expiry}][{commodity}] Could not get open for {win_start}: {exc}")
                    continue

                strike = CommodityOptionService.atm_strike(commodity_open, strike_interval)

                print(
                    f"  Expiry {expiry}  |  {commodity:<12}"
                    f"  Open {commodity_open:.2f}  |  ATM {strike}"
                )

                month_result = MonthlyExpiryResult(
                    expiry_date=expiry,
                    commodity=commodity,
                    atm_strike=strike,
                    commodity_open=commodity_open,
                )

                from_dt = datetime(win_start.year, win_start.month, win_start.day, 9, 0, 0)
                to_dt   = datetime(win_end.year,   win_end.month,   win_end.day,   23, 30, 0)

                for opt_type in ("CE", "PE"):
                    try:
                        candles = self.commodity_service.get_option_candles(
                            stock_code=commodity,
                            strike=strike,
                            expiry_date=expiry,
                            option_type=opt_type,
                            start=from_dt,
                            end=to_dt,
                            interval=self.interval,
                        )
                        trades = self._run_symbol(candles, commodity, opt_type, strike, expiry)

                        if opt_type == "CE":
                            month_result.ce_trades = trades
                        else:
                            month_result.pe_trades = trades

                        print(f"    {opt_type}: {len(trades)} trades")
                    except Exception as exc:
                        print(f"    [{opt_type}] Error: {exc}")

                expiry_results.append(month_result)

        return expiry_results

    # ── Reporting ─────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(expiry_results: list[MonthlyExpiryResult]) -> None:
        all_trades: list[CommodityTradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[CommodityTradeResult]] = defaultdict(list)
        for t in all_trades:
            by_symbol[t.symbol].append(t)

        sep = "─" * 95

        # ── Results by expiry ─────────────────────────────────────────────
        print(f"\n{'='*95}")
        print("  RESULTS BY EXPIRY")
        print(f"{'='*95}")

        for er in expiry_results:
            if not er.all_trades:
                continue
            total_pnl = sum(t.pnl for t in er.all_trades)
            sign      = "+" if total_pnl >= 0 else ""
            print(
                f"\n  Expiry {er.expiry_date}  |  {er.commodity:<12}"
                f"  ATM {er.atm_strike}  |  Open {er.commodity_open:.2f}"
                f"  |  Trades {len(er.all_trades)}"
                f"  |  PnL {sign}{total_pnl:.2f}"
            )
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

        # ── Per-symbol metrics ────────────────────────────────────────────
        print(f"\n{'='*95}")
        print("  SYMBOL-WISE METRICS")
        print(f"{'='*95}")

        col_w = {"sym": 22, "trades": 7, "wins": 5, "loss": 6,
                 "wr": 6, "pnl": 11, "avg": 10, "pf": 7,
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
                f" {m.avg_duration_minutes:>{col_w['dur']}.1f}"
                f" {m.max_consecutive_losses:>{col_w['cons']}}"
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

        # ── Per-commodity summary ─────────────────────────────────────────
        print(f"{'='*95}")
        print("  COMMODITY SUMMARY")
        print(f"{'='*95}")

        by_commodity: dict[str, list[CommodityTradeResult]] = defaultdict(list)
        for t in all_trades:
            by_commodity[t.commodity].append(t)

        print(f"  {'Commodity':<14} {'Trades':>7} {'Wins':>5} {'Loss':>6}"
              f" {'Win%':>6} {'Total PnL':>11}")
        print(f"  {'─'*55}")

        for commodity, trades in sorted(by_commodity.items()):
            m = _compute_metrics(commodity, trades)
            pnl_s = f"{'+' if m.total_pnl >= 0 else ''}{m.total_pnl:.2f}"
            print(
                f"  {commodity:<14} {m.total_trades:>7} {m.wins:>5} {m.losses:>6}"
                f" {m.win_rate:>6.1f} {pnl_s:>11}"
            )

        print(f"{'='*95}\n")
