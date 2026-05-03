"""
RSI Convergence & Divergence strategy for Nifty weekly options.

Signals
-------
Bullish divergence  → buy CE
  Price makes a lower low while RSI makes a higher low over the lookback window.

Bearish divergence  → buy PE
  Price makes a higher high while RSI makes a lower high over the lookback window.

Bullish convergence → buy CE
  RSI was below the oversold threshold and crosses back above it (momentum
  aligning with a potential upswing).

Bearish convergence → buy PE
  RSI was above the overbought threshold and crosses back below it (momentum
  aligning with a potential downswing).

Exit rules
----------
  - RSI reaches the neutral zone (crosses 50 from either side)
  - Square-off at 15:20 IST
  - Max 5 trades per day per symbol
  - No new entries before 9:30 or after 14:45
"""

from collections import defaultdict
from datetime import date, datetime, time
from math import floor

import pandas as pd

from models.rsi_models import RSISymbolMetrics, RSITradeResult, RSIWeeklyExpiryResult
from services.nifty_option_service import NiftyOptionService


_ENTRY_START  = time(9, 30)
_ENTRY_CUTOFF = time(14, 45)
_SQUARE_OFF   = time(15, 20)
_MAX_TRADES_PER_DAY = 5


def compute_rsi(candles: list[dict], period: int = 14) -> pd.DataFrame:
    """
    Compute RSI from a list of OHLC dicts.
    Returns a DataFrame with columns: datetime, close, rsi.
    """
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["close"] = df["close"].astype(float)

    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs  = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    # When avg_loss is 0 the RSI is 100 by definition
    rsi = rsi.fillna(avg_loss.apply(lambda x: 100.0 if x == 0 else float("nan")))

    result = df[["datetime", "close"]].copy()
    result["rsi"] = rsi.round(4)
    return result


def _detect_divergence(
    prices: pd.Series,
    rsis: pd.Series,
    lookback: int,
) -> str | None:
    """
    Scan the last `lookback` bars for a divergence pattern.

    Returns "BULLISH_DIV", "BEARISH_DIV", or None.

    Bullish divergence : latest close < min close in window  AND
                         latest RSI   > min RSI   in window
    Bearish divergence : latest close > max close in window  AND
                         latest RSI   < max RSI   in window
    """
    if len(prices) < lookback + 1:
        return None

    window_prices = prices.iloc[-(lookback + 1):-1]
    window_rsis   = rsis.iloc[-(lookback + 1):-1]
    cur_price     = prices.iloc[-1]
    cur_rsi       = rsis.iloc[-1]

    if pd.isna(cur_rsi) or window_rsis.isna().any():
        return None

    if cur_price < window_prices.min() and cur_rsi > window_rsis.min():
        return "BULLISH_DIV"
    if cur_price > window_prices.max() and cur_rsi < window_rsis.max():
        return "BEARISH_DIV"
    return None


def _compute_metrics(symbol: str, trades: list[RSITradeResult]) -> RSISymbolMetrics:
    if not trades:
        return RSISymbolMetrics(
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

    return RSISymbolMetrics(
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


class RSIOptionStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        divergence_lookback: int = 5,
        start_date: str = "",
        end_date: str = "",
        interval: str = "1minute",
    ):
        self.nifty_service        = nifty_service
        self.capital              = capital
        self.rsi_period           = rsi_period
        self.oversold             = oversold
        self.overbought           = overbought
        self.divergence_lookback  = divergence_lookback
        self.start_date           = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date             = datetime.strptime(end_date,   "%d-%b-%Y").date()
        self.interval             = interval

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _run_symbol(
        self,
        candles: list[dict],
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[RSITradeResult]:
        if not candles:
            return []

        indicators   = compute_rsi(candles, self.rsi_period)
        symbol_label = f"NIFTY{strike}{option_type}"
        trades: list[RSITradeResult] = []
        daily_trade_count: dict[date, int] = defaultdict(int)

        in_position = False
        entry_row   = None
        entry_price = 0.0
        entry_signal_type = ""
        shares      = 0

        prev_rsi = None

        for i, row in indicators.iterrows():
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            today: date  = dt.date()

            price = row["close"]
            rsi   = row["rsi"]

            if pd.isna(rsi):
                prev_rsi = rsi
                continue

            # ── Exit logic ────────────────────────────────────────────────
            if in_position:
                exit_reason = None

                if t >= _SQUARE_OFF:
                    exit_reason = "SQUARE_OFF"
                elif option_type == "CE" and prev_rsi is not None and prev_rsi < 50 and rsi >= 50:
                    exit_reason = "RSI_NEUTRAL"
                elif option_type == "PE" and prev_rsi is not None and prev_rsi > 50 and rsi <= 50:
                    exit_reason = "RSI_NEUTRAL"

                if exit_reason:
                    exit_price = price
                    duration   = int((dt - entry_row["datetime"]).total_seconds() / 60)
                    pnl        = round(shares * (exit_price - entry_price), 2)

                    trades.append(RSITradeResult(
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
                        signal_type=entry_signal_type,
                        rsi_at_entry=entry_row["rsi"],
                        rsi_at_exit=rsi,
                        price_at_entry=entry_row["close"],
                        price_at_exit=exit_price,
                        duration_minutes=duration,
                    ))

                    in_position = False
                    entry_row   = None

            # ── Entry logic ───────────────────────────────────────────────
            if (
                not in_position
                and prev_rsi is not None
                and _ENTRY_START <= t <= _ENTRY_CUTOFF
                and daily_trade_count[today] < _MAX_TRADES_PER_DAY
                and price > 0
            ):
                signal: str | None = None

                # Convergence signals
                if option_type == "CE" and prev_rsi < self.oversold and rsi >= self.oversold:
                    signal = "CONVERGENCE"
                elif option_type == "PE" and prev_rsi > self.overbought and rsi <= self.overbought:
                    signal = "CONVERGENCE"

                # Divergence signals (only if no convergence signal)
                if signal is None and i >= self.divergence_lookback:
                    price_window = indicators.loc[: i, "close"]
                    rsi_window   = indicators.loc[: i, "rsi"]
                    div = _detect_divergence(price_window, rsi_window, self.divergence_lookback)
                    if div == "BULLISH_DIV" and option_type == "CE":
                        signal = "DIVERGENCE"
                    elif div == "BEARISH_DIV" and option_type == "PE":
                        signal = "DIVERGENCE"

                if signal:
                    entry_price       = price
                    shares            = max(floor(self.capital / entry_price), 1)
                    in_position       = True
                    entry_row         = row
                    entry_signal_type = signal
                    daily_trade_count[today] += 1

            prev_rsi = rsi

        # Force-close any open position at end of data
        if in_position and entry_row is not None:
            last       = indicators.iloc[-1]
            price      = last["close"]
            dt         = last["datetime"]
            duration   = int((dt - entry_row["datetime"]).total_seconds() / 60)
            pnl        = round(shares * (price - entry_price), 2)

            trades.append(RSITradeResult(
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
                signal_type=entry_signal_type,
                rsi_at_entry=entry_row["rsi"],
                rsi_at_exit=last["rsi"],
                price_at_entry=entry_row["close"],
                price_at_exit=price,
                duration_minutes=duration,
            ))

        return trades

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[RSIWeeklyExpiryResult]:
        expiry_results: list[RSIWeeklyExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        print(f"\n{'='*70}")
        print(f"  NIFTY RSI CONVERGENCE/DIVERGENCE STRATEGY — WEEKLY EXPIRY BACKTEST")
        print(f"  Period     : {self.start_date}  →  {self.end_date}")
        print(f"  Capital    : ₹{self.capital:,.0f}  |  RSI period: {self.rsi_period}")
        print(f"  Oversold   : {self.oversold}  |  Overbought: {self.overbought}")
        print(f"  Div lookback: {self.divergence_lookback} bars")
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

            week_result = RSIWeeklyExpiryResult(
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
    def print_report(expiry_results: list[RSIWeeklyExpiryResult]) -> None:
        all_trades: list[RSITradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[RSITradeResult]] = defaultdict(list)
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
                f" {'PnL':>10} {'Reason':<14} {'Signal':<12}"
                f" {'RSI-In':>7} {'RSI-Out':>7}"
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
                    f" {t.signal_type:<12}"
                    f" {t.rsi_at_entry:>7.2f}"
                    f" {t.rsi_at_exit:>7.2f}"
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
