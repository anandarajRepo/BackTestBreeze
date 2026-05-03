"""
RSI Convergence & Divergence strategy for Nifty weekly options.

Signal types
------------
DIVERGENCE
  Bullish divergence  → CE:  price makes a lower low  but RSI makes a higher low  (oversold area)
  Bearish divergence  → PE:  price makes a higher high but RSI makes a lower high  (overbought area)

CONVERGENCE  (RSI extreme mean-reversion)
  RSI crosses back above RSI_OVERSOLD  (was below, now above)  → CE
  RSI crosses back below RSI_OVERBOUGHT (was above, now below) → PE

Exit rules
----------
  - RSI returns to neutral zone (crosses RSI_EXIT_LEVEL from either side)
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


_ENTRY_START        = time(9, 30)
_ENTRY_CUTOFF       = time(14, 45)
_SQUARE_OFF         = time(15, 20)
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

    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))

    result = df[["datetime", "close"]].copy()
    result["rsi"] = rsi.round(4)
    return result


def _detect_bullish_divergence(
    prices: pd.Series,
    rsis: pd.Series,
    lookback: int,
) -> bool:
    """
    True when price made a lower low but RSI made a higher low over the last `lookback` bars.
    Requires at least two swing lows separated by at least lookback//3 bars.
    """
    if len(prices) < lookback:
        return False

    p = prices.iloc[-lookback:].reset_index(drop=True)
    r = rsis.iloc[-lookback:].reset_index(drop=True)

    # Find the two lowest price points
    first_low_idx  = int(p.iloc[: lookback // 2].idxmin())
    second_low_idx = int(p.iloc[lookback // 2 :].idxmin()) + lookback // 2

    if first_low_idx >= second_low_idx:
        return False

    price_lower_low = p.iloc[second_low_idx] < p.iloc[first_low_idx]
    rsi_higher_low  = r.iloc[second_low_idx] > r.iloc[first_low_idx]

    return bool(price_lower_low and rsi_higher_low)


def _detect_bearish_divergence(
    prices: pd.Series,
    rsis: pd.Series,
    lookback: int,
) -> bool:
    """
    True when price made a higher high but RSI made a lower high over the last `lookback` bars.
    """
    if len(prices) < lookback:
        return False

    p = prices.iloc[-lookback:].reset_index(drop=True)
    r = rsis.iloc[-lookback:].reset_index(drop=True)

    first_high_idx  = int(p.iloc[: lookback // 2].idxmax())
    second_high_idx = int(p.iloc[lookback // 2 :].idxmax()) + lookback // 2

    if first_high_idx >= second_high_idx:
        return False

    price_higher_high = p.iloc[second_high_idx] > p.iloc[first_high_idx]
    rsi_lower_high    = r.iloc[second_high_idx] < r.iloc[first_high_idx]

    return bool(price_higher_high and rsi_lower_high)


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
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        rsi_exit_level: float = 50.0,
        divergence_lookback: int = 20,
        start_date: str = "",
        end_date: str = "",
        interval: str = "1minute",
    ):
        self.nifty_service       = nifty_service
        self.capital             = capital
        self.rsi_period          = rsi_period
        self.rsi_oversold        = rsi_oversold
        self.rsi_overbought      = rsi_overbought
        self.rsi_exit_level      = rsi_exit_level
        self.divergence_lookback = divergence_lookback
        self.start_date          = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date            = datetime.strptime(end_date,   "%d-%b-%Y").date()
        self.interval            = interval

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

        df           = compute_rsi(candles, self.rsi_period)
        symbol_label = f"NIFTY{strike}{option_type}"
        trades: list[RSITradeResult] = []

        daily_trade_count: dict[date, int] = defaultdict(int)

        in_position  = False
        entry_row    = None
        entry_price  = 0.0
        shares       = 0
        signal_type  = ""

        prev_rsi = None

        for i, row in df.iterrows():
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
                elif option_type == "CE" and not pd.isna(prev_rsi):
                    # CE: exit when RSI crosses above exit level (neutral)
                    if prev_rsi < self.rsi_exit_level <= rsi:
                        exit_reason = "RSI_NEUTRAL"
                elif option_type == "PE" and not pd.isna(prev_rsi):
                    # PE: exit when RSI crosses below exit level (neutral)
                    if prev_rsi > self.rsi_exit_level >= rsi:
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
                        signal_type=signal_type,
                        rsi_at_entry=entry_row["rsi"],
                        rsi_at_exit=round(float(rsi), 4),
                        price_at_entry=entry_row["close"],
                        duration_minutes=duration,
                    ))

                    in_position = False
                    entry_row   = None

            # ── Entry logic ───────────────────────────────────────────────
            if (
                not in_position
                and prev_rsi is not None
                and not pd.isna(prev_rsi)
                and _ENTRY_START <= t <= _ENTRY_CUTOFF
                and daily_trade_count[today] < _MAX_TRADES_PER_DAY
                and price > 0
                and i >= self.divergence_lookback
            ):
                signal      = False
                sig_type    = ""

                prices_window = df["close"].iloc[max(0, i - self.divergence_lookback): i + 1]
                rsi_window    = df["rsi"].iloc[max(0, i - self.divergence_lookback): i + 1]

                if option_type == "CE":
                    # Convergence: RSI crosses back above oversold level
                    convergence = (prev_rsi < self.rsi_oversold) and (rsi >= self.rsi_oversold)
                    # Divergence: bullish divergence with RSI in oversold territory
                    divergence  = (
                        rsi < self.rsi_oversold + 10
                        and _detect_bullish_divergence(prices_window, rsi_window, self.divergence_lookback)
                    )
                    if convergence:
                        signal   = True
                        sig_type = "CONVERGENCE"
                    elif divergence:
                        signal   = True
                        sig_type = "DIVERGENCE"

                elif option_type == "PE":
                    # Convergence: RSI crosses back below overbought level
                    convergence = (prev_rsi > self.rsi_overbought) and (rsi <= self.rsi_overbought)
                    # Divergence: bearish divergence with RSI in overbought territory
                    divergence  = (
                        rsi > self.rsi_overbought - 10
                        and _detect_bearish_divergence(prices_window, rsi_window, self.divergence_lookback)
                    )
                    if convergence:
                        signal   = True
                        sig_type = "CONVERGENCE"
                    elif divergence:
                        signal   = True
                        sig_type = "DIVERGENCE"

                if signal:
                    entry_price  = price
                    shares       = max(floor(self.capital / entry_price), 1)
                    in_position  = True
                    entry_row    = row
                    signal_type  = sig_type
                    daily_trade_count[today] += 1

            prev_rsi = rsi

        # Force-close any open position at end of data
        if in_position and entry_row is not None:
            last      = df.iloc[-1]
            price     = last["close"]
            dt        = last["datetime"]
            duration  = int((dt - entry_row["datetime"]).total_seconds() / 60)
            pnl       = round(shares * (price - entry_price), 2)

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
                signal_type=signal_type,
                rsi_at_entry=entry_row["rsi"],
                rsi_at_exit=round(float(last["rsi"]), 4) if not pd.isna(last["rsi"]) else 0.0,
                price_at_entry=entry_row["close"],
                duration_minutes=duration,
            ))

        return trades

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[RSIWeeklyExpiryResult]:
        expiry_results: list[RSIWeeklyExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        print(f"\n{'='*70}")
        print(f"  NIFTY RSI CONVERGENCE & DIVERGENCE — WEEKLY EXPIRY BACKTEST")
        print(f"  Period     : {self.start_date}  →  {self.end_date}")
        print(f"  Capital    : ₹{self.capital:,.0f}  |  RSI period: {self.rsi_period}")
        print(f"  Oversold   : {self.rsi_oversold}  |  Overbought: {self.rsi_overbought}"
              f"  |  Exit level: {self.rsi_exit_level}")
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

        sep = "─" * 105
        print(f"\n{'='*105}")
        print("  RESULTS BY EXPIRY")
        print(f"{'='*105}")

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
                f" {'PnL':>10} {'Reason':<12} {'Signal':<12}"
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
                    f" {t.exit_reason:<12}"
                    f" {t.signal_type:<12}"
                    f" {t.rsi_at_entry:>7.2f}"
                    f" {t.rsi_at_exit:>7.2f}"
                )

        # Per-symbol metrics
        print(f"\n{'='*105}")
        print("  SYMBOL-WISE METRICS")
        print(f"{'='*105}")

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
        print(f"  {'─'*103}")

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
        print(f"  {'─'*103}")
        print(
            f"  {'OVERALL':<{col_w['sym']}} {overall_trades:>{col_w['trades']}}"
            f" {overall_wins:>{col_w['wins']}} {overall_losses:>{col_w['loss']}}"
            f" {wr_overall:>{col_w['wr']}.1f} {pnl_s:>{col_w['pnl']}}"
        )
        print(f"{'='*105}\n")
