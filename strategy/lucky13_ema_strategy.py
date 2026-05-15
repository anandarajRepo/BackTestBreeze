"""
Lucky13EMA — 1-Minute Momentum Scalper for Nifty Weekly Options.

Signal source: Nifty 50 spot (1-min candles)
Instrument   : Nifty ATM CE / PE (weekly expiry)

Entry rules:
  CE — first green candle (close > open) closing above 13 EMA, after close was below EMA
  PE — first red  candle (close < open) closing below 13 EMA, after close was above EMA

Triple filter (each toggleable via use_filters):
  1. Volume > 20-period SMA × vol_multiplier
  2. VWAP alignment  (CE: close > VWAP or VWAP cross, PE: close < VWAP or VWAP cross)
  3. 5-min EMA bias  (CE: spot above 5-min EMA, PE: spot below 5-min EMA)

Exit rules (option price based):
  - Profit target : option price >= entry × (1 + profit_pct / 100)
  - Stop loss     : option price <= entry × (1 - stop_pct  / 100)
  - Trailing stop : option price drops trailing_stop_pts below its peak (0 = disabled)
  - EMA reversal  : spot crosses back through 13 EMA against the trade
  - Square-off    : 15:20 IST
  - Entry window  : 9:30 – 14:45 IST | max 5 trades per day per symbol
"""

from collections import defaultdict
from datetime import date, datetime, time
from math import floor

import pandas as pd

from models.lucky13_models import (
    Lucky13SymbolMetrics,
    Lucky13TradeResult,
    Lucky13WeeklyResult,
)
from services.nifty_option_service import NiftyOptionService


_ENTRY_START        = time(9, 30)
_ENTRY_CUTOFF       = time(14, 45)
_SQUARE_OFF         = time(15, 20)
_MAX_TRADES_PER_DAY = 5


# ── Indicator computation ─────────────────────────────────────────────────────

def _compute_indicators(
    spot_candles: list[dict],
    ema_period: int = 13,
    vol_period: int = 20,
) -> pd.DataFrame:
    """
    Compute 13 EMA, volume SMA, VWAP (daily reset), and 5-min EMA from spot 1-min candles.
    Returns a DataFrame indexed by bar with all indicator columns.
    """
    df = pd.DataFrame(spot_candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 13 EMA on 1-min close
    df["ema13"] = df["close"].ewm(span=ema_period, adjust=False).mean()

    # Volume 20-period SMA
    df["vol_sma"] = df["volume"].rolling(vol_period, min_periods=1).mean()

    # VWAP — reset each calendar day
    df["typical"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"]  = df["typical"] * df["volume"]

    day_groups: list[pd.DataFrame] = []
    for _, day_df in df.groupby(df["datetime"].dt.date, sort=True):
        day_df = day_df.copy()
        cum_vol = day_df["volume"].cumsum()
        day_df["vwap"] = day_df["tp_vol"].cumsum() / cum_vol.replace(0, pd.NA)
        day_df["vwap"] = day_df["vwap"].ffill()
        day_groups.append(day_df)

    df = pd.concat(day_groups).sort_values("datetime").reset_index(drop=True)

    # 5-min EMA — resample 1-min to 5-min, compute EMA, forward-fill back to 1-min
    df_5m = (
        df.set_index("datetime")["close"]
        .resample("5min")
        .last()
        .dropna()
    )
    df_5m_ema = df_5m.ewm(span=ema_period, adjust=False).mean().rename("ema5m")

    df = df.set_index("datetime")
    df["ema5m"] = df_5m_ema
    df["ema5m"] = df["ema5m"].ffill()
    df = df.reset_index()

    return df


# ── Metrics helper ────────────────────────────────────────────────────────────

def _compute_metrics(symbol: str, trades: list[Lucky13TradeResult]) -> Lucky13SymbolMetrics:
    if not trades:
        return Lucky13SymbolMetrics(
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

    return Lucky13SymbolMetrics(
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


# ── Strategy class ────────────────────────────────────────────────────────────

class Lucky13EmaStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        ema_period: int = 13,
        vol_period: int = 20,
        vol_multiplier: float = 1.2,
        profit_pct: float = 1.5,
        stop_pct: float = 0.75,
        trailing_stop_pts: float = 0.0,   # 0 = disabled
        use_filters: bool = True,
        use_vwap_cross: bool = False,      # True: require actual VWAP cross; False: close vs VWAP
        start_date: str = "",
        end_date: str = "",
    ):
        self.nifty_service      = nifty_service
        self.capital            = capital
        self.ema_period         = ema_period
        self.vol_period         = vol_period
        self.vol_multiplier     = vol_multiplier
        self.profit_pct         = profit_pct
        self.stop_pct           = stop_pct
        self.trailing_stop_pts  = trailing_stop_pts
        self.use_filters        = use_filters
        self.use_vwap_cross     = use_vwap_cross
        self.start_date         = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date           = datetime.strptime(end_date,   "%d-%b-%Y").date()

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _run_symbol(
        self,
        spot_df: pd.DataFrame,
        option_candles: list[dict],
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[Lucky13TradeResult]:
        """
        Run Lucky13EMA on one option contract.

        spot_df       : indicators DataFrame from _compute_indicators (1-min spot)
        option_candles: raw option OHLC candles for the same period
        """
        if spot_df.empty or not option_candles:
            return []

        # Build option close price lookup: datetime → price
        opt_df = pd.DataFrame(option_candles)
        opt_df["datetime"] = pd.to_datetime(opt_df["datetime"])
        opt_df["close"] = opt_df["close"].astype(float)
        opt_price: dict[pd.Timestamp, float] = dict(
            zip(opt_df["datetime"], opt_df["close"])
        )

        symbol_label = f"NIFTY{strike}{option_type}"
        trades: list[Lucky13TradeResult] = []
        daily_trade_count: dict[date, int] = defaultdict(int)

        in_position      = False
        entry_opt_price  = 0.0
        peak_opt_price   = 0.0
        shares           = 0
        entry_row: pd.Series | None = None

        prev_close: float | None = None
        prev_ema13: float | None = None
        prev_vwap:  float | None = None

        for _, row in spot_df.iterrows():
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            today: date  = dt.date()

            spot_close = row["close"]
            ema13      = row["ema13"]
            vwap       = row["vwap"] if not pd.isna(row.get("vwap", float("nan"))) else None
            vol_sma    = row["vol_sma"]
            volume     = row["volume"]
            ema5m      = row["ema5m"] if not pd.isna(row.get("ema5m", float("nan"))) else None
            is_green   = spot_close > row["open"]
            is_red     = spot_close < row["open"]

            # Require a minimum warm-up before trading
            if pd.isna(ema13) or pd.isna(vol_sma):
                prev_close = spot_close
                prev_ema13 = ema13
                prev_vwap  = vwap
                continue

            # Fetch matching option price for this bar
            opt_close = opt_price.get(dt)

            # ── Exit logic ────────────────────────────────────────────────
            if in_position and entry_row is not None and opt_close is not None:
                if opt_close > peak_opt_price:
                    peak_opt_price = opt_close

                exit_reason = None

                if t >= _SQUARE_OFF:
                    exit_reason = "SQUARE_OFF"
                elif opt_close >= entry_opt_price * (1 + self.profit_pct / 100):
                    exit_reason = "PROFIT_TARGET"
                elif opt_close <= entry_opt_price * (1 - self.stop_pct / 100):
                    exit_reason = "STOP_LOSS"
                elif (
                    self.trailing_stop_pts > 0
                    and peak_opt_price - opt_close >= self.trailing_stop_pts
                ):
                    exit_reason = "TRAIL_STOP"
                elif prev_ema13 is not None:
                    # EMA reversal: spot crosses back against the trade
                    if option_type == "CE" and prev_close >= prev_ema13 and spot_close < ema13:
                        exit_reason = "EMA_REVERSAL"
                    elif option_type == "PE" and prev_close <= prev_ema13 and spot_close > ema13:
                        exit_reason = "EMA_REVERSAL"

                if exit_reason:
                    duration = int((dt - entry_row["datetime"]).total_seconds() / 60)
                    pnl      = round(shares * (opt_close - entry_opt_price), 2)

                    trades.append(Lucky13TradeResult(
                        symbol=symbol_label,
                        option_type=option_type,
                        strike=strike,
                        expiry_date=expiry_date,
                        entry_time=entry_row["datetime"],
                        exit_time=dt,
                        entry_price=entry_opt_price,
                        exit_price=opt_close,
                        shares=shares,
                        pnl=pnl,
                        exit_reason=exit_reason,
                        spot_close_at_entry=float(entry_row["close"]),
                        ema13_at_entry=float(entry_row["ema13"]),
                        vwap_at_entry=float(entry_row.get("vwap", 0.0) or 0.0),
                        volume_ratio_at_entry=float(entry_row["volume"] / entry_row["vol_sma"])
                            if entry_row["vol_sma"] > 0 else 0.0,
                        ema5m_at_entry=float(entry_row.get("ema5m", 0.0) or 0.0),
                        ema13_at_exit=float(ema13),
                        duration_minutes=duration,
                        volume_filter=entry_row.get("_vf", False),
                        vwap_filter=entry_row.get("_vwapf", False),
                        htf_ema_filter=entry_row.get("_htff", False),
                    ))

                    in_position    = False
                    entry_row      = None
                    entry_opt_price = 0.0
                    peak_opt_price  = 0.0

            # ── Entry logic ───────────────────────────────────────────────
            if (
                not in_position
                and prev_close is not None
                and prev_ema13 is not None
                and _ENTRY_START <= t <= _ENTRY_CUTOFF
                and daily_trade_count[today] < _MAX_TRADES_PER_DAY
                and opt_close is not None
                and opt_close > 0
            ):
                # Filter states
                vol_filter  = volume > vol_sma * self.vol_multiplier
                vwap_filter = True
                htf_filter  = True

                if vwap is not None:
                    prev_vwap_val = prev_vwap or vwap
                    if option_type == "CE":
                        if self.use_vwap_cross:
                            vwap_filter = (prev_close <= prev_vwap_val and spot_close > vwap)
                        else:
                            vwap_filter = spot_close > vwap
                    else:  # PE
                        if self.use_vwap_cross:
                            vwap_filter = (prev_close >= prev_vwap_val and spot_close < vwap)
                        else:
                            vwap_filter = spot_close < vwap

                if ema5m is not None:
                    htf_filter = (
                        spot_close > ema5m if option_type == "CE" else spot_close < ema5m
                    )

                filters_pass = (not self.use_filters) or (vol_filter and vwap_filter and htf_filter)

                # Entry signal: first colored candle crossing EMA
                signal = False
                if option_type == "CE":
                    # Green candle closes above EMA, previous close was below EMA
                    signal = is_green and spot_close > ema13 and prev_close < prev_ema13
                else:  # PE
                    # Red candle closes below EMA, previous close was above EMA
                    signal = is_red and spot_close < ema13 and prev_close > prev_ema13

                if signal and filters_pass:
                    entry_opt_price = opt_close
                    peak_opt_price  = opt_close
                    shares          = max(floor(self.capital / entry_opt_price), 1)
                    in_position     = True
                    # stash filter flags on a mutable copy of row for logging
                    row_copy = row.copy()
                    row_copy["_vf"]    = vol_filter
                    row_copy["_vwapf"] = vwap_filter
                    row_copy["_htff"]  = htf_filter
                    entry_row = row_copy
                    daily_trade_count[today] += 1

            prev_close = spot_close
            prev_ema13 = ema13
            prev_vwap  = vwap

        # Force-close any open position at end of data
        if in_position and entry_row is not None:
            last_opt = opt_df.iloc[-1]
            opt_close = float(last_opt["close"])
            dt        = last_opt["datetime"]
            duration  = int((dt - entry_row["datetime"]).total_seconds() / 60)
            pnl       = round(shares * (opt_close - entry_opt_price), 2)

            last_spot = spot_df.iloc[-1]
            trades.append(Lucky13TradeResult(
                symbol=symbol_label,
                option_type=option_type,
                strike=strike,
                expiry_date=expiry_date,
                entry_time=entry_row["datetime"],
                exit_time=dt,
                entry_price=entry_opt_price,
                exit_price=opt_close,
                shares=shares,
                pnl=pnl,
                exit_reason="SQUARE_OFF",
                spot_close_at_entry=float(entry_row["close"]),
                ema13_at_entry=float(entry_row["ema13"]),
                vwap_at_entry=float(entry_row.get("vwap", 0.0) or 0.0),
                volume_ratio_at_entry=float(entry_row["volume"] / entry_row["vol_sma"])
                    if entry_row["vol_sma"] > 0 else 0.0,
                ema5m_at_entry=float(entry_row.get("ema5m", 0.0) or 0.0),
                ema13_at_exit=float(last_spot["ema13"]),
                duration_minutes=duration,
                volume_filter=entry_row.get("_vf", False),
                vwap_filter=entry_row.get("_vwapf", False),
                htf_ema_filter=entry_row.get("_htff", False),
            ))

        return trades

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[Lucky13WeeklyResult]:
        expiry_results: list[Lucky13WeeklyResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)

        filters_label = (
            f"ON  (vol×{self.vol_multiplier}, VWAP={'cross' if self.use_vwap_cross else 'side'}, 5mEMA)"
            if self.use_filters else "OFF"
        )
        trail_label = f"{self.trailing_stop_pts} pts" if self.trailing_stop_pts > 0 else "OFF"

        print(f"\n{'='*75}")
        print(f"  NIFTY LUCKY13-EMA STRATEGY — WEEKLY EXPIRY BACKTEST")
        print(f"  Period     : {self.start_date}  →  {self.end_date}")
        print(f"  Capital    : ₹{self.capital:,.0f}  |  EMA: {self.ema_period}  |  Vol period: {self.vol_period}")
        print(f"  Profit     : {self.profit_pct}%  |  Stop: {self.stop_pct}%  |  Trail: {trail_label}")
        print(f"  Filters    : {filters_label}")
        print(f"  Expiries   : {len(wednesdays)}")
        print(f"{'='*75}\n")

        for expiry in wednesdays:
            monday           = NiftyOptionService.monday_of_week(expiry)
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

            # Fetch spot candles once per week, compute indicators
            try:
                spot_candles = self.nifty_service.get_nifty_spot_candles(
                    start=from_dt, end=to_dt, interval="1minute"
                )
                spot_df = _compute_indicators(spot_candles, self.ema_period, self.vol_period)
            except Exception as exc:
                print(f"  [{expiry}] Could not fetch spot candles: {exc}")
                continue

            week_result = Lucky13WeeklyResult(
                expiry_date=expiry,
                atm_strike=strike,
                nifty_open=nifty_open,
            )

            for opt_type in ("CE", "PE"):
                try:
                    option_candles = self.nifty_service.get_option_candles(
                        strike=strike,
                        expiry_date=expiry,
                        option_type=opt_type,
                        start=from_dt,
                        end=to_dt,
                        interval="1minute",
                    )
                    trades = self._run_symbol(spot_df, option_candles, opt_type, strike, expiry)

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
    def print_report(expiry_results: list[Lucky13WeeklyResult]) -> None:
        all_trades: list[Lucky13TradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[Lucky13TradeResult]] = defaultdict(list)
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
            sign      = "+" if total_pnl >= 0 else ""
            print(
                f"\n  Expiry {er.expiry_date}  |  ATM {er.atm_strike}"
                f"  |  Nifty open {er.nifty_open:.2f}"
                f"  |  Trades {len(er.all_trades)}"
                f"  |  PnL {sign}{total_pnl:.2f}"
            )
            print(f"  {sep}")

            header = (
                f"  {'Symbol':<22} {'Entry':>16} {'Exit':>16}"
                f" {'OptEntry':>9} {'OptExit':>8} {'Qty':>5}"
                f" {'PnL':>10} {'Reason':<14}"
                f" {'Spot@Entry':>11} {'EMA13':>7} {'VolRatio':>9}"
                f" {'VF':>3} {'VWAPF':>6} {'HTFF':>5}"
            )
            print(header)
            print(f"  {sep}")

            for t in sorted(er.all_trades, key=lambda x: x.entry_time):
                sign = "+" if t.pnl >= 0 else ""
                print(
                    f"  {t.symbol:<22}"
                    f" {t.entry_time.strftime('%d-%b %H:%M'):>16}"
                    f" {t.exit_time.strftime('%d-%b %H:%M'):>16}"
                    f" {t.entry_price:>9.2f}"
                    f" {t.exit_price:>8.2f}"
                    f" {t.shares:>5}"
                    f" {sign+f'{t.pnl:.2f}':>10}"
                    f" {t.exit_reason:<14}"
                    f" {t.spot_close_at_entry:>11.2f}"
                    f" {t.ema13_at_entry:>7.2f}"
                    f" {t.volume_ratio_at_entry:>9.2f}"
                    f" {'Y' if t.volume_filter else 'N':>3}"
                    f" {'Y' if t.vwap_filter else 'N':>6}"
                    f" {'Y' if t.htf_ema_filter else 'N':>5}"
                )

        # Per-symbol metrics
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
            avg_s = f"{'+' if m.avg_pnl   >= 0 else ''}{m.avg_pnl:.2f}"
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
