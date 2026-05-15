"""
HTF Candle Direction Strategy for Nifty weekly options.

Core idea:
  Determine directional bias from a Higher Timeframe (HTF) candle:
    Bullish bias  (HTF close > HTF open) → buy CE
    Bearish bias  (HTF close < HTF open) → buy PE

Filters (optional):
  EMA filter    — CE only above EMA, PE only below EMA
  Volume filter — trade only when volume > avg_volume * multiplier

Execution rules:
  - One trade per day (first valid signal only)
  - Entry window: 9:30 – 14:45 IST
  - Square-off at 15:20 IST
"""

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from math import floor

import pandas as pd

from models.htf_models import HTFTradeResult, SymbolMetrics, WeeklyExpiryResult
from services.nifty_option_service import NiftyOptionService


_ENTRY_START  = time(9, 30)
_ENTRY_CUTOFF = time(14, 45)
_SQUARE_OFF   = time(15, 20)


def _compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _prepare_spot_df(candles: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _htf_bias_map(htf_candles: list[dict]) -> dict[date, tuple[str, float, float]]:
    """
    Build a map: trading_date → (bias, htf_open, htf_close).
    The HTF candle for day D is used as the bias for that same day
    (uses the most recent completed HTF candle before the session opens).
    For daily HTF, the previous day's candle is used.
    """
    df = _prepare_spot_df(htf_candles)
    bias_map: dict[date, tuple[str, float, float]] = {}
    for _, row in df.iterrows():
        o, c = row["open"], row["close"]
        if pd.isna(o) or pd.isna(c):
            continue
        bias = "BULLISH" if c > o else "BEARISH"
        candle_date = row["datetime"].date()
        bias_map[candle_date] = (bias, float(o), float(c))
    return bias_map


def _compute_metrics(symbol: str, trades: list[HTFTradeResult]) -> SymbolMetrics:
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


class HTFCandleStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        htf_interval: str = "1day",
        lf_interval: str = "5minute",
        ema_period: int = 0,             # 0 = EMA filter disabled
        use_volume_filter: bool = False,
        volume_avg_period: int = 20,
        volume_multiplier: float = 1.5,
        trade_direction: str = "BOTH",   # "BOTH" | "CE_ONLY" | "PE_ONLY"
        start_date: str = "",
        end_date: str = "",
    ):
        self.nifty_service       = nifty_service
        self.capital             = capital
        self.htf_interval        = htf_interval
        self.lf_interval         = lf_interval
        self.ema_period          = ema_period
        self.use_volume_filter   = use_volume_filter
        self.volume_avg_period   = volume_avg_period
        self.volume_multiplier   = volume_multiplier
        self.trade_direction     = trade_direction
        self.start_date          = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date            = datetime.strptime(end_date,   "%d-%b-%Y").date()

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _run_symbol(
        self,
        option_candles: list[dict],
        spot_lf_df: pd.DataFrame,
        bias_map: dict[date, tuple[str, float, float]],
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[HTFTradeResult]:
        """
        Run the HTF direction strategy on a single option contract.
        spot_lf_df supplies EMA and volume signals on the lower timeframe.
        """
        if not option_candles:
            return []

        opt_df = _prepare_spot_df(option_candles)
        if opt_df.empty:
            return []

        # Build EMA on LF spot closes if filter is enabled
        if self.ema_period > 0 and not spot_lf_df.empty:
            spot_lf_df = spot_lf_df.copy()
            spot_lf_df["ema"] = _compute_ema(spot_lf_df["close"], self.ema_period)
        else:
            spot_lf_df = spot_lf_df.copy()
            spot_lf_df["ema"] = float("nan")

        # Build volume average on LF spot if filter is enabled
        if self.use_volume_filter and "volume" in spot_lf_df.columns:
            spot_lf_df["vol_avg"] = (
                spot_lf_df["volume"]
                .rolling(self.volume_avg_period, min_periods=1)
                .mean()
            )
        else:
            spot_lf_df["vol_avg"] = float("nan")

        # Index LF spot by datetime for fast lookup
        spot_lf_df = spot_lf_df.set_index("datetime")

        symbol_label = f"NIFTY{strike}{option_type}"
        trades: list[HTFTradeResult] = []
        traded_today: set[date] = set()

        in_position   = False
        entry_row     = None
        entry_price   = 0.0
        shares        = 0
        entry_htf     = ("", 0.0, 0.0)
        entry_ema     = float("nan")
        entry_vol     = float("nan")

        for _, row in opt_df.iterrows():
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            today: date  = dt.date()
            price: float = float(row["close"])

            if pd.isna(price) or price <= 0:
                continue

            # ── Exit ─────────────────────────────────────────────────────────
            if in_position and t >= _SQUARE_OFF:
                duration = int((dt - entry_row["datetime"]).total_seconds() / 60)
                pnl      = round(shares * (price - entry_price), 2)
                trades.append(HTFTradeResult(
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
                    htf_bias=entry_htf[0],
                    htf_open=entry_htf[1],
                    htf_close=entry_htf[2],
                    ema_at_entry=entry_ema,
                    volume_at_entry=entry_vol,
                    duration_minutes=duration,
                ))
                in_position = False
                entry_row   = None

            # ── Entry ─────────────────────────────────────────────────────────
            if (
                not in_position
                and today not in traded_today
                and _ENTRY_START <= t <= _ENTRY_CUTOFF
            ):
                # HTF bias check — use previous day's bias for daily HTF,
                # or current day for intraday HTF
                if self.htf_interval == "1day":
                    bias_date = today - timedelta(days=1)
                    # walk back past weekends
                    while bias_date not in bias_map and bias_date >= self.start_date:
                        bias_date -= timedelta(days=1)
                else:
                    bias_date = today

                if bias_date not in bias_map:
                    continue

                bias, htf_o, htf_c = bias_map[bias_date]

                # Direction filter
                if option_type == "CE" and bias != "BULLISH":
                    continue
                if option_type == "PE" and bias != "BEARISH":
                    continue
                if option_type == "CE" and self.trade_direction == "PE_ONLY":
                    continue
                if option_type == "PE" and self.trade_direction == "CE_ONLY":
                    continue

                # Get LF spot row for EMA/volume lookup (nearest timestamp)
                spot_row = spot_lf_df.asof(dt) if not spot_lf_df.empty else None

                ema_val = float(spot_row["ema"]) if spot_row is not None and not pd.isna(spot_row["ema"]) else float("nan")
                vol_val = float(spot_row["volume"]) if spot_row is not None and "volume" in spot_row.index and not pd.isna(spot_row["volume"]) else float("nan")
                vol_avg = float(spot_row["vol_avg"]) if spot_row is not None and "vol_avg" in spot_row.index and not pd.isna(spot_row["vol_avg"]) else float("nan")
                spot_close = float(spot_row["close"]) if spot_row is not None and not pd.isna(spot_row["close"]) else float("nan")

                # EMA filter
                if self.ema_period > 0 and not pd.isna(ema_val) and not pd.isna(spot_close):
                    if option_type == "CE" and spot_close <= ema_val:
                        continue
                    if option_type == "PE" and spot_close >= ema_val:
                        continue

                # Volume filter
                if self.use_volume_filter and not pd.isna(vol_val) and not pd.isna(vol_avg):
                    if vol_val < vol_avg * self.volume_multiplier:
                        continue

                entry_price = price
                shares      = max(floor(self.capital / entry_price), 1)
                in_position = True
                entry_row   = row
                entry_htf   = (bias, htf_o, htf_c)
                entry_ema   = ema_val
                entry_vol   = vol_val
                traded_today.add(today)

        # Force-close any open position at end of data
        if in_position and entry_row is not None:
            last  = opt_df.iloc[-1]
            price = float(last["close"])
            dt    = last["datetime"]
            duration = int((dt - entry_row["datetime"]).total_seconds() / 60)
            pnl      = round(shares * (price - entry_price), 2)
            trades.append(HTFTradeResult(
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
                htf_bias=entry_htf[0],
                htf_open=entry_htf[1],
                htf_close=entry_htf[2],
                ema_at_entry=entry_ema,
                volume_at_entry=entry_vol,
                duration_minutes=duration,
            ))

        return trades

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[WeeklyExpiryResult]:
        expiry_results: list[WeeklyExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        print(f"\n{'='*70}")
        print(f"  NIFTY HTF CANDLE DIRECTION STRATEGY — WEEKLY EXPIRY BACKTEST")
        print(f"  Period    : {self.start_date}  →  {self.end_date}")
        print(f"  Capital   : ₹{self.capital:,.0f}")
        print(f"  HTF       : {self.htf_interval}  |  LF: {self.lf_interval}")
        print(f"  EMA period: {self.ema_period if self.ema_period > 0 else 'disabled'}")
        print(f"  Volume    : {'enabled (x' + str(self.volume_multiplier) + ')' if self.use_volume_filter else 'disabled'}")
        print(f"  Direction : {self.trade_direction}")
        print(f"  Expiries  : {len(wednesdays)}")
        print(f"{'='*70}\n")

        for expiry in wednesdays:
            monday = NiftyOptionService.monday_of_week(expiry)
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

            # Fetch HTF spot candles to build bias map
            # For daily HTF we look back one extra day to cover Monday bias
            htf_from = datetime(win_start.year, win_start.month, win_start.day, 0, 0, 0) - timedelta(days=2)
            try:
                htf_candles = self.nifty_service.get_nifty_spot_candles(
                    start=htf_from, end=to_dt, interval=self.htf_interval
                )
                bias_map = _htf_bias_map(htf_candles)
            except Exception as exc:
                print(f"  [{expiry}] Could not fetch HTF candles: {exc}")
                continue

            # Fetch LF spot candles for EMA / volume filters
            try:
                spot_lf_candles = self.nifty_service.get_nifty_spot_candles(
                    start=from_dt, end=to_dt, interval=self.lf_interval
                )
                spot_lf_df = _prepare_spot_df(spot_lf_candles) if spot_lf_candles else pd.DataFrame()
            except Exception as exc:
                print(f"  [{expiry}] Could not fetch LF spot candles: {exc}")
                spot_lf_df = pd.DataFrame()

            week_result = WeeklyExpiryResult(
                expiry_date=expiry,
                atm_strike=strike,
                nifty_open=nifty_open,
            )

            opt_types = []
            if self.trade_direction in ("BOTH", "CE_ONLY"):
                opt_types.append("CE")
            if self.trade_direction in ("BOTH", "PE_ONLY"):
                opt_types.append("PE")

            for opt_type in opt_types:
                try:
                    candles = self.nifty_service.get_option_candles(
                        strike=strike,
                        expiry_date=expiry,
                        option_type=opt_type,
                        start=from_dt,
                        end=to_dt,
                        interval=self.lf_interval,
                    )
                    trades = self._run_symbol(
                        candles, spot_lf_df, bias_map, opt_type, strike, expiry
                    )

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
    def print_report(expiry_results: list[WeeklyExpiryResult]) -> None:
        all_trades: list[HTFTradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[HTFTradeResult]] = defaultdict(list)
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
            print(
                f"\n  Expiry {er.expiry_date}  |  ATM {er.atm_strike}"
                f"  |  Nifty open {er.nifty_open:.2f}"
                f"  |  Trades {len(er.all_trades)}"
                f"  |  PnL {pnl_sign}{total_pnl:.2f}"
            )
            print(f"  {sep}")

            header = (
                f"  {'Symbol':<22} {'Entry':>19} {'Exit':>19}"
                f" {'Entry₹':>8} {'Exit₹':>8} {'Qty':>5}"
                f" {'PnL':>10} {'Reason':<12}"
                f" {'Bias':<9} {'HTF O':>8} {'HTF C':>8}"
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
                    f" {t.htf_bias:<9}"
                    f" {t.htf_open:>8.2f}"
                    f" {t.htf_close:>8.2f}"
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
