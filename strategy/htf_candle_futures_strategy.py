"""
HTF Candle Direction Strategy for MCX commodity monthly futures.

Core idea:
  Determine directional bias from a Higher Timeframe (HTF) candle:
    Bullish bias  (HTF close > HTF open) → go LONG futures
    Bearish bias  (HTF close < HTF open) → go SHORT futures

Filters (optional):
  EMA filter    — LONG only above EMA, SHORT only below EMA
  Volume filter — trade only when volume > avg_volume * multiplier

Execution rules:
  - One trade per day (first valid signal only)
  - Entry window: 09:00 – 22:45 IST (MCX session)
  - Square-off at 23:25 IST
"""

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from math import floor

import pandas as pd

from models.htf_models import (
    HTFFuturesTradeResult,
    MonthlyFuturesExpiryResult,
    SymbolMetrics,
)
from services.commodity_option_service import (
    COMMODITY_CONFIG,
    GOLD_FUTURES_CONTRACT_MONTHS,
    CommodityOptionService,
)


_ENTRY_START  = time(9, 0)
_ENTRY_CUTOFF = time(22, 45)
_SQUARE_OFF   = time(23, 25)


def _compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _prepare_df(candles: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _htf_bias_map(htf_candles: list[dict]) -> dict[date, tuple[str, float, float]]:
    """Build a map: candle_date → (bias, htf_open, htf_close)."""
    df = _prepare_df(htf_candles)
    bias_map: dict[date, tuple[str, float, float]] = {}
    for _, row in df.iterrows():
        o, c = row["open"], row["close"]
        if pd.isna(o) or pd.isna(c):
            continue
        bias = "BULLISH" if c > o else "BEARISH"
        bias_map[row["datetime"].date()] = (bias, float(o), float(c))
    return bias_map


def _compute_metrics(symbol: str, trades: list[HTFFuturesTradeResult]) -> SymbolMetrics:
    if not trades:
        return SymbolMetrics(
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


class HTFCandleFuturesStrategy:
    def __init__(
        self,
        commodity_service: CommodityOptionService,
        commodity: str = "GOLD",
        lot_size: int = 1,
        capital: float = 100_000.0,
        htf_interval: str = "1day",
        lf_interval: str = "5minute",
        ema_period: int = 0,
        use_volume_filter: bool = False,
        volume_avg_period: int = 20,
        volume_multiplier: float = 1.5,
        trade_direction: str = "BOTH",   # "BOTH" | "LONG_ONLY" | "SHORT_ONLY"
        start_date: str = "",
        end_date: str = "",
    ):
        if commodity not in COMMODITY_CONFIG:
            raise ValueError(f"Unknown commodity '{commodity}'. Choose from: {list(COMMODITY_CONFIG)}")

        self.commodity_service = commodity_service
        self.commodity         = commodity
        self.lot_size          = lot_size
        self.capital           = capital
        self.htf_interval      = htf_interval
        self.lf_interval       = lf_interval
        self.ema_period        = ema_period
        self.use_volume_filter = use_volume_filter
        self.volume_avg_period = volume_avg_period
        self.volume_multiplier = volume_multiplier
        self.trade_direction   = trade_direction
        self.start_date        = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date          = datetime.strptime(end_date,   "%d-%b-%Y").date()

    # ── Core per-day backtest ─────────────────────────────────────────────────

    def _run_day(
        self,
        futures_candles: list[dict],
        lf_df: pd.DataFrame,
        bias_map: dict[date, tuple[str, float, float]],
        expiry_date: date,
    ) -> list[HTFFuturesTradeResult]:
        """Apply HTF direction strategy on a single day's futures candles."""
        if not futures_candles:
            return []

        fut_df = _prepare_df(futures_candles)
        if fut_df.empty:
            return []

        # EMA on LF candles if enabled
        if self.ema_period > 0 and not lf_df.empty:
            lf_df = lf_df.copy()
            lf_df["ema"] = _compute_ema(lf_df["close"], self.ema_period)
        else:
            lf_df = lf_df.copy()
            lf_df["ema"] = float("nan")

        # Volume average on LF candles if enabled
        if self.use_volume_filter and "volume" in lf_df.columns:
            lf_df["vol_avg"] = (
                lf_df["volume"].rolling(self.volume_avg_period, min_periods=1).mean()
            )
        else:
            lf_df["vol_avg"] = float("nan")

        lf_df = lf_df.set_index("datetime") if not lf_df.empty else lf_df

        symbol_label = f"{self.commodity}-FUT-{expiry_date}"
        trades: list[HTFFuturesTradeResult] = []
        traded_today: set[date] = set()

        in_position  = False
        entry_row    = None
        entry_price  = 0.0
        lots         = 0
        direction    = ""
        entry_htf    = ("", 0.0, 0.0)
        entry_ema    = float("nan")
        entry_vol    = float("nan")

        for _, row in fut_df.iterrows():
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            today: date  = dt.date()
            price: float = float(row["close"])

            if pd.isna(price) or price <= 0:
                continue

            # ── Exit ─────────────────────────────────────────────────────────
            if in_position and t >= _SQUARE_OFF:
                duration   = int((dt - entry_row["datetime"]).total_seconds() / 60)
                multiplier = 1 if direction == "LONG" else -1
                pnl        = round(lots * self.lot_size * multiplier * (price - entry_price), 2)
                trades.append(HTFFuturesTradeResult(
                    symbol=symbol_label,
                    commodity=self.commodity,
                    direction=direction,
                    expiry_date=expiry_date,
                    entry_time=entry_row["datetime"],
                    exit_time=dt,
                    entry_price=entry_price,
                    exit_price=price,
                    lots=lots,
                    lot_size=self.lot_size,
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
                # For daily HTF use previous day's bias
                if self.htf_interval == "1day":
                    bias_date = today - timedelta(days=1)
                    while bias_date not in bias_map and bias_date >= self.start_date:
                        bias_date -= timedelta(days=1)
                else:
                    bias_date = today

                if bias_date not in bias_map:
                    continue

                bias, htf_o, htf_c = bias_map[bias_date]

                # Direction filter
                if bias == "BULLISH" and self.trade_direction == "SHORT_ONLY":
                    continue
                if bias == "BEARISH" and self.trade_direction == "LONG_ONLY":
                    continue

                desired_dir = "LONG" if bias == "BULLISH" else "SHORT"

                # EMA / volume lookup from LF candles
                spot_row = lf_df.asof(dt) if not lf_df.empty and isinstance(lf_df.index, pd.DatetimeIndex) else None

                ema_val    = float(spot_row["ema"])     if spot_row is not None and not pd.isna(spot_row.get("ema", float("nan")))     else float("nan")
                vol_val    = float(spot_row["volume"])  if spot_row is not None and "volume" in spot_row.index and not pd.isna(spot_row["volume"]) else float("nan")
                vol_avg    = float(spot_row["vol_avg"]) if spot_row is not None and "vol_avg" in spot_row.index and not pd.isna(spot_row["vol_avg"]) else float("nan")
                lf_close   = float(spot_row["close"])   if spot_row is not None and not pd.isna(spot_row.get("close", float("nan")))   else float("nan")

                # EMA filter
                if self.ema_period > 0 and not pd.isna(ema_val) and not pd.isna(lf_close):
                    if desired_dir == "LONG"  and lf_close <= ema_val:
                        continue
                    if desired_dir == "SHORT" and lf_close >= ema_val:
                        continue

                # Volume filter
                if self.use_volume_filter and not pd.isna(vol_val) and not pd.isna(vol_avg):
                    if vol_val < vol_avg * self.volume_multiplier:
                        continue

                entry_price = price
                lots        = max(floor(self.capital / (entry_price * self.lot_size)), 1)
                direction   = desired_dir
                in_position = True
                entry_row   = row
                entry_htf   = (bias, htf_o, htf_c)
                entry_ema   = ema_val
                entry_vol   = vol_val
                traded_today.add(today)

        # Force-close any open position at end of data
        if in_position and entry_row is not None:
            last       = fut_df.iloc[-1]
            exit_price = float(last["close"])
            dt         = last["datetime"]
            duration   = int((dt - entry_row["datetime"]).total_seconds() / 60)
            multiplier = 1 if direction == "LONG" else -1
            pnl        = round(lots * self.lot_size * multiplier * (exit_price - entry_price), 2)
            trades.append(HTFFuturesTradeResult(
                symbol=symbol_label,
                commodity=self.commodity,
                direction=direction,
                expiry_date=expiry_date,
                entry_time=entry_row["datetime"],
                exit_time=dt,
                entry_price=entry_price,
                exit_price=exit_price,
                lots=lots,
                lot_size=self.lot_size,
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

    # ── Monthly contract orchestration ────────────────────────────────────────

    def run_monthly_backtest(self) -> list[MonthlyFuturesExpiryResult]:
        expiry_results: list[MonthlyFuturesExpiryResult] = []

        print(f"\n{'='*75}")
        print(f"  {self.commodity} FUTURES — HTF CANDLE DIRECTION STRATEGY — MCX")
        print(f"  Period    : {self.start_date}  →  {self.end_date}")
        print(f"  Capital   : ₹{self.capital:,.0f}  |  Lot size: {self.lot_size}")
        print(f"  HTF       : {self.htf_interval}  |  LF: {self.lf_interval}")
        print(f"  EMA period: {self.ema_period if self.ema_period > 0 else 'disabled'}")
        print(f"  Volume    : {'enabled (x' + str(self.volume_multiplier) + ')' if self.use_volume_filter else 'disabled'}")
        print(f"  Direction : {self.trade_direction}")
        print(f"{'='*75}\n")

        contracts = self._monthly_contracts()

        for futures_expiry, win_start, win_end in contracts:
            win_start = max(win_start, self.start_date)
            win_end   = min(win_end,   self.end_date)
            if win_start > win_end:
                continue

            print(f"  Contract {futures_expiry}  |  Window {win_start} → {win_end}")

            # Fetch HTF candles for the whole window (extra lookback for bias on day 1)
            htf_from = datetime(win_start.year, win_start.month, win_start.day, 0, 0, 0) - timedelta(days=3)
            htf_to   = datetime(win_end.year,   win_end.month,   win_end.day,   23, 59, 0)
            try:
                htf_candles = self.commodity_service.get_futures_candles(
                    stock_code=self.commodity,
                    expiry_date=futures_expiry,
                    start=htf_from,
                    end=htf_to,
                    interval=self.htf_interval,
                )
                bias_map = _htf_bias_map(htf_candles)
            except Exception as exc:
                print(f"    Could not fetch HTF candles: {exc}")
                continue

            # Opening price for reference (first trading day)
            try:
                first_open = self.commodity_service.get_commodity_open(
                    stock_code=self.commodity,
                    trade_date=win_start,
                    expiry_date=futures_expiry,
                )
            except Exception:
                first_open = 0.0

            result = MonthlyFuturesExpiryResult(
                expiry_date=futures_expiry,
                commodity=self.commodity,
                futures_open=first_open,
            )

            trade_date = win_start
            while trade_date <= win_end:
                if not self.commodity_service.is_mcx_trading_day(trade_date):
                    trade_date += timedelta(days=1)
                    continue

                day_start = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 0, 0)
                day_end   = datetime(trade_date.year, trade_date.month, trade_date.day, 23, 30, 0)

                # Fetch LF candles for EMA / volume filters
                try:
                    lf_candles = self.commodity_service.get_futures_candles(
                        stock_code=self.commodity,
                        expiry_date=futures_expiry,
                        start=day_start,
                        end=day_end,
                        interval=self.lf_interval,
                    )
                    lf_df = _prepare_df(lf_candles) if lf_candles else pd.DataFrame()
                except Exception:
                    lf_df = pd.DataFrame()

                try:
                    fut_candles = self.commodity_service.get_futures_candles(
                        stock_code=self.commodity,
                        expiry_date=futures_expiry,
                        start=day_start,
                        end=day_end,
                        interval=self.lf_interval,
                    )
                    day_trades = self._run_day(fut_candles, lf_df, bias_map, futures_expiry)
                    result.trades.extend(day_trades)

                    if day_trades:
                        day_pnl = sum(t.pnl for t in day_trades)
                        print(f"    {trade_date}: {len(day_trades)} trade(s)  PnL ₹{day_pnl:+.2f}")
                except Exception as exc:
                    print(f"    [{trade_date}] error: {exc}")

                trade_date += timedelta(days=1)

            expiry_results.append(result)
            total_pnl = sum(t.pnl for t in result.trades)
            print(f"  → Contract {futures_expiry}: {len(result.trades)} trades  PnL ₹{total_pnl:+.2f}\n")

        return expiry_results

    def _monthly_contracts(self) -> list[tuple[date, date, date]]:
        """
        Return (futures_expiry, window_start, window_end) for each monthly
        futures contract overlapping the backtest period.

        For GOLD: even months only (Feb/Apr/Jun/Aug/Oct/Dec).
        For other commodities: every calendar month.
        """
        contracts: list[tuple[date, date, date]] = []
        year  = self.start_date.year
        month = self.start_date.month

        # Step back one period to catch contracts started before start_date
        if month == 1:
            year -= 1
            month = 12
        else:
            month -= 1

        seen: set[date] = set()
        for _ in range(36):
            include = True
            if self.commodity == "GOLD" and month not in GOLD_FUTURES_CONTRACT_MONTHS:
                include = False

            if include:
                expiry = CommodityOptionService._nominal_futures_expiry(year, month) \
                    if self.commodity == "GOLD" \
                    else CommodityOptionService._nominal_silver_futures_expiry(year, month)
                if expiry not in seen:
                    seen.add(expiry)
                    win_start = date(year, month, 1)
                    win_end   = expiry
                    if win_end >= self.start_date:
                        contracts.append((expiry, win_start, win_end))
                    if win_start > self.end_date:
                        break

            month += 1
            if month > 12:
                month = 1
                year += 1

        return contracts

    # ── Reporting ─────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(expiry_results: list[MonthlyFuturesExpiryResult]) -> None:
        all_trades: list[HTFFuturesTradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.trades)

        sep = "─" * 108

        print(f"\n{'='*108}")
        print("  RESULTS BY FUTURES CONTRACT")
        print(f"{'='*108}")

        for er in expiry_results:
            if not er.trades:
                continue
            total_pnl = sum(t.pnl for t in er.trades)
            longs  = [t for t in er.trades if t.direction == "LONG"]
            shorts = [t for t in er.trades if t.direction == "SHORT"]
            print(
                f"\n  Contract {er.expiry_date}  |  {er.commodity}"
                f"  |  First open ₹{er.futures_open:.2f}"
                f"  |  Trades {len(er.trades)} (L:{len(longs)} S:{len(shorts)})"
                f"  |  PnL ₹{total_pnl:+.2f}"
            )
            print(f"  {sep}")
            print(
                f"  {'Dir':<6} {'Entry':>19} {'Exit':>19}"
                f" {'EntryPx':>9} {'ExitPx':>9} {'Lots':>4} {'LotSz':>5}"
                f" {'PnL':>11} {'Reason':<12}"
                f" {'Bias':<9} {'HTF O':>8} {'HTF C':>8}"
            )
            print(f"  {sep}")

            for t in sorted(er.trades, key=lambda x: x.entry_time):
                pnl_sign = "+" if t.pnl >= 0 else ""
                print(
                    f"  {t.direction:<6}"
                    f" {t.entry_time.strftime('%d-%b %H:%M'):>19}"
                    f" {t.exit_time.strftime('%d-%b %H:%M'):>19}"
                    f" {t.entry_price:>9.2f}"
                    f" {t.exit_price:>9.2f}"
                    f" {t.lots:>4}"
                    f" {t.lot_size:>5}"
                    f" {pnl_sign+f'{t.pnl:.2f}':>11}"
                    f" {t.exit_reason:<12}"
                    f" {t.htf_bias:<9}"
                    f" {t.htf_open:>8.2f}"
                    f" {t.htf_close:>8.2f}"
                )

        # ── Overall metrics ───────────────────────────────────────────────
        commodity = expiry_results[0].commodity if expiry_results else "FUTURES"
        m = _compute_metrics(f"{commodity}-FUT", all_trades)

        print(f"\n{'='*108}")
        print(f"  OVERALL METRICS — {commodity} FUTURES HTF CANDLE STRATEGY")
        print(f"{'='*108}")
        print(f"  Total trades       : {m.total_trades}")
        print(f"  Wins / Losses      : {m.wins} / {m.losses}")
        print(f"  Win rate           : {m.win_rate:.1f}%")
        print(f"  Total PnL          : ₹{m.total_pnl:+.2f}")
        print(f"  Avg PnL per trade  : ₹{m.avg_pnl:+.2f}")
        pf = m.profit_factor
        print(f"  Profit factor      : {pf if pf != float('inf') else '∞'}")
        print(f"  Best trade         : ₹{m.best_trade:+.2f}")
        print(f"  Worst trade        : ₹{m.worst_trade:+.2f}")
        print(f"  Avg duration (min) : {m.avg_duration_minutes:.1f}")
        print(f"  Max consec. losses : {m.max_consecutive_losses}")

        longs      = [t for t in all_trades if t.direction == "LONG"]
        shorts     = [t for t in all_trades if t.direction == "SHORT"]
        long_pnl   = sum(t.pnl for t in longs)
        short_pnl  = sum(t.pnl for t in shorts)
        print(f"\n  LONG  trades: {len(longs):>4}  |  PnL ₹{long_pnl:+.2f}")
        print(f"  SHORT trades: {len(shorts):>4}  |  PnL ₹{short_pnl:+.2f}")
        print(f"{'='*108}\n")
