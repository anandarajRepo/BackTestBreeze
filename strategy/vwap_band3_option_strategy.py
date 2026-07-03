"""
VWAP Band 3 mean-reversion strategy for Nifty weekly options (seconds data).

Port of the TradingView Pine v6 "VWAP Band 3 Mean-Reversion Strategy" to the
option premium series:

  VWAP + bands:
    Session-anchored (per trading day) VWAP on the typical price hlc3,
    volume-weighted. Band 3 sits `band_mult` volume-weighted standard
    deviations away from the VWAP ("Standard Deviation" mode), or
    `band_mult` percent away ("Percentage" mode).

  Entry (resting limit orders at the bands, filled intrabar on a touch):
    LONG  — price touches the LOWER band 3  → filled at the band price
    SHORT — price touches the UPPER band 3  → filled at the band price
    Only one position open at a time; no new entries while a trade is active.

  Exit:
    TP at the VWAP value captured at entry (fixed for the trade's life);
    SL the same distance on the other side of the entry (1:1 R:R).
    If a bar spans both levels the stop-loss is assumed to fill first
    (conservative). Square-off at 15:20 IST closes anything still open.

  Position sizing:
    qty = equity × risk% ÷ (distance from band to VWAP), so a stop-loss hit
    loses exactly `risk_percent` of current equity. Equity starts at
    `capital` and compounds with realized PnL over the backtest.

  Timing:
    No new entries until `trading_delay_hours` have passed since the session
    start (first bar of the day), giving the bands time to develop.
    No new entries after 14:45; max 5 trades per day per symbol.
"""

from collections import defaultdict
from datetime import date, datetime, time
from math import floor

import pandas as pd

from models.vwap_band3_models import (
    VWAPBand3SymbolMetrics,
    VWAPBand3TradeResult,
    VWAPBand3WeeklyExpiryResult,
)
from services.nifty_option_service import NiftyOptionService


def resample_candles(candles: list[dict], seconds: int) -> list[dict]:
    """
    Resample a list of 1-second OHLC dicts into N-second candles.

    Each input dict must have keys: datetime, open, high, low, close, volume
    (optional). seconds=1 returns the original data unchanged.
    """
    if seconds <= 1 or not candles:
        return candles

    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    df["_bucket"] = df["datetime"].apply(lambda ts: ts.floor(f"{seconds}s"))

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
    resampled["datetime"] = pd.to_datetime(resampled["datetime"])
    return resampled.to_dict("records")


_ENTRY_CUTOFF = time(14, 45)
_SQUARE_OFF   = time(15, 20)
_MAX_TRADES_PER_DAY = 5


def compute_vwap_bands(
    candles: list[dict],
    band_mult: float,
    calc_mode: str = "Standard Deviation",
) -> pd.DataFrame:
    """
    Compute the session (per-day) VWAP and the Band 3 envelope from a list of
    OHLC dicts, mirroring Pine's ta.vwap(src, isNewPeriod, 1) with a Session
    anchor and src = hlc3.

    The standard deviation is the volume-weighted stdev of the typical price
    around the VWAP:  var = Σ(v·src²)/Σv − vwap².

    Returns a DataFrame with columns: datetime, open, high, low, close,
    volume, vwap, upper_band, lower_band, session_start.
    """
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    day = df["datetime"].dt.date
    vol = df["volume"].where(df["volume"] > 0, 0.0)

    cum_vol  = vol.groupby(day).cumsum()
    cum_pv   = (typical * vol).groupby(day).cumsum()
    cum_pv2  = (typical * typical * vol).groupby(day).cumsum()

    vwap = (cum_pv / cum_vol).where(cum_vol > 0, typical)
    variance = (cum_pv2 / cum_vol - vwap * vwap).where(cum_vol > 0, 0.0)
    stdev = variance.clip(lower=0.0) ** 0.5

    if calc_mode == "Percentage":
        band_basis = vwap * 0.01
    else:
        band_basis = stdev

    result = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    result["vwap"]       = vwap
    result["upper_band"] = vwap + band_basis * band_mult
    result["lower_band"] = vwap - band_basis * band_mult
    result["session_start"] = df.groupby(day)["datetime"].transform("first")
    return result


def _compute_metrics(
    symbol: str, trades: list[VWAPBand3TradeResult]
) -> VWAPBand3SymbolMetrics:
    if not trades:
        return VWAPBand3SymbolMetrics(
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

    return VWAPBand3SymbolMetrics(
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


class VWAPBand3OptionStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        band_mult: float = 3.0,
        calc_mode: str = "Standard Deviation",
        risk_percent: float = 5.0,
        trading_delay_hours: float = 2.0,
        allow_short: bool = True,
        start_date: str = "",
        end_date: str = "",
        interval: str = "1second",
        resample_seconds: int = 1,
        print_resampled: bool = False,
        cache_only: bool = False,
        market_holidays: set[date] | None = None,
        per_day_atm: bool = False,
    ):
        self.nifty_service = nifty_service
        # Starting equity. Position size is risk-based: a stop-loss hit loses
        # exactly `risk_percent` of current equity, which compounds with
        # realized PnL over the backtest.
        self.capital       = capital
        self.band_mult     = band_mult
        # "Standard Deviation" or "Percentage" (a multiplier of 1 = 1% of VWAP)
        self.calc_mode     = calc_mode
        self.risk_percent  = risk_percent
        # No new entries until this many hours have passed since the session
        # start, giving the bands time to develop (Pine's timing gate).
        self.trading_delay_hours = trading_delay_hours
        # When False, upper-band touches (short the premium) are ignored and
        # only lower-band LONG entries are taken.
        self.allow_short   = allow_short
        self.start_date    = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date      = datetime.strptime(end_date,   "%d-%b-%Y").date()
        self.interval      = interval
        self.resample_seconds = resample_seconds
        self.print_resampled  = print_resampled
        self.cache_only       = cache_only
        self.market_holidays  = set(market_holidays) if market_holidays else set()
        self.per_day_atm      = per_day_atm

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _run_symbol(
        self,
        candles: list[dict],
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[VWAPBand3TradeResult]:
        """
        Run the VWAP Band 3 mean-reversion strategy on a single option
        contract's candle data. Returns a list of completed trades.
        """
        if not candles:
            return []

        bands = compute_vwap_bands(candles, self.band_mult, self.calc_mode)
        symbol_label = f"NIFTY{strike}{option_type}"
        trades: list[VWAPBand3TradeResult] = []

        daily_trade_count: dict[date, int] = defaultdict(int)

        equity = self.capital
        risk_frac = self.risk_percent / 100.0

        in_position  = False
        direction    = ""       # "LONG" or "SHORT"
        entry_time: datetime | None = None
        entry_price  = 0.0
        shares       = 0
        tp_price     = 0.0
        sl_price     = 0.0
        vwap_at_entry = 0.0
        band_at_entry = 0.0

        # Resting limit levels carried over from the previous bar, so a fill on
        # the current bar uses band levels known before the bar opened (no
        # lookahead — mirrors Pine's next-bar limit order execution).
        prev_lower = None
        prev_upper = None
        prev_vwap  = None
        prev_day: date | None = None

        def _close_trade(dt: datetime, exit_price: float, reason: str) -> None:
            nonlocal equity, in_position
            if direction == "LONG":
                pnl = shares * (exit_price - entry_price)
            else:
                pnl = shares * (entry_price - exit_price)
            equity += pnl
            duration = int((dt - entry_time).total_seconds() / 60)
            trades.append(VWAPBand3TradeResult(
                symbol=symbol_label,
                option_type=option_type,
                direction=direction,
                strike=strike,
                expiry_date=expiry_date,
                entry_time=entry_time,
                exit_time=dt,
                entry_price=round(entry_price, 2),
                exit_price=round(exit_price, 2),
                shares=shares,
                pnl=round(pnl, 2),
                exit_reason=reason,
                vwap_at_entry=round(vwap_at_entry, 4),
                band_at_entry=round(band_at_entry, 4),
                target_price=round(tp_price, 2),
                stop_price=round(sl_price, 2),
                duration_minutes=duration,
            ))
            in_position = False

        for _, row in bands.iterrows():
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            today: date  = dt.date()

            if today != prev_day:
                # New session: bands reset with the VWAP anchor — drop any
                # resting limit levels carried over from yesterday.
                prev_lower = prev_upper = prev_vwap = None
                prev_day = today

            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            vwap  = row["vwap"]
            hours_since_start = (
                dt - row["session_start"]
            ).total_seconds() / 3600.0
            can_trade = hours_since_start >= self.trading_delay_hours

            # ── Exit logic (checked intrabar while in position) ────────────
            if in_position:
                if t >= _SQUARE_OFF:
                    _close_trade(dt, c, "SQUARE_OFF")
                else:
                    if direction == "LONG":
                        hit_sl = l <= sl_price
                        hit_tp = h >= tp_price
                        # Gap through a level fills at the open.
                        sl_fill = min(o, sl_price)
                        tp_fill = tp_price if o < tp_price else o
                    else:
                        hit_sl = h >= sl_price
                        hit_tp = l <= tp_price
                        sl_fill = max(o, sl_price)
                        tp_fill = tp_price if o > tp_price else o

                    # Conservative: when a single bar spans both levels,
                    # assume the stop-loss filled first.
                    if hit_sl:
                        _close_trade(dt, sl_fill, "STOP_LOSS")
                    elif hit_tp:
                        _close_trade(dt, tp_fill, "TARGET")

            # ── Entry logic (resting limits at previous bar's bands) ───────
            if (
                not in_position
                and can_trade
                and prev_lower is not None
                and t <= _ENTRY_CUTOFF
                and daily_trade_count[today] < _MAX_TRADES_PER_DAY
            ):
                long_dist  = prev_vwap - prev_lower
                short_dist = prev_upper - prev_vwap

                fill_dir = ""
                if l <= prev_lower and long_dist > 0:
                    fill_dir   = "LONG"
                    fill_price = min(o, prev_lower)
                    risk_dist  = long_dist
                    band_level = prev_lower
                elif self.allow_short and h >= prev_upper and short_dist > 0:
                    fill_dir   = "SHORT"
                    fill_price = max(o, prev_upper)
                    risk_dist  = short_dist
                    band_level = prev_upper

                if fill_dir and fill_price > 0:
                    qty = floor(equity * risk_frac / risk_dist)
                    if qty >= 1:
                        in_position   = True
                        direction     = fill_dir
                        entry_time    = dt
                        entry_price   = fill_price
                        shares        = qty
                        vwap_at_entry = vwap
                        band_at_entry = band_level
                        # TP at the VWAP of the fill bar, fixed for the life
                        # of the trade; SL mirrored 1:1 on the other side.
                        tp_price = vwap
                        dist     = abs(tp_price - entry_price)
                        sl_price = (
                            entry_price - dist if fill_dir == "LONG"
                            else entry_price + dist
                        )
                        daily_trade_count[today] += 1

                        # If the same bar already spans the fresh TP/SL,
                        # resolve it immediately (stop-loss first).
                        if fill_dir == "LONG":
                            if l <= sl_price:
                                _close_trade(dt, sl_price, "STOP_LOSS")
                            elif h >= tp_price:
                                _close_trade(dt, tp_price, "TARGET")
                        else:
                            if h >= sl_price:
                                _close_trade(dt, sl_price, "STOP_LOSS")
                            elif l <= tp_price:
                                _close_trade(dt, tp_price, "TARGET")

            prev_lower = row["lower_band"]
            prev_upper = row["upper_band"]
            prev_vwap  = vwap

        # Force-close any open position at end of data
        if in_position:
            last = bands.iloc[-1]
            _close_trade(last["datetime"], last["close"], "SQUARE_OFF")

        return trades

    # ── Debug / inspection helpers ────────────────────────────────────────────

    def _print_resampled_with_trades(
        self,
        candles: list[dict],
        trades: list[VWAPBand3TradeResult],
        strike: int,
        option_type: str,
        expiry_date: date,
    ) -> None:
        symbol_label = f"NIFTY{strike}{option_type}"
        print(f"\n{'='*90}")
        print(f"  RESAMPLED DATA + TRADES — {symbol_label}  (expiry {expiry_date})")
        print(f"{'='*90}")

        if not candles:
            print("  (no candle data)")
            return

        bands = compute_vwap_bands(candles, self.band_mult, self.calc_mode)

        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", None,
        ):
            print(bands.drop(columns=["session_start"]).to_string(index=False))

        print(f"\n  Trades for {symbol_label}: {len(trades)}")
        for t in sorted(trades, key=lambda x: x.entry_time):
            pnl_sign = "+" if t.pnl >= 0 else ""
            print(
                f"    {t.entry_time.strftime('%d-%b %H:%M:%S')} → "
                f"{t.exit_time.strftime('%d-%b %H:%M:%S')}"
                f"  {t.direction:<5} entry {t.entry_price:.2f}"
                f"  exit {t.exit_price:.2f}  tp {t.target_price:.2f}"
                f"  sl {t.stop_price:.2f}  qty {t.shares}"
                f"  pnl {pnl_sign}{t.pnl:.2f}  ({t.exit_reason})"
            )

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[VWAPBand3WeeklyExpiryResult]:
        expiry_results: list[VWAPBand3WeeklyExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        effective_tf = (
            f"{self.resample_seconds}s" if self.resample_seconds > 1 else self.interval
        )
        print(f"\n{'='*70}")
        print(f"  NIFTY VWAP BAND 3 MEAN-REVERSION — WEEKLY EXPIRY BACKTEST")
        print(f"  Period    : {self.start_date}  →  {self.end_date}")
        print(f"  Fetch TF  : {self.interval}  |  Effective TF: {effective_tf}")
        print(f"  Capital   : ₹{self.capital:,.0f}  |  Band mult: {self.band_mult}"
              f"  |  Mode: {self.calc_mode}")
        print(f"  Risk/trade: {self.risk_percent}%  |  Entry delay: "
              f"{self.trading_delay_hours}h  |  Shorts: {self.allow_short}")
        print(f"  Expiries found: {len(wednesdays)}")
        print(f"{'='*70}\n")

        for tuesday in wednesdays:
            monday       = NiftyOptionService.monday_of_week(tuesday)
            win_start, _ = NiftyOptionService.week_window(tuesday)

            expiry = NiftyOptionService.adjust_expiry_for_holidays(
                tuesday, self.market_holidays
            )
            if expiry != tuesday:
                print(f"  [holiday] Expiry {tuesday} is a holiday — rolled back to {expiry}")
            win_end = expiry

            if self.per_day_atm:
                week_result = self._run_expiry_per_day(expiry, win_start, win_end)
            else:
                week_result = self._run_expiry_weekly(expiry, monday, win_start, win_end)

            if week_result is not None:
                expiry_results.append(week_result)

        return expiry_results

    def _run_expiry_weekly(
        self, expiry: date, monday: date, win_start: date, win_end: date
    ) -> VWAPBand3WeeklyExpiryResult | None:
        """
        Single ATM strike for the whole expiry week, anchored to the week's
        Monday open, traded across the full window.
        """
        try:
            nifty_open = self.nifty_service.get_nifty_open(monday)
        except Exception as exc:
            print(f"  [{expiry}] Could not get Nifty open for {monday}: {exc}")
            return None

        strike = NiftyOptionService.atm_strike(nifty_open)

        print(f"  Expiry {expiry}  |  Monday open {nifty_open:.2f}  |  ATM {strike}")

        week_result = VWAPBand3WeeklyExpiryResult(
            expiry_date=expiry,
            atm_strike=strike,
            nifty_open=nifty_open,
        )

        from_dt = datetime(win_start.year, win_start.month, win_start.day, 9, 15, 0)
        to_dt   = datetime(win_end.year,   win_end.month,   win_end.day,   15, 30, 0)

        if self.cache_only:
            missing = False
            for opt_type in ("CE", "PE"):
                cached = self.nifty_service.get_option_candles(
                    strike=strike,
                    expiry_date=expiry,
                    option_type=opt_type,
                    start=from_dt,
                    end=to_dt,
                    interval=self.interval,
                    cache_only=True,
                )
                if not cached:
                    missing = True
                    break
            if missing:
                print(f"    [cache-only] No cached data — skipping expiry {expiry}")
                return None

        for opt_type in ("CE", "PE"):
            try:
                candles = self.nifty_service.get_option_candles(
                    strike=strike,
                    expiry_date=expiry,
                    option_type=opt_type,
                    start=from_dt,
                    end=to_dt,
                    interval=self.interval,
                    cache_only=self.cache_only,
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

        return week_result

    def _run_expiry_per_day(
        self, expiry: date, win_start: date, win_end: date
    ) -> VWAPBand3WeeklyExpiryResult | None:
        """
        Per-day ATM mode: for each trading day in the expiry window, choose a
        fresh ATM strike from that day's Nifty open and trade only that day.
        """
        print(f"  Expiry {expiry}  |  per-day ATM")

        week_result = VWAPBand3WeeklyExpiryResult(
            expiry_date=expiry,
            atm_strike=0,
            nifty_open=0.0,
        )

        days = self.nifty_service.trading_days(
            win_start, win_end, self.market_holidays
        )

        for day in days:
            try:
                nifty_open = self.nifty_service.get_nifty_open(day)
            except Exception as exc:
                print(f"    [{day}] Could not get Nifty open: {exc}")
                continue

            strike = NiftyOptionService.atm_strike(nifty_open)
            if day == expiry:
                week_result.atm_strike = strike
                week_result.nifty_open = nifty_open

            from_dt = datetime(day.year, day.month, day.day, 9, 15, 0)
            to_dt   = datetime(day.year, day.month, day.day, 15, 30, 0)

            print(f"    {day}  |  open {nifty_open:.2f}  |  ATM {strike}")

            if self.cache_only:
                missing = False
                for opt_type in ("CE", "PE"):
                    cached = self.nifty_service.get_option_candles(
                        strike=strike,
                        expiry_date=expiry,
                        option_type=opt_type,
                        start=from_dt,
                        end=to_dt,
                        interval=self.interval,
                        cache_only=True,
                    )
                    if not cached:
                        missing = True
                        break
                if missing:
                    print(f"      [cache-only] No cached data — skipping {day}")
                    continue

            for opt_type in ("CE", "PE"):
                try:
                    candles = self.nifty_service.get_option_candles(
                        strike=strike,
                        expiry_date=expiry,
                        option_type=opt_type,
                        start=from_dt,
                        end=to_dt,
                        interval=self.interval,
                        cache_only=self.cache_only,
                    )
                    if self.resample_seconds > 1:
                        candles = resample_candles(candles, self.resample_seconds)
                    trades = self._run_symbol(candles, opt_type, strike, expiry)

                    if opt_type == "CE":
                        week_result.ce_trades.extend(trades)
                    else:
                        week_result.pe_trades.extend(trades)

                    if self.print_resampled:
                        self._print_resampled_with_trades(
                            candles, trades, strike, opt_type, expiry
                        )

                    print(f"      {opt_type}: {len(trades)} trades")
                except Exception as exc:
                    print(f"      [{opt_type}] Error: {exc}")

        return week_result

    # ── Reporting ─────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(expiry_results: list[VWAPBand3WeeklyExpiryResult]) -> None:
        all_trades: list[VWAPBand3TradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[VWAPBand3TradeResult]] = defaultdict(list)
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
                f"  {'Symbol':<22} {'Dir':<5} {'Entry':>15} {'Exit':>15}"
                f" {'Entry₹':>8} {'Exit₹':>8} {'TP₹':>8} {'SL₹':>8} {'Qty':>5}"
                f" {'PnL':>10} {'Reason':<12}"
            )
            print(header)
            print(f"  {sep}")

            for t in sorted(er.all_trades, key=lambda x: x.entry_time):
                pnl_sign = "+" if t.pnl >= 0 else ""
                print(
                    f"  {t.symbol:<22}"
                    f" {t.direction:<5}"
                    f" {t.entry_time.strftime('%d-%b %H:%M'):>15}"
                    f" {t.exit_time.strftime('%d-%b %H:%M'):>15}"
                    f" {t.entry_price:>8.2f}"
                    f" {t.exit_price:>8.2f}"
                    f" {t.target_price:>8.2f}"
                    f" {t.stop_price:>8.2f}"
                    f" {t.shares:>5}"
                    f" {pnl_sign+f'{t.pnl:.2f}':>10}"
                    f" {t.exit_reason:<12}"
                )

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
