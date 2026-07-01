"""
Heikin Ashi Exhaustion + Supertrend strategy for Nifty weekly options (1-second data).

Ported from a "Shinobi HA" style Pine Script that trades Heikin Ashi exhaustion
patterns (a doji-like indecision candle following two same-direction HA candles
with a clean wick on the far side) as pullback reversals within a Supertrend
trend, filtered by ADX trend-strength and a "rising swing lows" structure
filter, with an optional volume surge confirmation.

Like the other per-second option strategies in this repo, the indicators are
computed directly on each option contract's OWN price series (each leg is a
long-only bet on its own price action — a CE reversal-to-bullish signal is a
bullish bet on the index, a PE reversal-to-bullish signal is a bearish bet on
the index). Only the bullish ("buy") exhaustion setup from the source script is
used since options can only be bought, never shorted, in this backtest.

Entry (per leg, independently on CE and PE):
  1. Bar i-2 and bar i-1 are both bearish Heikin Ashi candles with (near) no
     upper wick (a clean two-candle pullback/down-leg).
  2. Bar i (the signal/entry bar) is a Heikin-Ashi doji: its body is a small
     fraction of its total range and both wicks are at least
     `doji_wick_mult` times the body (genuine indecision, not just a small
     range candle).
  3. Bar i's total range is larger than bar i-1's (momentum picking back up).
  4. The Supertrend computed on the contract's own OHLC is bullish.
  5. ADX >= `adx_min` (the move is happening inside an already-trending regime).
  6. The swing-low structure filter confirms rising swing lows (or not enough
     pivots have formed yet to disqualify the trade).
  7. Volume confirmation (optional): the entry bar's volume is at least
     `volume_mult` times its EMA(`volume_ma_len`).

Exit:
  - Stop-loss  : the Heikin Ashi low of the entry (signal) bar.
  - Target     : entry + (entry - stop) * rr_ratio  (1:1 R:R by default).
  - Square-off : forced exit at 15:20 IST.
  - No new entries before 9:30 or after 14:45.
  - Max `max_trades_per_day` entries per day per contract.

Usage:
  1. Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  2. Set RESAMPLE_SECONDS to the desired candle size for the strategy.
  Data is always fetched as 1-second bars from Breeze; resampling is done
  locally before the indicators and strategy logic run.
"""

from collections import defaultdict
from datetime import date, datetime, time
from math import floor

import numpy as np
import pandas as pd

from models.heikin_ashi_supertrend_models import (
    HeikinAshiSupertrendSymbolMetrics,
    HeikinAshiSupertrendTradeResult,
    WeeklyHeikinAshiSupertrendExpiryResult,
)
from services.nifty_option_service import NiftyOptionService
from strategy.adx_option_strategy import compute_adx
from strategy.supertrend_option_strategy import compute_supertrend, resample_candles

_ENTRY_START  = time(9, 30)
_ENTRY_CUTOFF = time(14, 45)
_SQUARE_OFF   = time(15, 20)


def _capital_allocation_pct(price: float) -> float:
    """Fraction of capital to allocate based on the option price."""
    if price <= 20:
        return 0.30
    elif price <= 60:
        return 1.00
    elif price <= 100:
        return 1.00
    else:
        return 1.00


def compute_heikin_ashi(candles: list[dict]) -> pd.DataFrame:
    """
    Compute Heikin Ashi OHLC from a list of raw OHLC dicts. Returns a DataFrame
    with columns: datetime, open, high, low, close (raw) and ha_open, ha_high,
    ha_low, ha_close.
    """
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(df)

    ha_close = (o + h + l + c) / 4.0
    ha_open  = np.empty(n)
    for i in range(n):
        ha_open[i] = (o[i] + c[i]) / 2.0 if i == 0 else (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
    ha_low  = np.minimum(l, np.minimum(ha_open, ha_close))

    result = df[["datetime", "open", "high", "low", "close"]].copy()
    result["ha_open"]  = ha_open
    result["ha_high"]  = ha_high
    result["ha_low"]   = ha_low
    result["ha_close"] = ha_close
    return result


def _swing_bull_ok(ha_high: np.ndarray, ha_low: np.ndarray, swing_len: int) -> np.ndarray:
    """
    Mirror of the Pine Script's `swing_bull_ok` structure filter: True whenever
    the most recently CONFIRMED Heikin Ashi swing low is higher than the swing
    low before it (rising swing lows = bullish structure), or when fewer than
    two swing lows have been confirmed yet (no filter applied).

    A pivot low at index `p` is confirmed `swing_len` bars later, once `p` is
    known to be the lowest ha_low within the symmetric window
    [p - swing_len, p + swing_len] — matching ta.pivotlow(swing_len, swing_len).
    """
    n = len(ha_low)
    bull_ok = np.ones(n, dtype=bool)
    if swing_len <= 0:
        return bull_ok

    last_sl = None
    last_sl2 = None
    for i in range(n):
        pivot_idx = i - swing_len
        if pivot_idx >= swing_len:
            lo = pivot_idx - swing_len
            window = ha_low[lo:i + 1]
            center_pos = pivot_idx - lo
            if window[center_pos] == window.min():
                last_sl2 = last_sl
                last_sl = ha_low[pivot_idx]
        bull_ok[i] = (last_sl2 is None) or (last_sl > last_sl2)
    return bull_ok


def compute_signals(
    candles: list[dict],
    st_period: int,
    st_multiplier: float,
    adx_period: int,
    adx_min: float,
    doji_max_pct: float,
    wick_tolerance_pct: float,
    doji_wick_mult: float,
    use_volume_filter: bool,
    volume_ma_len: int,
    volume_mult: float,
    swing_len: int,
) -> pd.DataFrame:
    """
    Build the full indicator/signal DataFrame for one option contract's own
    price series: Heikin Ashi candles, Supertrend trend, ADX, the swing-low
    structure filter, the volume filter and the bullish HA exhaustion pattern,
    combined into a single `long_signal` column.
    """
    ha  = compute_heikin_ashi(candles)
    st  = compute_supertrend(candles, st_period, st_multiplier)
    adx = compute_adx(candles, adx_period)

    merged = ha.merge(st[["datetime", "trend"]], on="datetime", how="left")
    merged = merged.merge(adx[["datetime", "adx", "volume"]], on="datetime", how="left")

    # ── Volume filter ────────────────────────────────────────────────────
    if use_volume_filter and "volume" in merged.columns and merged["volume"].notna().any():
        vol_ema = merged["volume"].ewm(span=max(volume_ma_len, 1), adjust=False).mean()
        vol_ok = (merged["volume"] >= volume_mult * vol_ema).fillna(False)
    else:
        vol_ok = pd.Series(True, index=merged.index)
    merged["vol_ok"] = vol_ok

    # ── ADX filter ───────────────────────────────────────────────────────
    merged["adx_ok"] = (merged["adx"] >= adx_min).fillna(False)

    # ── Heikin Ashi candle geometry ─────────────────────────────────────
    ha_open  = merged["ha_open"].to_numpy()
    ha_close = merged["ha_close"].to_numpy()
    ha_high  = merged["ha_high"].to_numpy()
    ha_low   = merged["ha_low"].to_numpy()

    total = ha_high - ha_low
    body  = np.abs(ha_close - ha_open)
    upper_wick = ha_high - np.maximum(ha_close, ha_open)
    lower_wick = np.minimum(ha_close, ha_open) - ha_low
    bearish = ha_close < ha_open

    upper_ratio = np.where(total > 0, upper_wick / np.where(total > 0, total, 1.0), 0.0)
    body_pct    = np.where(total > 0, body / np.where(total > 0, total, 1.0), 1.0)

    no_upper_wick = upper_ratio <= wick_tolerance_pct
    both_wicks_ok = (upper_wick >= body * doji_wick_mult) & (lower_wick >= body * doji_wick_mult)
    is_doji = (body_pct <= doji_max_pct) & both_wicks_ok

    s_bearish   = pd.Series(bearish)
    s_no_upper  = pd.Series(no_upper_wick)
    s_total     = pd.Series(total)

    bearish1  = s_bearish.shift(1).fillna(False).to_numpy()
    bearish2  = s_bearish.shift(2).fillna(False).to_numpy()
    no_upper1 = s_no_upper.shift(1).fillna(False).to_numpy()
    no_upper2 = s_no_upper.shift(2).fillna(False).to_numpy()
    size_up   = (s_total > s_total.shift(1)).fillna(False).to_numpy()

    buy_setup = bearish2 & no_upper2 & bearish1 & no_upper1 & is_doji & size_up
    merged["buy_setup"] = buy_setup

    # ── Structure filter (rising swing lows) ────────────────────────────
    merged["struct_bull"] = _swing_bull_ok(ha_high, ha_low, swing_len)

    trend_ok = merged["trend"] == 1

    merged["long_signal"] = (
        merged["buy_setup"]
        & trend_ok
        & merged["adx_ok"]
        & merged["struct_bull"]
        & merged["vol_ok"]
    )

    return merged


def _compute_metrics(
    symbol: str, trades: list[HeikinAshiSupertrendTradeResult]
) -> HeikinAshiSupertrendSymbolMetrics:
    if not trades:
        return HeikinAshiSupertrendSymbolMetrics(
            symbol=symbol, total_trades=0, wins=0, losses=0,
            win_rate=0.0, total_pnl=0.0, avg_pnl=0.0,
            profit_factor=0.0, best_trade=0.0, worst_trade=0.0,
            avg_duration_minutes=0.0, max_consecutive_losses=0,
        )

    pnls   = [t.pnl for t in trades]
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

    return HeikinAshiSupertrendSymbolMetrics(
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


class HeikinAshiSupertrendOptionStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        st_period: int = 15,
        st_multiplier: float = 3.44,
        adx_period: int = 14,
        adx_min: float = 22.0,
        doji_max_pct: float = 20.0,
        wick_tolerance_pct: float = 3.0,
        doji_wick_mult: float = 0.3,
        rr_ratio: float = 1.0,
        use_volume_filter: bool = True,
        volume_ma_len: int = 20,
        volume_mult: float = 1.0,
        swing_len: int = 5,
        start_date: str = "",
        end_date: str = "",
        interval: str = "1second",
        resample_seconds: int = 1,
        max_trades_per_day: int = 5,
        print_resampled: bool = False,
        cache_only: bool = False,
        market_holidays: set[date] | None = None,
        per_day_atm: bool = False,
    ):
        self.nifty_service  = nifty_service
        self.capital        = capital
        # Supertrend trend filter, computed on the option's own OHLC.
        self.st_period      = st_period
        self.st_multiplier  = st_multiplier
        # ADX trend-strength filter.
        self.adx_period     = adx_period
        self.adx_min        = adx_min
        # Heikin Ashi doji detection. doji_max_pct/wick_tolerance_pct are
        # percentages (e.g. 20.0 = 20%) and are converted to fractions
        # internally to match the Pine Script's `/ 100` inputs.
        self.doji_max_pct        = doji_max_pct / 100.0
        self.wick_tolerance_pct  = wick_tolerance_pct / 100.0
        self.doji_wick_mult      = doji_wick_mult
        # Risk:reward ratio for the fixed target. 1.0 = 1:1.
        self.rr_ratio        = rr_ratio
        # Volume surge confirmation on the entry bar.
        self.use_volume_filter = use_volume_filter
        self.volume_ma_len     = volume_ma_len
        self.volume_mult       = volume_mult
        # Swing high/low structure filter sensitivity.
        self.swing_len       = swing_len
        self.start_date      = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date        = datetime.strptime(end_date,   "%d-%b-%Y").date()
        self.interval        = interval
        self.resample_seconds = resample_seconds
        self.max_trades_per_day = max_trades_per_day
        self.print_resampled = print_resampled
        self.cache_only      = cache_only
        self.market_holidays = set(market_holidays) if market_holidays else set()
        self.per_day_atm     = per_day_atm

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _run_symbol(
        self,
        candles: list[dict],
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[HeikinAshiSupertrendTradeResult]:
        """Run the HA exhaustion + Supertrend strategy on one contract's candles."""
        if not candles:
            return []

        ind = compute_signals(
            candles,
            self.st_period, self.st_multiplier,
            self.adx_period, self.adx_min,
            self.doji_max_pct, self.wick_tolerance_pct, self.doji_wick_mult,
            self.use_volume_filter, self.volume_ma_len, self.volume_mult,
            self.swing_len,
        )
        symbol_label = f"NIFTY{strike}{option_type}"
        trades: list[HeikinAshiSupertrendTradeResult] = []

        daily_trade_count: dict[date, int] = defaultdict(int)

        in_position = False
        entry_row   = None
        entry_price = 0.0
        stop_loss   = 0.0
        target      = 0.0
        shares      = 0

        for _, row in ind.iterrows():
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            today: date  = dt.date()

            high  = row["high"]
            low   = row["low"]
            close = row["close"]

            # ── Exit logic ───────────────────────────────────────────────
            if in_position:
                exit_reason = None
                exit_price  = close

                if t >= _SQUARE_OFF:
                    exit_reason = "SQUARE_OFF"
                    exit_price  = close
                elif low <= stop_loss:
                    exit_reason = "STOP_LOSS"
                    exit_price  = stop_loss
                elif high >= target:
                    exit_reason = "TARGET"
                    exit_price  = target

                if exit_reason:
                    duration = int((dt - entry_row["datetime"]).total_seconds() / 60)
                    pnl      = round(shares * (exit_price - entry_price), 2)
                    trades.append(HeikinAshiSupertrendTradeResult(
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
                        stop_loss=stop_loss,
                        target=target,
                        adx_at_entry=entry_row["adx"],
                        duration_minutes=duration,
                    ))
                    in_position = False
                    entry_row   = None

            # ── Entry logic ───────────────────────────────────────────────
            if (
                not in_position
                and _ENTRY_START <= t <= _ENTRY_CUTOFF
                and daily_trade_count[today] < self.max_trades_per_day
                and bool(row["long_signal"])
                and close > 0
            ):
                candidate_stop = row["ha_low"]
                risk = close - candidate_stop
                if risk > 0:
                    entry_price = close
                    stop_loss   = candidate_stop
                    target      = entry_price + risk * self.rr_ratio
                    alloc_pct   = _capital_allocation_pct(entry_price)
                    shares      = max(floor(self.capital * alloc_pct / entry_price), 1)
                    in_position = True
                    entry_row   = row
                    daily_trade_count[today] += 1

        # Force-close any open position at end of data.
        if in_position and entry_row is not None:
            last  = ind.iloc[-1]
            price = last["close"]
            dt    = last["datetime"]
            duration = int((dt - entry_row["datetime"]).total_seconds() / 60)
            pnl      = round(shares * (price - entry_price), 2)
            trades.append(HeikinAshiSupertrendTradeResult(
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
                stop_loss=stop_loss,
                target=target,
                adx_at_entry=entry_row["adx"],
                duration_minutes=duration,
            ))

        return trades

    # ── Debug / inspection helpers ────────────────────────────────────────────

    def _print_resampled_with_trades(
        self,
        candles: list[dict],
        trades: list[HeikinAshiSupertrendTradeResult],
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

        ind = compute_signals(
            candles,
            self.st_period, self.st_multiplier,
            self.adx_period, self.adx_min,
            self.doji_max_pct, self.wick_tolerance_pct, self.doji_wick_mult,
            self.use_volume_filter, self.volume_ma_len, self.volume_mult,
            self.swing_len,
        )

        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", None,
        ):
            print(ind.to_string(index=False))

        print(f"\n  Trades for {symbol_label}: {len(trades)}")
        for t in sorted(trades, key=lambda x: x.entry_time):
            pnl_sign = "+" if t.pnl >= 0 else ""
            print(
                f"    {t.entry_time.strftime('%d-%b %H:%M:%S')} → "
                f"{t.exit_time.strftime('%d-%b %H:%M:%S')}"
                f"  entry {t.entry_price:.2f}  sl {t.stop_loss:.2f}  tgt {t.target:.2f}"
                f"  exit {t.exit_price:.2f}"
                f"  qty {t.shares}  pnl {pnl_sign}{t.pnl:.2f}"
                f"  ({t.exit_reason})"
            )

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[WeeklyHeikinAshiSupertrendExpiryResult]:
        expiry_results: list[WeeklyHeikinAshiSupertrendExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        effective_tf = (
            f"{self.resample_seconds}s" if self.resample_seconds > 1 else self.interval
        )
        print(f"\n{'='*70}")
        print(f"  NIFTY HEIKIN ASHI EXHAUSTION + SUPERTREND STRATEGY — WEEKLY EXPIRY BACKTEST")
        print(f"  Period    : {self.start_date}  →  {self.end_date}")
        print(f"  Fetch TF  : {self.interval}  |  Effective TF: {effective_tf}")
        print(f"  Capital   : ₹{self.capital:,.0f}  |  ST period: {self.st_period}"
              f"  |  ST mult: {self.st_multiplier}  |  ADX min: {self.adx_min}"
              f"  |  RR: 1:{self.rr_ratio}")
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
    ) -> WeeklyHeikinAshiSupertrendExpiryResult | None:
        try:
            nifty_open = self.nifty_service.get_nifty_open(monday)
        except Exception as exc:
            print(f"  [{expiry}] Could not get Nifty open for {monday}: {exc}")
            return None

        strike = NiftyOptionService.atm_strike(nifty_open)
        print(f"  Expiry {expiry}  |  Monday open {nifty_open:.2f}  |  ATM {strike}")

        week_result = WeeklyHeikinAshiSupertrendExpiryResult(
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
                    strike=strike, expiry_date=expiry, option_type=opt_type,
                    start=from_dt, end=to_dt, interval=self.interval, cache_only=True,
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
                    strike=strike, expiry_date=expiry, option_type=opt_type,
                    start=from_dt, end=to_dt, interval=self.interval,
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
    ) -> WeeklyHeikinAshiSupertrendExpiryResult | None:
        print(f"  Expiry {expiry}  |  per-day ATM")

        week_result = WeeklyHeikinAshiSupertrendExpiryResult(
            expiry_date=expiry, atm_strike=0, nifty_open=0.0,
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
                        strike=strike, expiry_date=expiry, option_type=opt_type,
                        start=from_dt, end=to_dt, interval=self.interval, cache_only=True,
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
                        strike=strike, expiry_date=expiry, option_type=opt_type,
                        start=from_dt, end=to_dt, interval=self.interval,
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
    def print_report(expiry_results: list[WeeklyHeikinAshiSupertrendExpiryResult]) -> None:
        all_trades: list[HeikinAshiSupertrendTradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[HeikinAshiSupertrendTradeResult]] = defaultdict(list)
        for t in all_trades:
            by_symbol[t.symbol].append(t)

        sep = "─" * 96
        print(f"\n{'='*96}")
        print("  RESULTS BY EXPIRY")
        print(f"{'='*96}")

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
                f"  {'Symbol':<22} {'Entry':>16} {'Exit':>16}"
                f" {'Entry₹':>8} {'SL₹':>8} {'Tgt₹':>8} {'Exit₹':>8} {'Qty':>5}"
                f" {'PnL':>10} {'Reason':<12}"
            )
            print(header)
            print(f"  {sep}")

            for t in sorted(er.all_trades, key=lambda x: x.entry_time):
                pnl_sign = "+" if t.pnl >= 0 else ""
                print(
                    f"  {t.symbol:<22}"
                    f" {t.entry_time.strftime('%d-%b %H:%M'):>16}"
                    f" {t.exit_time.strftime('%d-%b %H:%M'):>16}"
                    f" {t.entry_price:>8.2f}"
                    f" {t.stop_loss:>8.2f}"
                    f" {t.target:>8.2f}"
                    f" {t.exit_price:>8.2f}"
                    f" {t.shares:>5}"
                    f" {pnl_sign+f'{t.pnl:.2f}':>10}"
                    f" {t.exit_reason:<12}"
                )

        # Per-symbol metrics
        print(f"\n{'='*96}")
        print("  SYMBOL-WISE METRICS")
        print(f"{'='*96}")

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
        print(f"  {'─'*94}")

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
        print(f"  {'─'*94}")
        print(
            f"  {'OVERALL':<{col_w['sym']}} {overall_trades:>{col_w['trades']}}"
            f" {overall_wins:>{col_w['wins']}} {overall_losses:>{col_w['loss']}}"
            f" {wr_overall:>{col_w['wr']}.1f} {pnl_s:>{col_w['pnl']}}"
        )
        print(f"{'='*96}\n")
