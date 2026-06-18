"""
Fibonacci Retracement strategy for Nifty weekly options on 1-second data.

This is the options analogue of an algorithmic Fibonacci-retracement system,
run directly on each option contract's own price series (resampled to N-second
candles). Buying an option is an inherently long/directional bet, so the
strategy only ever looks for LONG entries on bullish pullbacks of the option's
own price (a CE long = bullish on the index, a PE long = bearish on the index).

How the algorithm works
=======================
1. Trend detection
   A simple moving average (`ma_period`) defines the overarching trend. A long
   entry is only considered while the price trades ABOVE its moving average
   (an established uptrend in the option's own price).

2. Swing identification
   Over a rolling window of the last `swing_lookback` candles the system tracks
   the significant Swing High (peak) and Swing Low (trough) of the recent move.

3. Level calculation (uptrend retracement)
   For an uptrend the retracement price for a ratio r is:
       level = swing_high - (swing_high - swing_low) * r
   The key ratios are 38.2%, 50%, 61.8% (the "Golden Zone") and 78.6%.

Entry (long option only)
========================
A buy is taken when the price corrects into the Golden Zone — i.e. it pulls back
to at least the 50% level but has not broken below the 61.8% (or the configured
`entry_ratio`) level — AND a confirmation trigger fires:

  - Trend filter : close is above the moving average.
  - Golden Zone  : the candle's low dips to/through the entry Fibonacci level
                   while the close holds above the 78.6% level (the pullback is
                   not a full reversal).
  - Momentum     : RSI confirms a bounce out of oversold — RSI crosses back
                   above `rsi_oversold` (or, when `rsi_cross_required` is False,
                   simply trades above it) — optionally combined with a bullish
                   candle (close > open).
  - Volume       : confluence — the candle's volume is at least `volume_factor`
                   times the recent average volume (set 0 to disable).

Exit
====
  - Stop-loss    : programmed just beyond the next Fibonacci level — slightly
                   below the 78.6% retracement (`stop_ratio`). Falls back to a
                   hard `stop_loss_pct` below entry if the Fib stop is invalid.
  - Profit target: a Fibonacci extension — target = swing_low + range *
                   `extension_ratio` (e.g. 127.2% / 161.8%), i.e. beyond the
                   prior swing high.
  - Trailing stop: optional; ratchets up with the peak price.
  - Break-even   : optional; once price moves `breakeven_trigger_pct` above
                   entry the stop is moved up to entry, optionally booking 50%.
  - Square-off   : forced exit at 15:20 IST.
  - No new entries before the warm-up completes or after 14:45.
  - Max `max_trades_per_day` entries per day per contract.
"""

from collections import defaultdict
from datetime import date, datetime, time
from math import floor

import pandas as pd

from models.fibonacci_seconds_models import (
    FibSecondsTradeResult,
    FibSymbolMetrics,
    WeeklyFibExpiryResult,
)
from services.nifty_option_service import NiftyOptionService
from strategy.adx_option_strategy import resample_candles


_ENTRY_CUTOFF = time(14, 45)
_SQUARE_OFF   = time(15, 20)


def _capital_allocation_pct(price: float) -> float:
    """
    Return the fraction of capital to allocate based on option price.
    Mirrors the ORB / ADX seconds strategy allocation buckets.
    """
    if price <= 20:
        return 0.30
    return 1.00


def _compute_metrics(symbol: str, trades: list[FibSecondsTradeResult]) -> FibSymbolMetrics:
    if not trades:
        return FibSymbolMetrics(
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

    return FibSymbolMetrics(
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


class FibonacciOptionSecondsStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        ma_period: int = 50,
        swing_lookback: int = 60,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_cross_required: bool = True,
        require_bullish_candle: bool = True,
        entry_ratio: float = 0.618,
        golden_zone_start: float = 0.5,
        stop_ratio: float = 0.786,
        extension_ratio: float = 1.272,
        volume_factor: float = 1.5,
        volume_avg_period: int = 60,
        stop_loss_pct: float = 25.0,
        start_date: str = "",
        end_date: str = "",
        interval: str = "1second",
        resample_seconds: int = 5,
        max_trades_per_day: int = 5,
        print_resampled: bool = False,
        cache_only: bool = False,
        market_holidays: set[date] | None = None,
        per_day_atm: bool = False,
        trailing_stop_enabled: bool = False,
        trailing_stop_pct: float = 0.0,
        breakeven_enabled: bool = False,
        breakeven_trigger_pct: float = 5.0,
        breakeven_partial_book_enabled: bool = True,
    ):
        self.nifty_service    = nifty_service
        self.capital          = capital
        # Moving-average period used for trend detection (uptrend = close > MA).
        self.ma_period        = max(int(ma_period), 1)
        # Rolling window (in candles) over which the recent swing high / swing low
        # of the move are identified.
        self.swing_lookback   = max(int(swing_lookback), 2)
        # RSI configuration used as the momentum/oversold confirmation.
        self.rsi_period       = max(int(rsi_period), 2)
        self.rsi_oversold     = rsi_oversold
        # When True, require RSI to CROSS back above the oversold threshold on the
        # entry candle; when False, RSI simply needs to be above the threshold.
        self.rsi_cross_required = rsi_cross_required
        # When True, additionally require a bullish (close > open) entry candle.
        self.require_bullish_candle = require_bullish_candle
        # Fibonacci ratios. `entry_ratio` is the retracement the buy is taken at
        # (the deep end of the Golden Zone, 61.8% by default); `golden_zone_start`
        # is the shallow end (50%). `stop_ratio` is the level the stop sits just
        # beyond (78.6%). `extension_ratio` is the profit-target extension (e.g.
        # 127.2% / 161.8%).
        self.entry_ratio      = entry_ratio
        self.golden_zone_start = golden_zone_start
        self.stop_ratio       = stop_ratio
        self.extension_ratio  = extension_ratio
        # Volume confluence: the entry candle's volume must be at least
        # `volume_factor` times the trailing average volume over `volume_avg_period`
        # candles. Set `volume_factor` to 0 to disable the volume filter.
        self.volume_factor    = volume_factor
        self.volume_avg_period = max(int(volume_avg_period), 1)
        # Hard stop-loss fallback (percent below entry) used when the Fibonacci
        # 78.6% stop is not strictly below the entry price.
        self.stop_loss_pct    = stop_loss_pct
        self.start_date       = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date         = datetime.strptime(end_date,   "%d-%b-%Y").date()
        self.interval         = interval
        self.resample_seconds = resample_seconds
        self.max_trades_per_day = max_trades_per_day
        self.print_resampled  = print_resampled
        self.cache_only       = cache_only
        self.market_holidays  = set(market_holidays) if market_holidays else set()
        self.per_day_atm      = per_day_atm
        # Trailing stop-loss. Tracks the highest option price (peak) since entry
        # and exits if the price falls `trailing_stop_pct` percent from that peak.
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_pct     = trailing_stop_pct
        # Break-even stop (see ORB seconds strategy for the shared semantics).
        self.breakeven_enabled     = breakeven_enabled
        self.breakeven_trigger_pct = breakeven_trigger_pct
        self.breakeven_partial_book_enabled = breakeven_partial_book_enabled

    # ── Indicators ──────────────────────────────────────────────────────────────

    def _prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach SMA, RSI, rolling swing high/low and average volume columns."""
        df = df.copy()
        df["sma"] = df["close"].rolling(self.ma_period).mean()

        # Wilder's RSI.
        delta = df["close"].diff()
        gain  = delta.clip(lower=0.0)
        loss  = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.rsi_period, min_periods=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0.0, pd.NA)
        df["rsi"] = (100.0 - 100.0 / (1.0 + rs)).fillna(100.0)

        df["swing_high"] = df["high"].rolling(self.swing_lookback).max()
        df["swing_low"]  = df["low"].rolling(self.swing_lookback).min()
        df["avg_volume"] = df["volume"].rolling(self.volume_avg_period).mean()
        return df

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _run_symbol(
        self,
        candles: list[dict],
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[FibSecondsTradeResult]:
        if not candles:
            return []

        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].astype(float)
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        else:
            df["volume"] = 0.0

        symbol_label = f"NIFTY{strike}{option_type}"
        trades: list[FibSecondsTradeResult] = []

        for _today, day_df in df.groupby(df["datetime"].dt.date):
            trades.extend(
                self._run_day(day_df, option_type, strike, expiry_date, symbol_label)
            )

        return trades

    def _run_day(
        self,
        day_df: pd.DataFrame,
        option_type: str,
        strike: int,
        expiry_date: date,
        symbol_label: str,
    ) -> list[FibSecondsTradeResult]:
        day_df = day_df.sort_values("datetime").reset_index(drop=True)
        if day_df.empty:
            return []

        df = self._prepare_indicators(day_df)

        trades: list[FibSecondsTradeResult] = []
        trade_count = 0

        in_position = False
        entry_price = 0.0
        entry_dt    = None
        shares      = 0
        target      = 0.0
        stop_loss   = 0.0
        peak_price  = 0.0
        moved_to_breakeven = False
        entry_swing_high = 0.0
        entry_swing_low  = 0.0
        entry_fib_level  = 0.0
        entry_rsi        = 0.0
        breakout_vol     = 0.0
        avg_vol_entry    = 0.0
        vol_ratio        = 0.0

        prev_rsi = None

        for _, row in df.iterrows():
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            open_  = float(row["open"])
            high   = float(row["high"])
            low    = float(row["low"])
            close  = float(row["close"])
            volume = float(row["volume"])

            sma        = row["sma"]
            rsi        = row["rsi"]
            swing_high = row["swing_high"]
            swing_low  = row["swing_low"]
            avg_volume = row["avg_volume"]

            # ── Exit logic ────────────────────────────────────────────────
            if in_position:
                exit_reason = None
                exit_price  = close

                if high > peak_price:
                    peak_price = high

                # Break-even: ratchet stop up to entry, optionally booking 50%.
                if (
                    self.breakeven_enabled
                    and not moved_to_breakeven
                    and high >= entry_price * (1 + self.breakeven_trigger_pct / 100.0)
                ):
                    stop_loss = entry_price
                    moved_to_breakeven = True

                    book_price  = round(
                        entry_price * (1 + self.breakeven_trigger_pct / 100.0), 2
                    )
                    half_shares = shares // 2
                    if self.breakeven_partial_book_enabled and half_shares >= 1:
                        duration = int((dt - entry_dt).total_seconds() / 60)
                        pnl      = round(half_shares * (book_price - entry_price), 2)
                        trades.append(FibSecondsTradeResult(
                            symbol=symbol_label,
                            option_type=option_type,
                            strike=strike,
                            expiry_date=expiry_date,
                            entry_time=entry_dt,
                            exit_time=dt,
                            entry_price=entry_price,
                            exit_price=book_price,
                            shares=half_shares,
                            pnl=pnl,
                            exit_reason="PARTIAL_BOOK",
                            swing_high=entry_swing_high,
                            swing_low=entry_swing_low,
                            fib_entry_level=entry_fib_level,
                            fib_ratio=self.entry_ratio,
                            rsi_at_entry=round(entry_rsi, 2),
                            breakout_volume=breakout_vol,
                            avg_volume=round(avg_vol_entry, 2),
                            volume_ratio=round(vol_ratio, 2),
                            duration_minutes=duration,
                        ))
                        shares -= half_shares

                # Square-off at 15:20
                if t >= _SQUARE_OFF:
                    exit_reason = "SQUARE_OFF"
                    exit_price  = close
                # Hard stop-loss (or break-even after the stop has moved to entry)
                elif low <= stop_loss:
                    exit_reason = "BREAKEVEN" if moved_to_breakeven else "STOP_LOSS"
                    exit_price  = stop_loss
                # Trailing stop-loss
                elif (
                    self.trailing_stop_enabled
                    and self.trailing_stop_pct > 0
                    and peak_price > 0
                    and low <= peak_price * (1 - self.trailing_stop_pct / 100.0)
                ):
                    exit_reason = "TRAILING_STOP"
                    exit_price  = round(peak_price * (1 - self.trailing_stop_pct / 100.0), 2)
                # Fibonacci-extension target
                elif high >= target:
                    exit_reason = "TARGET"
                    exit_price  = target

                if exit_reason:
                    duration = int((dt - entry_dt).total_seconds() / 60)
                    pnl      = round(shares * (exit_price - entry_price), 2)
                    trades.append(FibSecondsTradeResult(
                        symbol=symbol_label,
                        option_type=option_type,
                        strike=strike,
                        expiry_date=expiry_date,
                        entry_time=entry_dt,
                        exit_time=dt,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        shares=shares,
                        pnl=pnl,
                        exit_reason=exit_reason,
                        swing_high=entry_swing_high,
                        swing_low=entry_swing_low,
                        fib_entry_level=entry_fib_level,
                        fib_ratio=self.entry_ratio,
                        rsi_at_entry=round(entry_rsi, 2),
                        breakout_volume=breakout_vol,
                        avg_volume=round(avg_vol_entry, 2),
                        volume_ratio=round(vol_ratio, 2),
                        duration_minutes=duration,
                    ))
                    in_position = False
                    entry_dt    = None
                    prev_rsi    = rsi if not pd.isna(rsi) else prev_rsi
                    continue

            # ── Entry logic ───────────────────────────────────────────────
            if (
                not in_position
                and t <= _ENTRY_CUTOFF
                and trade_count < self.max_trades_per_day
                and not pd.isna(sma)
                and not pd.isna(rsi)
                and not pd.isna(swing_high)
                and not pd.isna(swing_low)
            ):
                swing_high = float(swing_high)
                swing_low  = float(swing_low)
                price_range = swing_high - swing_low

                if price_range > 0 and close > 0:
                    # Uptrend retracement Fibonacci levels.
                    fib_entry  = swing_high - price_range * self.entry_ratio
                    fib_zone_hi = swing_high - price_range * self.golden_zone_start
                    fib_stop   = swing_high - price_range * self.stop_ratio
                    target_px  = swing_low + price_range * self.extension_ratio

                    # 1. Trend filter — price above its moving average.
                    trend_ok = close > float(sma)

                    # 2. Golden-Zone pullback: the candle dipped into the zone
                    #    (low reached the 50% level or deeper) but the close still
                    #    holds above the 78.6% level (not a full reversal).
                    zone_ok = low <= fib_zone_hi and close >= fib_stop

                    # 3. Momentum confirmation via RSI bouncing out of oversold.
                    rsi_val = float(rsi)
                    if self.rsi_cross_required:
                        rsi_ok = (
                            prev_rsi is not None
                            and prev_rsi <= self.rsi_oversold
                            and rsi_val > self.rsi_oversold
                        )
                    else:
                        rsi_ok = rsi_val > self.rsi_oversold

                    # Optional bullish-candle confirmation.
                    candle_ok = (close > open_) if self.require_bullish_candle else True

                    # 4. Volume confluence.
                    if self.volume_factor <= 0 or pd.isna(avg_volume) or avg_volume <= 0:
                        volume_ok = self.volume_factor <= 0
                        avg_v     = 0.0 if pd.isna(avg_volume) else float(avg_volume)
                    else:
                        avg_v     = float(avg_volume)
                        volume_ok = volume >= self.volume_factor * avg_v

                    if trend_ok and zone_ok and rsi_ok and candle_ok and volume_ok:
                        entry_price = close
                        # Stop just beyond the 78.6% level; fall back to a hard
                        # percentage stop if that level is not below the entry.
                        if 0 < fib_stop < entry_price:
                            stop_loss = round(fib_stop, 2)
                        else:
                            stop_loss = round(entry_price * (1 - self.stop_loss_pct / 100.0), 2)
                        target      = round(max(target_px, entry_price), 2)
                        alloc_pct   = _capital_allocation_pct(entry_price)
                        shares      = max(floor(self.capital * alloc_pct / entry_price), 1)
                        peak_price  = entry_price
                        moved_to_breakeven = False
                        entry_swing_high = swing_high
                        entry_swing_low  = swing_low
                        entry_fib_level  = round(fib_entry, 2)
                        entry_rsi        = rsi_val
                        breakout_vol     = volume
                        avg_vol_entry    = avg_v
                        vol_ratio        = volume / avg_v if avg_v > 0 else 0.0
                        in_position  = True
                        entry_dt     = dt
                        trade_count += 1

            if not pd.isna(rsi):
                prev_rsi = float(rsi)

        # Force-close any open position at end of day's data.
        if in_position and entry_dt is not None:
            last  = df.iloc[-1]
            price = float(last["close"])
            dt    = last["datetime"]
            duration = int((dt - entry_dt).total_seconds() / 60)
            pnl      = round(shares * (price - entry_price), 2)
            trades.append(FibSecondsTradeResult(
                symbol=symbol_label,
                option_type=option_type,
                strike=strike,
                expiry_date=expiry_date,
                entry_time=entry_dt,
                exit_time=dt,
                entry_price=entry_price,
                exit_price=price,
                shares=shares,
                pnl=pnl,
                exit_reason="SQUARE_OFF",
                swing_high=entry_swing_high,
                swing_low=entry_swing_low,
                fib_entry_level=entry_fib_level,
                fib_ratio=self.entry_ratio,
                rsi_at_entry=round(entry_rsi, 2),
                breakout_volume=breakout_vol,
                avg_volume=round(avg_vol_entry, 2),
                volume_ratio=round(vol_ratio, 2),
                duration_minutes=duration,
            ))

        return trades

    # ── Debug / inspection helpers ────────────────────────────────────────────

    def _print_resampled_with_trades(
        self,
        candles: list[dict],
        trades: list[FibSecondsTradeResult],
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

        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", None,
        ):
            print(df.to_string(index=False))

        print(f"\n  Trades for {symbol_label}: {len(trades)}")
        for t in sorted(trades, key=lambda x: x.entry_time):
            pnl_sign = "+" if t.pnl >= 0 else ""
            print(
                f"    {t.entry_time.strftime('%d-%b %H:%M:%S')} → "
                f"{t.exit_time.strftime('%d-%b %H:%M:%S')}"
                f"  entry {t.entry_price:.2f}  exit {t.exit_price:.2f}"
                f"  qty {t.shares}  pnl {pnl_sign}{t.pnl:.2f}"
                f"  ({t.exit_reason})  rsi{t.rsi_at_entry:.1f}  volx{t.volume_ratio:.2f}"
            )

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[WeeklyFibExpiryResult]:
        expiry_results: list[WeeklyFibExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        effective_tf = (
            f"{self.resample_seconds}s" if self.resample_seconds > 1 else self.interval
        )
        print(f"\n{'='*70}")
        print(f"  NIFTY FIBONACCI RETRACEMENT STRATEGY — WEEKLY EXPIRY BACKTEST (seconds data)")
        print(f"  Period    : {self.start_date}  →  {self.end_date}")
        print(f"  Fetch TF  : {self.interval}  |  Effective TF: {effective_tf}")
        print(f"  Capital   : ₹{self.capital:,.0f}  |  MA: {self.ma_period}"
              f"  |  Swing: {self.swing_lookback}  |  Entry fib: {self.entry_ratio}"
              f"  |  Stop fib: {self.stop_ratio}  |  Ext: {self.extension_ratio}")
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
    ) -> WeeklyFibExpiryResult | None:
        try:
            nifty_open = self.nifty_service.get_nifty_open(monday)
        except Exception as exc:
            print(f"  [{expiry}] Could not get Nifty open for {monday}: {exc}")
            return None

        strike = NiftyOptionService.atm_strike(nifty_open)
        print(f"  Expiry {expiry}  |  Monday open {nifty_open:.2f}  |  ATM {strike}")

        week_result = WeeklyFibExpiryResult(
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
    ) -> WeeklyFibExpiryResult | None:
        print(f"  Expiry {expiry}  |  per-day ATM")

        week_result = WeeklyFibExpiryResult(
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
    def print_report(expiry_results: list[WeeklyFibExpiryResult]) -> None:
        all_trades: list[FibSecondsTradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[FibSecondsTradeResult]] = defaultdict(list)
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
                f"  {'Symbol':<22} {'Entry':>19} {'Exit':>19}"
                f" {'Entry₹':>8} {'Exit₹':>8} {'Qty':>5}"
                f" {'PnL':>10} {'Reason':<14}"
                f" {'Fib₹':>8} {'RSI':>6} {'VolX':>6}"
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
                    f" {t.fib_entry_level:>8.2f}"
                    f" {t.rsi_at_entry:>6.1f}"
                    f" {t.volume_ratio:>6.2f}"
                )

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
