"""
Open Range Breakout (ORB) strategy for Nifty weekly options on 1-second data.

The strategy is the options analogue of the intraday ORB system, run directly on
each option contract's own price series (resampled to N-second candles):

Opening range:
  For every trading day the opening range is built from the first `orb_minutes`
  of trading (from 9:15). ORB high/low and the average ORB-candle volume are
  recorded as the breakout reference for that day.

Entry (long option only):
  After the opening range is complete, enter LONG when a candle CLOSES above the
  ORB high *with good volume* — i.e. the breakout candle's volume is at least
  `volume_factor` times the average opening-range volume. Buying an option is an
  inherently long/directional bet, so only upside breakouts of the option's own
  price are traded (a CE breakout = bullish on the index, a PE breakout =
  bearish on the index). A close back below the ORB low is treated as a failed
  range and no short is taken (you cannot "sell" a long-only option backtest).

  Fair Value Gap confirmation (optional):
    When `fvg_confirmation_enabled` is True, a breakout is only taken if a bullish
    Fair Value Gap is present within the last `fvg_lookback` candles. A bullish
    FVG is a 3-candle imbalance where the most recent candle's low is strictly
    above the high of the candle two bars earlier (low[i] > high[i-2]), leaving an
    unfilled gap that confirms strong upward momentum behind the breakout.

Exit:
  - Target        : entry + risk * risk_reward_ratio, where risk = stop distance
  - Stop-loss      : `stop_loss_pct` percent below entry (hard stop)
  - Trailing stop  : optional; ratchets up with the peak price and exits when the
                     price falls `trailing_stop_pct` percent below that peak
  - Break-even stop: optional; once price moves `breakeven_trigger_pct` percent
                     above entry, the stop-loss is moved up to the entry price.
                     When `breakeven_partial_book_enabled` is True, 50% of the
                     position is also booked while the remaining 50% continues to
                     run; when False, the full position is held on with the stop
                     at break-even
  - Square-off     : forced exit at 15:20 IST
  - No new entries before the opening range completes or after 14:45
  - Max `max_trades_per_day` entries per day per contract
"""

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from math import floor

import pandas as pd

from models.orb_seconds_models import (
    ORBSecondsTradeResult,
    ORBSymbolMetrics,
    WeeklyORBExpiryResult,
)
from services.nifty_option_service import NiftyOptionService
from strategy.adx_option_strategy import resample_candles


_ENTRY_CUTOFF = time(14, 45)
_SQUARE_OFF   = time(15, 20)


def _capital_allocation_pct(price: float) -> float:
    """
    Return the fraction of capital to allocate based on option price.
    Mirrors the ADX seconds strategy allocation buckets.
    """
    if price <= 20:
        return 0.30
    elif price <= 60:
        return 1.00
    elif price <= 100:
        return 1.00
    else:
        return 1.00


def _compute_metrics(symbol: str, trades: list[ORBSecondsTradeResult]) -> ORBSymbolMetrics:
    if not trades:
        return ORBSymbolMetrics(
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

    return ORBSymbolMetrics(
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


class ORBOptionSecondsStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        orb_minutes: int = 15,
        volume_factor: float = 1.5,
        stop_loss_pct: float = 25.0,
        risk_reward_ratio: float = 2.0,
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
        fvg_confirmation_enabled: bool = False,
        fvg_lookback: int = 3,
    ):
        self.nifty_service    = nifty_service
        self.capital          = capital
        # Length of the opening range, in minutes from 9:15.
        self.orb_minutes      = orb_minutes
        # Minimum ratio of the breakout candle's volume to the average opening
        # range volume required to confirm a "good volume" breakout. 1.0 demands
        # at least average ORB volume; >1.0 demands a surge. Set to 0 to disable.
        self.volume_factor    = volume_factor
        # Hard stop-loss as a percentage below the entry price.
        self.stop_loss_pct    = stop_loss_pct
        # Target distance as a multiple of the stop distance (risk:reward).
        self.risk_reward_ratio = risk_reward_ratio
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
        # Break-even stop. Once the option price moves `breakeven_trigger_pct`
        # percent above entry, the stop-loss is moved up to the entry price so the
        # trade can no longer turn into a loss (locks in break-even).
        self.breakeven_enabled     = breakeven_enabled
        self.breakeven_trigger_pct = breakeven_trigger_pct
        # Partial profit booking at break-even. When enabled (the default), 50% of
        # the position is booked at the moment the stop is moved to break-even,
        # while the remaining 50% continues to run. When disabled, the stop is
        # still moved up to the entry price but the full position is held on.
        self.breakeven_partial_book_enabled = breakeven_partial_book_enabled
        # Fair Value Gap (FVG) entry confirmation. When enabled, a breakout entry
        # is only taken if a bullish FVG (a 3-candle imbalance where the most
        # recent candle's low is strictly above the high of the candle two bars
        # earlier) is present within the last `fvg_lookback` candles up to and
        # including the breakout candle.
        self.fvg_confirmation_enabled = fvg_confirmation_enabled
        self.fvg_lookback             = max(int(fvg_lookback), 3)

    @staticmethod
    def _has_bullish_fvg(highs: list[float], lows: list[float], lookback: int) -> bool:
        """
        Return True if a bullish Fair Value Gap is present within the last
        `lookback` candles of the supplied high/low series.

        A bullish FVG is a 3-candle imbalance: for some triple (i-2, i-1, i) the
        low of the most recent candle is strictly above the high of the candle two
        bars earlier — i.e. low[i] > high[i-2]. This leaves an unfilled price gap
        spanning candle i-1, signalling strong upward momentum.
        """
        n = len(highs)
        if n < 3:
            return False
        # Only inspect FVGs whose most-recent candle falls inside the lookback
        # window ending at the latest candle.
        start = max(2, n - lookback)
        for i in range(start, n):
            if lows[i] > highs[i - 2]:
                return True
        return False

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _run_symbol(
        self,
        candles: list[dict],
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[ORBSecondsTradeResult]:
        """
        Run the ORB strategy on a single option contract's candle data. The
        opening range is rebuilt for every trading day in the data. Returns a
        list of completed trades.
        """
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
        trades: list[ORBSecondsTradeResult] = []

        # Process each trading day independently — the opening range resets daily.
        for today, day_df in df.groupby(df["datetime"].dt.date):
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
    ) -> list[ORBSecondsTradeResult]:
        day_df = day_df.sort_values("datetime").reset_index(drop=True)
        if day_df.empty:
            return []

        first_dt: datetime = day_df.iloc[0]["datetime"]
        session_open = datetime.combine(first_dt.date(), time(9, 15))
        orb_end      = session_open + timedelta(minutes=self.orb_minutes)

        orb_mask = day_df["datetime"] < orb_end
        orb_df   = day_df[orb_mask]
        post_df  = day_df[~orb_mask]

        if orb_df.empty or post_df.empty:
            return []

        orb_high   = float(orb_df["high"].max())
        orb_low    = float(orb_df["low"].min())
        orb_avg_vol = float(orb_df["volume"].mean())

        trades: list[ORBSecondsTradeResult] = []
        trade_count = 0

        in_position = False
        entry_price = 0.0
        entry_dt    = None
        shares      = 0
        target      = 0.0
        stop_loss   = 0.0
        peak_price  = 0.0
        moved_to_breakeven = False
        breakout_vol = 0.0
        vol_ratio    = 0.0

        # Rolling high/low history of post-ORB candles, used to detect a bullish
        # Fair Value Gap when FVG entry confirmation is enabled.
        recent_highs: list[float] = []
        recent_lows:  list[float] = []

        for _, row in post_df.iterrows():
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            high  = float(row["high"])
            low   = float(row["low"])
            close = float(row["close"])
            volume = float(row["volume"])

            # Track candle highs/lows for Fair Value Gap detection.
            recent_highs.append(high)
            recent_lows.append(low)

            # ── Exit logic ────────────────────────────────────────────────
            if in_position:
                exit_reason = None
                exit_price  = close

                if high > peak_price:
                    peak_price = high

                # Break-even: once price moves `breakeven_trigger_pct` above entry,
                # ratchet the stop-loss up to the entry price (lock in break-even)
                # AND book 50% of the position, letting the rest ride on.
                if (
                    self.breakeven_enabled
                    and not moved_to_breakeven
                    and high >= entry_price * (1 + self.breakeven_trigger_pct / 100.0)
                ):
                    stop_loss = entry_price
                    moved_to_breakeven = True

                    # Book half the position at the break-even trigger level and
                    # continue holding the remaining half (only when partial
                    # booking is enabled).
                    book_price  = round(
                        entry_price * (1 + self.breakeven_trigger_pct / 100.0), 2
                    )
                    half_shares = shares // 2
                    if self.breakeven_partial_book_enabled and half_shares >= 1:
                        duration = int((dt - entry_dt).total_seconds() / 60)
                        pnl      = round(half_shares * (book_price - entry_price), 2)
                        trades.append(ORBSecondsTradeResult(
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
                            orb_high=orb_high,
                            orb_low=orb_low,
                            breakout_volume=breakout_vol,
                            orb_avg_volume=round(orb_avg_vol, 2),
                            volume_ratio=round(vol_ratio, 2),
                            duration_minutes=duration,
                        ))
                        # Continue the position with the remaining shares only.
                        shares -= half_shares

                # Square-off at 15:20
                if t >= _SQUARE_OFF:
                    exit_reason = "SQUARE_OFF"
                    exit_price  = close
                # Hard stop-loss (checked before target). After the stop has been
                # moved to entry, a hit is a break-even exit rather than a loss.
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
                # Hard target
                elif high >= target:
                    exit_reason = "TARGET"
                    exit_price  = target

                if exit_reason:
                    duration = int((dt - entry_dt).total_seconds() / 60)
                    pnl      = round(shares * (exit_price - entry_price), 2)
                    trades.append(ORBSecondsTradeResult(
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
                        orb_high=orb_high,
                        orb_low=orb_low,
                        breakout_volume=breakout_vol,
                        orb_avg_volume=round(orb_avg_vol, 2),
                        volume_ratio=round(vol_ratio, 2),
                        duration_minutes=duration,
                    ))
                    in_position = False
                    entry_dt    = None
                    continue

            # ── Entry logic ───────────────────────────────────────────────
            if (
                not in_position
                and t <= _ENTRY_CUTOFF
                and trade_count < self.max_trades_per_day
            ):
                # Good-volume filter.
                if self.volume_factor <= 0 or orb_avg_vol <= 0:
                    volume_ok = self.volume_factor <= 0
                else:
                    volume_ok = volume >= self.volume_factor * orb_avg_vol

                # Upside breakout: candle closes above the opening-range high.
                breakout = close > orb_high

                # Fair Value Gap confirmation: require a recent bullish FVG.
                if self.fvg_confirmation_enabled:
                    fvg_ok = self._has_bullish_fvg(
                        recent_highs, recent_lows, self.fvg_lookback
                    )
                else:
                    fvg_ok = True

                if breakout and volume_ok and fvg_ok and close > 0:
                    entry_price = close
                    stop_loss   = round(entry_price * (1 - self.stop_loss_pct / 100.0), 2)
                    risk        = entry_price - stop_loss
                    target      = round(entry_price + risk * self.risk_reward_ratio, 2)
                    alloc_pct   = _capital_allocation_pct(entry_price)
                    shares      = max(floor(self.capital * alloc_pct / entry_price), 1)
                    peak_price  = entry_price
                    moved_to_breakeven = False
                    breakout_vol = volume
                    vol_ratio    = volume / orb_avg_vol if orb_avg_vol > 0 else 0.0
                    in_position  = True
                    entry_dt     = dt
                    trade_count += 1

        # Force-close any open position at end of day's data.
        if in_position and entry_dt is not None:
            last  = post_df.iloc[-1]
            price = float(last["close"])
            dt    = last["datetime"]
            duration = int((dt - entry_dt).total_seconds() / 60)
            pnl      = round(shares * (price - entry_price), 2)
            trades.append(ORBSecondsTradeResult(
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
                orb_high=orb_high,
                orb_low=orb_low,
                breakout_volume=breakout_vol,
                orb_avg_volume=round(orb_avg_vol, 2),
                volume_ratio=round(vol_ratio, 2),
                duration_minutes=duration,
            ))

        return trades

    # ── Debug / inspection helpers ────────────────────────────────────────────

    def _print_resampled_with_trades(
        self,
        candles: list[dict],
        trades: list[ORBSecondsTradeResult],
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
                f"  ({t.exit_reason})  volx{t.volume_ratio:.2f}"
            )

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[WeeklyORBExpiryResult]:
        expiry_results: list[WeeklyORBExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        effective_tf = (
            f"{self.resample_seconds}s" if self.resample_seconds > 1 else self.interval
        )
        print(f"\n{'='*70}")
        print(f"  NIFTY ORB STRATEGY — WEEKLY EXPIRY BACKTEST (seconds data)")
        print(f"  Period    : {self.start_date}  →  {self.end_date}")
        print(f"  Fetch TF  : {self.interval}  |  Effective TF: {effective_tf}")
        print(f"  Capital   : ₹{self.capital:,.0f}  |  ORB: {self.orb_minutes}min"
              f"  |  Vol factor: {self.volume_factor}  |  SL: {self.stop_loss_pct}%"
              f"  |  RR: 1:{self.risk_reward_ratio}")
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
    ) -> WeeklyORBExpiryResult | None:
        try:
            nifty_open = self.nifty_service.get_nifty_open(monday)
        except Exception as exc:
            print(f"  [{expiry}] Could not get Nifty open for {monday}: {exc}")
            return None

        strike = NiftyOptionService.atm_strike(nifty_open)
        print(f"  Expiry {expiry}  |  Monday open {nifty_open:.2f}  |  ATM {strike}")

        week_result = WeeklyORBExpiryResult(
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
    ) -> WeeklyORBExpiryResult | None:
        print(f"  Expiry {expiry}  |  per-day ATM")

        week_result = WeeklyORBExpiryResult(
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
    def print_report(expiry_results: list[WeeklyORBExpiryResult]) -> None:
        all_trades: list[ORBSecondsTradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[ORBSecondsTradeResult]] = defaultdict(list)
        for t in all_trades:
            by_symbol[t.symbol].append(t)

        sep = "─" * 90
        print(f"\n{'='*90}")
        print("  RESULTS BY EXPIRY")
        print(f"{'='*90}")

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
                f" {'ORBHi':>8} {'VolX':>6}"
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
                    f" {t.orb_high:>8.2f}"
                    f" {t.volume_ratio:>6.2f}"
                )

        print(f"\n{'='*90}")
        print("  SYMBOL-WISE METRICS")
        print(f"{'='*90}")

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
        print(f"  {'─'*88}")

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
        print(f"  {'─'*88}")
        print(
            f"  {'OVERALL':<{col_w['sym']}} {overall_trades:>{col_w['trades']}}"
            f" {overall_wins:>{col_w['wins']}} {overall_losses:>{col_w['loss']}}"
            f" {wr_overall:>{col_w['wr']}.1f} {pnl_s:>{col_w['pnl']}}"
        )
        print(f"{'='*90}\n")
