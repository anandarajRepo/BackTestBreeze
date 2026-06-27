"""
Initial Balance (IB) strategy for Nifty weekly options on 1-second data.

The Initial Balance is the price range established during the very first part of
the trading session — by default the first hour (9:15 → 10:15). Because the
opening hour carries the heaviest volume, it sets the tone for the day. Once the
IB window closes the high and low are locked and used as the key intraday
boundaries. Historically the price only breaks ONE side of the IB in ~70-85% of
sessions, which is the statistical edge behind both approaches below.

This is the options analogue of the IB system, run directly on each option
contract's own price series (resampled to N-second candles). Buying an option is
an inherently long/directional bet, so the strategy is always LONG the premium —
a CE trade = bullish on the index, a PE trade = bearish on the index. The two
classic IB approaches are mapped onto the option's OWN price as follows:

Initial Balance window:
  For every trading day the IB is built from the first `ib_minutes` of trading
  (from 9:15). The IB high, IB low and the average IB-candle volume are recorded
  as that day's reference.

1. BREAKOUT approach (momentum continuation):
     Enter LONG when a candle CLOSES firmly above the IB high — "firmly" meaning
     it clears the IB high by at least `breakout_buffer_pct` percent — with good
     volume (the breakout candle's volume is at least `volume_factor` times the
     average IB volume). This is the institutional momentum continuation: once the
     premium breaks its opening-hour high, the move is assumed to continue.

     Retest confirmation (optional, `retest_enabled`):
       Rather than entering on the breakout candle itself, wait for a slight
       pullback that retests the IB high (price dips back to within
       `retest_tolerance_pct` of the IB high) and then resumes upward (a later
       candle closes back above the IB high). This mirrors the common practice of
       waiting for a pullback to the broken boundary before entering the trend.

2. REVERSAL approach (failed breakdown / bear trap):
     Enter LONG when the premium first dips below the IB low (a "breakdown") but
     fails to hold there and CLOSES back above the IB low, rotating back up into
     the range. On the option's own price this is a failed-breakdown trap that
     anticipates a rotation back up toward the IB mid / IB high.

Use `strategy_mode` to select "breakout", "reversal" or "both".

Exit (common to both approaches):
  - Target        : entry + risk * risk_reward_ratio, where risk = stop distance
  - Stop-loss      : `stop_loss_pct` percent below entry (hard stop)
  - Trailing stop  : optional; ratchets up with the peak price and exits when the
                     price falls `trailing_stop_pct` percent below that peak
  - Break-even stop: optional; once price moves `breakeven_trigger_pct` percent
                     above entry, the stop-loss is moved up to the entry price.
                     When `breakeven_partial_book_enabled` is True, 50% of the
                     position is also booked while the remaining 50% runs on.
  - Square-off     : forced exit at 15:20 IST
  - No new entries before the IB window completes or after 14:45
  - Max `max_trades_per_day` entries per day per contract
"""

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from math import floor

import pandas as pd

from models.ib_models import (
    IBSecondsTradeResult,
    IBSymbolMetrics,
    WeeklyIBExpiryResult,
)
from services.nifty_option_service import NiftyOptionService
from strategy.adx_option_strategy import resample_candles


_ENTRY_CUTOFF = time(14, 45)
_SQUARE_OFF   = time(15, 20)


def _capital_allocation_pct(price: float) -> float:
    """
    Return the fraction of capital to allocate based on option price.
    Mirrors the allocation buckets used by the other seconds option strategies.
    """
    if price <= 20:
        return 0.30
    elif price <= 60:
        return 1.00
    elif price <= 100:
        return 1.00
    else:
        return 1.00


def _compute_metrics(symbol: str, trades: list[IBSecondsTradeResult]) -> IBSymbolMetrics:
    if not trades:
        return IBSymbolMetrics(
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

    return IBSymbolMetrics(
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


class IBOptionSecondsStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        ib_minutes: int = 60,
        strategy_mode: str = "both",
        volume_factor: float = 1.5,
        breakout_buffer_pct: float = 0.0,
        retest_enabled: bool = False,
        retest_tolerance_pct: float = 2.0,
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
    ):
        self.nifty_service    = nifty_service
        self.capital          = capital
        # Length of the Initial Balance window, in minutes from 9:15. The default
        # of 60 captures the first trading hour (9:15 → 10:15).
        self.ib_minutes       = ib_minutes
        # Which IB approach(es) to trade: "breakout", "reversal" or "both".
        mode = (strategy_mode or "both").strip().lower()
        if mode not in ("breakout", "reversal", "both"):
            raise ValueError(
                f"strategy_mode must be 'breakout', 'reversal' or 'both', got {strategy_mode!r}"
            )
        self.strategy_mode    = mode
        # Minimum ratio of the breakout candle's volume to the average IB volume
        # required to confirm a "good volume" breakout. 1.0 demands at least
        # average IB volume; >1.0 demands a surge. Set to 0 to disable.
        self.volume_factor    = volume_factor
        # A breakout candle must clear the IB high by at least this percent to be
        # considered a *firm* break (filters marginal pokes through the boundary).
        # 0 = any close above the IB high qualifies.
        self.breakout_buffer_pct = breakout_buffer_pct
        # Retest confirmation for breakouts. When enabled, the strategy waits for a
        # pullback that retests the IB high before entering the breakout, rather
        # than entering on the breakout candle itself.
        self.retest_enabled      = retest_enabled
        # How close (in percent above the IB high) a pullback must come to the IB
        # high to count as a valid retest of the broken boundary.
        self.retest_tolerance_pct = retest_tolerance_pct
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

    @property
    def _trade_breakout(self) -> bool:
        return self.strategy_mode in ("breakout", "both")

    @property
    def _trade_reversal(self) -> bool:
        return self.strategy_mode in ("reversal", "both")

    # ── Core per-symbol backtest ──────────────────────────────────────────────

    def _run_symbol(
        self,
        candles: list[dict],
        option_type: str,
        strike: int,
        expiry_date: date,
    ) -> list[IBSecondsTradeResult]:
        """
        Run the IB strategy on a single option contract's candle data. The
        Initial Balance window is rebuilt for every trading day in the data.
        Returns a list of completed trades.
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
        trades: list[IBSecondsTradeResult] = []

        # Process each trading day independently — the IB window resets daily.
        for _, day_df in df.groupby(df["datetime"].dt.date):
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
    ) -> list[IBSecondsTradeResult]:
        day_df = day_df.sort_values("datetime").reset_index(drop=True)
        if day_df.empty:
            return []

        first_dt: datetime = day_df.iloc[0]["datetime"]
        session_open = datetime.combine(first_dt.date(), time(9, 15))
        ib_end       = session_open + timedelta(minutes=self.ib_minutes)

        ib_mask = day_df["datetime"] < ib_end
        ib_df   = day_df[ib_mask]
        post_df = day_df[~ib_mask]

        if ib_df.empty or post_df.empty:
            return []

        ib_high    = float(ib_df["high"].max())
        ib_low     = float(ib_df["low"].min())
        ib_avg_vol = float(ib_df["volume"].mean())

        # Firm-breakout reference: the close must clear the IB high by the buffer.
        firm_break_level = ib_high * (1 + self.breakout_buffer_pct / 100.0)
        # Retest reference: a pullback low must come back to within the tolerance
        # band above the IB high to count as a retest of the broken boundary.
        retest_level     = ib_high * (1 + self.retest_tolerance_pct / 100.0)

        trades: list[IBSecondsTradeResult] = []
        trade_count = 0

        in_position = False
        entry_price = 0.0
        entry_dt    = None
        entry_mode  = ""
        shares      = 0
        target      = 0.0
        stop_loss   = 0.0
        peak_price  = 0.0
        moved_to_breakeven = False
        breakout_vol = 0.0
        vol_ratio    = 0.0

        # Reversal state: set once the premium breaks down below the IB low, so a
        # subsequent close back above the IB low can be recognised as a failed
        # breakdown (bear trap) rotation entry.
        broke_below_ib_low = False
        # Breakout retest state: armed once a firm breakout above the IB high is
        # seen, then completed when price pulls back to retest the IB high and
        # resumes upward.
        breakout_armed = False

        for _, row in post_df.iterrows():
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            high  = float(row["high"])
            low   = float(row["low"])
            close = float(row["close"])
            volume = float(row["volume"])

            # ── Exit logic ────────────────────────────────────────────────
            if in_position:
                exit_reason = None
                exit_price  = close

                if high > peak_price:
                    peak_price = high

                # Break-even: once price moves `breakeven_trigger_pct` above entry,
                # ratchet the stop-loss up to the entry price (lock in break-even)
                # and book 50% of the position, letting the rest ride on.
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
                        trades.append(IBSecondsTradeResult(
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
                            entry_mode=entry_mode,
                            ib_high=ib_high,
                            ib_low=ib_low,
                            breakout_volume=breakout_vol,
                            ib_avg_volume=round(ib_avg_vol, 2),
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
                    trades.append(IBSecondsTradeResult(
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
                        entry_mode=entry_mode,
                        ib_high=ib_high,
                        ib_low=ib_low,
                        breakout_volume=breakout_vol,
                        ib_avg_volume=round(ib_avg_vol, 2),
                        volume_ratio=round(vol_ratio, 2),
                        duration_minutes=duration,
                    ))
                    in_position = False
                    entry_dt    = None
                    continue

            # ── Breakdown tracking (for the reversal/trap entry) ──────────
            # Mark that the premium has broken below the IB low, arming a possible
            # failed-breakdown rotation entry on a later recovery above the IB low.
            if low < ib_low:
                broke_below_ib_low = True

            # ── Entry logic ───────────────────────────────────────────────
            if (
                not in_position
                and t <= _ENTRY_CUTOFF
                and trade_count < self.max_trades_per_day
            ):
                # Good-volume filter (shared by both approaches).
                if self.volume_factor <= 0 or ib_avg_vol <= 0:
                    volume_ok = self.volume_factor <= 0
                else:
                    volume_ok = volume >= self.volume_factor * ib_avg_vol

                mode_signal = None  # "BREAKOUT" | "REVERSAL"

                # 1. BREAKOUT — close firmly above the IB high (momentum).
                if self._trade_breakout:
                    firm_breakout = close >= firm_break_level
                    if not self.retest_enabled:
                        if firm_breakout:
                            mode_signal = "BREAKOUT"
                    else:
                        # Two-stage retest: arm on the firm breakout, then enter
                        # once price pulls back to retest the IB high and a later
                        # candle closes back above it.
                        if firm_breakout:
                            if not breakout_armed:
                                breakout_armed = True
                            elif low <= retest_level:
                                # Pullback retested the boundary and the candle
                                # still closed above the IB high → take the entry.
                                mode_signal = "BREAKOUT"
                                breakout_armed = False

                # 2. REVERSAL — failed breakdown: dipped below IB low then closes
                #    back above it, rotating up into the range.
                if (
                    mode_signal is None
                    and self._trade_reversal
                    and broke_below_ib_low
                    and close > ib_low
                ):
                    mode_signal = "REVERSAL"

                if mode_signal is not None and volume_ok and close > 0:
                    entry_price = close
                    stop_loss   = round(entry_price * (1 - self.stop_loss_pct / 100.0), 2)
                    risk        = entry_price - stop_loss
                    target      = round(entry_price + risk * self.risk_reward_ratio, 2)
                    alloc_pct   = _capital_allocation_pct(entry_price)
                    shares      = max(floor(self.capital * alloc_pct / entry_price), 1)
                    peak_price  = entry_price
                    moved_to_breakeven = False
                    breakout_vol = volume
                    vol_ratio    = volume / ib_avg_vol if ib_avg_vol > 0 else 0.0
                    in_position  = True
                    entry_dt     = dt
                    entry_mode   = mode_signal
                    trade_count += 1
                    # Reset the reversal arm so the next breakdown must form afresh.
                    if mode_signal == "REVERSAL":
                        broke_below_ib_low = False

        # Force-close any open position at end of day's data.
        if in_position and entry_dt is not None:
            last  = post_df.iloc[-1]
            price = float(last["close"])
            dt    = last["datetime"]
            duration = int((dt - entry_dt).total_seconds() / 60)
            pnl      = round(shares * (price - entry_price), 2)
            trades.append(IBSecondsTradeResult(
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
                entry_mode=entry_mode,
                ib_high=ib_high,
                ib_low=ib_low,
                breakout_volume=breakout_vol,
                ib_avg_volume=round(ib_avg_vol, 2),
                volume_ratio=round(vol_ratio, 2),
                duration_minutes=duration,
            ))

        return trades

    # ── Debug / inspection helpers ────────────────────────────────────────────

    def _print_resampled_with_trades(
        self,
        candles: list[dict],
        trades: list[IBSecondsTradeResult],
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
                f"  [{t.entry_mode}]"
                f"  entry {t.entry_price:.2f}  exit {t.exit_price:.2f}"
                f"  qty {t.shares}  pnl {pnl_sign}{t.pnl:.2f}"
                f"  ({t.exit_reason})  volx{t.volume_ratio:.2f}"
            )

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[WeeklyIBExpiryResult]:
        expiry_results: list[WeeklyIBExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        effective_tf = (
            f"{self.resample_seconds}s" if self.resample_seconds > 1 else self.interval
        )
        print(f"\n{'='*70}")
        print(f"  NIFTY INITIAL BALANCE (IB) STRATEGY — WEEKLY EXPIRY BACKTEST (seconds data)")
        print(f"  Period    : {self.start_date}  →  {self.end_date}")
        print(f"  Fetch TF  : {self.interval}  |  Effective TF: {effective_tf}")
        print(f"  Capital   : ₹{self.capital:,.0f}  |  IB: {self.ib_minutes}min"
              f"  |  Mode: {self.strategy_mode}  |  Vol factor: {self.volume_factor}")
        print(f"  SL: {self.stop_loss_pct}%  |  RR: 1:{self.risk_reward_ratio}"
              f"  |  Breakout buffer: {self.breakout_buffer_pct}%"
              f"  |  Retest: {'on' if self.retest_enabled else 'off'}")
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
    ) -> WeeklyIBExpiryResult | None:
        try:
            nifty_open = self.nifty_service.get_nifty_open(monday)
        except Exception as exc:
            print(f"  [{expiry}] Could not get Nifty open for {monday}: {exc}")
            return None

        strike = NiftyOptionService.atm_strike(nifty_open)
        print(f"  Expiry {expiry}  |  Monday open {nifty_open:.2f}  |  ATM {strike}")

        week_result = WeeklyIBExpiryResult(
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
    ) -> WeeklyIBExpiryResult | None:
        print(f"  Expiry {expiry}  |  per-day ATM")

        week_result = WeeklyIBExpiryResult(
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
    def print_report(expiry_results: list[WeeklyIBExpiryResult]) -> None:
        all_trades: list[IBSecondsTradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.all_trades)

        by_symbol: dict[str, list[IBSecondsTradeResult]] = defaultdict(list)
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
                f" {'Mode':<9} {'Entry₹':>8} {'Exit₹':>8} {'Qty':>5}"
                f" {'PnL':>10} {'Reason':<14}"
                f" {'IBHi':>8} {'VolX':>6}"
            )
            print(header)
            print(f"  {sep}")

            for t in sorted(er.all_trades, key=lambda x: x.entry_time):
                pnl_sign = "+" if t.pnl >= 0 else ""
                print(
                    f"  {t.symbol:<22}"
                    f" {t.entry_time.strftime('%d-%b %H:%M'):>16}"
                    f" {t.exit_time.strftime('%d-%b %H:%M'):>16}"
                    f" {t.entry_mode:<9}"
                    f" {t.entry_price:>8.2f}"
                    f" {t.exit_price:>8.2f}"
                    f" {t.shares:>5}"
                    f" {pnl_sign+f'{t.pnl:.2f}':>10}"
                    f" {t.exit_reason:<14}"
                    f" {t.ib_high:>8.2f}"
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

        # Breakdown by IB approach (BREAKOUT vs REVERSAL).
        by_mode: dict[str, list[IBSecondsTradeResult]] = defaultdict(list)
        for t in all_trades:
            by_mode[t.entry_mode].append(t)

        if by_mode:
            print(f"\n  {'─'*94}")
            print(f"  BY IB APPROACH")
            print(f"  {'─'*94}")
            for mode in ("BREAKOUT", "REVERSAL"):
                mt = by_mode.get(mode, [])
                if not mt:
                    continue
                m = _compute_metrics(mode, mt)
                pnl_s = f"{'+' if m.total_pnl >= 0 else ''}{m.total_pnl:.2f}"
                print(
                    f"  {mode:<{col_w['sym']}} {m.total_trades:>{col_w['trades']}}"
                    f" {m.wins:>{col_w['wins']}} {m.losses:>{col_w['loss']}}"
                    f" {m.win_rate:>{col_w['wr']}.1f} {pnl_s:>{col_w['pnl']}}"
                )

        print(f"{'='*96}\n")
