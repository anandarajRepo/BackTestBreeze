"""
Opening Range Breakout — Vertical Spread strategy for Nifty weekly options
(seconds data).

The strategy defines the opening range using a user-defined initial time window
after market open (built from the Nifty *spot* index). It then waits for the
spot to break out of that range and deploys a directional, defined-risk vertical
debit spread:

Opening range:
  For each trading day the opening range high/low is built from the first
  ``or_minutes`` of trading (from 9:15) on the Nifty spot index.

Entry trigger (one position per day, no re-entry after exit):
  • Spot crosses ABOVE the opening-range high  → bullish breakout
        Buy 1 ATM (or near-ATM) Call + Sell 1 higher-strike Call
        → Bull Call Spread (net debit)
  • Spot crosses BELOW the opening-range low   → bearish breakdown
        Buy 1 ATM (or near-ATM) Put  + Sell 1 lower-strike  Put
        → Bear Put Spread (net debit)
  The short strike is ``spread_distance`` points away from the ATM/long strike.

Exit trigger (combined two-leg position P&L monitored continuously):
  • Combined P&L ≥ +``profit_target`` (default ₹3,000)            → PROFIT_TARGET
  • Combined P&L ≤ −``stop_loss``      (default ₹3,000)           → STOP_LOSS
  • Time = 15:25 IST (square off whatever is still open)          → SQUARE_OFF

Re-entry logic:
  No re-entry after an exit within the same session. The strategy resets the
  next trading day.

Combined P&L (per unit):
    long-leg  move : (buy_now  − buy_entry)
    short-leg move : (sell_entry − sell_now)
    per_unit_pnl   = (buy_now − buy_entry) + (sell_entry − sell_now)
    total_pnl      = per_unit_pnl × lot_size × lots
"""

from collections import defaultdict
from datetime import date, datetime, time, timedelta

import pandas as pd

from models.orb_spread_models import (
    ORBSpreadDayResult,
    ORBSpreadMetrics,
    ORBSpreadTradeResult,
)
from services.nifty_option_service import NiftyOptionService
from strategy.orb_option_seconds_strategy import resample_candles


_SQUARE_OFF = time(15, 25)


def _compute_metrics(label: str, trades: list[ORBSpreadTradeResult]) -> ORBSpreadMetrics:
    if not trades:
        return ORBSpreadMetrics(
            label=label, total_trades=0, wins=0, losses=0,
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

    return ORBSpreadMetrics(
        label=label,
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


def _price_series(candles: list[dict] | None) -> pd.Series:
    """Build a datetime-indexed close-price Series from a list of OHLC dicts."""
    if not candles:
        return pd.Series(dtype=float)
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["close"] = df["close"].astype(float)
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")
    return df.set_index("datetime")["close"]


class ORBSpreadOptionSecondsStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        or_minutes: int = 15,
        spread_distance: int = 100,
        profit_target: float = 3000.0,
        stop_loss: float = 3000.0,
        lot_size: int = 75,
        lots: int = 1,
        start_date: str = "",
        end_date: str = "",
        interval: str = "1second",
        resample_seconds: int = 5,
        print_resampled: bool = False,
        cache_only: bool = False,
        market_holidays: set[date] | None = None,
    ):
        self.nifty_service   = nifty_service
        # Length of the opening range, in minutes from 9:15.
        self.or_minutes      = or_minutes
        # Distance (in index points) between the long (ATM) strike and the short
        # strike of the spread. e.g. 100 = two 50-point strikes away.
        self.spread_distance = spread_distance
        # Combined-position profit target (₹). The position is closed once the
        # combined P&L reaches +profit_target.
        self.profit_target   = profit_target
        # Combined-position stop-loss (₹, magnitude). The position is closed once
        # the combined P&L falls to −stop_loss.
        self.stop_loss       = stop_loss
        # Nifty lot size and number of lots traded per leg.
        self.lot_size        = lot_size
        self.lots            = lots
        self.start_date      = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date        = datetime.strptime(end_date,   "%d-%b-%Y").date()
        self.interval        = interval
        self.resample_seconds = resample_seconds
        self.print_resampled = print_resampled
        self.cache_only      = cache_only
        self.market_holidays = set(market_holidays) if market_holidays else set()

    @property
    def quantity(self) -> int:
        return self.lot_size * self.lots

    # ── Core per-day backtest ──────────────────────────────────────────────────

    def _run_day(
        self,
        day: date,
        expiry: date,
        spot_candles: list[dict],
        leg_prices: dict[str, pd.Series],
        atm_strike: int,
    ) -> ORBSpreadTradeResult | None:
        """
        Run the opening-range-breakout spread on a single trading day.

        ``leg_prices`` maps a leg key to its close-price Series:
            "CE_BUY", "CE_SELL", "PE_BUY", "PE_SELL".
        Returns the single completed trade for the day, or None.
        """
        if not spot_candles:
            return None

        spot = pd.DataFrame(spot_candles)
        spot["datetime"] = pd.to_datetime(spot["datetime"])
        for col in ("open", "high", "low", "close"):
            spot[col] = spot[col].astype(float)
        spot = spot.sort_values("datetime").reset_index(drop=True)

        session_open = datetime.combine(day, time(9, 15))
        or_end       = session_open + timedelta(minutes=self.or_minutes)

        or_mask = spot["datetime"] < or_end
        or_df   = spot[or_mask]
        post_df = spot[~or_mask]
        if or_df.empty or post_df.empty:
            return None

        or_high = float(or_df["high"].max())
        or_low  = float(or_df["low"].min())

        # Forward-fill each leg's price onto the post-OR spot timeline so the two
        # legs are always priced together at every monitoring timestamp.
        post_index = post_df["datetime"]
        legs: dict[str, pd.Series] = {}
        for key, series in leg_prices.items():
            if series.empty:
                return None
            legs[key] = series.reindex(post_index, method="ffill")

        bull_strike_buy  = atm_strike
        bull_strike_sell = atm_strike + self.spread_distance
        bear_strike_buy  = atm_strike
        bear_strike_sell = atm_strike - self.spread_distance

        in_position = False
        direction   = ""
        buy_leg = sell_leg = ""
        buy_strike = sell_strike = 0
        buy_entry = sell_entry = 0.0
        net_debit = 0.0
        entry_dt: datetime | None = None
        breakout_price = 0.0

        for i in range(len(post_df)):
            row = post_df.iloc[i]
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            close        = float(row["close"])

            # ── Exit logic (combined P&L monitored continuously) ──────────
            if in_position:
                buy_now  = float(legs[buy_leg].iloc[i])
                sell_now = float(legs[sell_leg].iloc[i])
                if pd.isna(buy_now) or pd.isna(sell_now):
                    continue

                per_unit = (buy_now - buy_entry) + (sell_entry - sell_now)
                pnl = per_unit * self.quantity

                exit_reason = None
                if t >= _SQUARE_OFF:
                    exit_reason = "SQUARE_OFF"
                elif pnl >= self.profit_target:
                    exit_reason = "PROFIT_TARGET"
                elif pnl <= -self.stop_loss:
                    exit_reason = "STOP_LOSS"

                if exit_reason:
                    duration = int((dt - entry_dt).total_seconds() / 60)
                    return ORBSpreadTradeResult(
                        direction=direction,
                        spread_type=(
                            "BULL_CALL_SPREAD" if direction == "BULL"
                            else "BEAR_PUT_SPREAD"
                        ),
                        expiry_date=expiry,
                        buy_strike=buy_strike,
                        sell_strike=sell_strike,
                        entry_time=entry_dt,
                        exit_time=dt,
                        buy_entry=round(buy_entry, 2),
                        sell_entry=round(sell_entry, 2),
                        buy_exit=round(buy_now, 2),
                        sell_exit=round(sell_now, 2),
                        net_debit=round(net_debit, 2),
                        exit_value=round(buy_now - sell_now, 2),
                        quantity=self.quantity,
                        pnl=round(pnl, 2),
                        exit_reason=exit_reason,
                        or_high=or_high,
                        or_low=or_low,
                        breakout_price=round(breakout_price, 2),
                        duration_minutes=duration,
                    )
                continue

            # ── Entry logic (no re-entry once a position has been taken) ──
            if t >= _SQUARE_OFF:
                break

            signal = None
            if close > or_high:
                signal = "BULL"
            elif close < or_low:
                signal = "BEAR"

            if signal is None:
                continue

            if signal == "BULL":
                buy_leg, sell_leg = "CE_BUY", "CE_SELL"
                buy_strike, sell_strike = bull_strike_buy, bull_strike_sell
            else:
                buy_leg, sell_leg = "PE_BUY", "PE_SELL"
                buy_strike, sell_strike = bear_strike_buy, bear_strike_sell

            b = float(legs[buy_leg].iloc[i])
            s = float(legs[sell_leg].iloc[i])
            if pd.isna(b) or pd.isna(s) or b <= 0:
                continue

            direction      = signal
            buy_entry      = b
            sell_entry     = s
            net_debit      = b - s
            entry_dt       = dt
            breakout_price = close
            in_position    = True

        # Force square-off at end of day's data if still open.
        if in_position and entry_dt is not None:
            last_i  = len(post_df) - 1
            row     = post_df.iloc[last_i]
            dt      = row["datetime"]
            buy_now  = float(legs[buy_leg].iloc[last_i])
            sell_now = float(legs[sell_leg].iloc[last_i])
            per_unit = (buy_now - buy_entry) + (sell_entry - sell_now)
            pnl      = per_unit * self.quantity
            duration = int((dt - entry_dt).total_seconds() / 60)
            return ORBSpreadTradeResult(
                direction=direction,
                spread_type=(
                    "BULL_CALL_SPREAD" if direction == "BULL" else "BEAR_PUT_SPREAD"
                ),
                expiry_date=expiry,
                buy_strike=buy_strike,
                sell_strike=sell_strike,
                entry_time=entry_dt,
                exit_time=dt,
                buy_entry=round(buy_entry, 2),
                sell_entry=round(sell_entry, 2),
                buy_exit=round(buy_now, 2),
                sell_exit=round(sell_now, 2),
                net_debit=round(net_debit, 2),
                exit_value=round(buy_now - sell_now, 2),
                quantity=self.quantity,
                pnl=round(pnl, 2),
                exit_reason="SQUARE_OFF",
                or_high=or_high,
                or_low=or_low,
                breakout_price=round(breakout_price, 2),
                duration_minutes=duration,
            )

        return None

    # ── Data orchestration ─────────────────────────────────────────────────────

    def _fetch_leg(
        self, strike: int, option_type: str, expiry: date,
        from_dt: datetime, to_dt: datetime,
    ) -> list[dict] | None:
        candles = self.nifty_service.get_option_candles(
            strike=strike, expiry_date=expiry, option_type=option_type,
            start=from_dt, end=to_dt, interval=self.interval, cache_only=self.cache_only,
        )
        if candles and self.resample_seconds > 1:
            candles = resample_candles(candles, self.resample_seconds)
        return candles

    def run_backtest(self) -> list[ORBSpreadDayResult]:
        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        effective_tf = (
            f"{self.resample_seconds}s" if self.resample_seconds > 1 else self.interval
        )
        print(f"\n{'='*70}")
        print(f"  NIFTY OPENING-RANGE-BREAKOUT SPREAD — WEEKLY EXPIRY BACKTEST")
        print(f"  Period    : {self.start_date}  →  {self.end_date}")
        print(f"  Fetch TF  : {self.interval}  |  Effective TF: {effective_tf}")
        print(f"  OR window : {self.or_minutes}min  |  Spread dist: {self.spread_distance}"
              f"  |  Target: ₹{self.profit_target:,.0f}  |  SL: ₹{self.stop_loss:,.0f}")
        print(f"  Lot size  : {self.lot_size} × {self.lots} lot(s) = {self.quantity} units")
        print(f"  Expiries found: {len(wednesdays)}")
        print(f"{'='*70}\n")

        day_results: list[ORBSpreadDayResult] = []

        for tuesday in wednesdays:
            win_start, _ = NiftyOptionService.week_window(tuesday)
            expiry = NiftyOptionService.adjust_expiry_for_holidays(
                tuesday, self.market_holidays
            )
            if expiry != tuesday:
                print(f"  [holiday] Expiry {tuesday} is a holiday — rolled back to {expiry}")

            days = self.nifty_service.trading_days(
                win_start, expiry, self.market_holidays
            )

            for day in days:
                result = self._run_single_day(day, expiry)
                if result is not None:
                    day_results.append(result)

        return day_results

    def _run_single_day(self, day: date, expiry: date) -> ORBSpreadDayResult | None:
        try:
            nifty_open = self.nifty_service.get_nifty_open(day)
        except Exception as exc:
            print(f"    [{day}] Could not get Nifty open: {exc}")
            return None

        atm = NiftyOptionService.atm_strike(nifty_open)
        from_dt = datetime(day.year, day.month, day.day, 9, 15, 0)
        to_dt   = datetime(day.year, day.month, day.day, 15, 30, 0)

        print(f"    {day}  |  open {nifty_open:.2f}  |  ATM {atm}"
              f"  |  expiry {expiry}")

        # Leg strikes needed for both possible directions.
        leg_specs = {
            "CE_BUY":  (atm,                        "CE"),
            "CE_SELL": (atm + self.spread_distance, "CE"),
            "PE_BUY":  (atm,                        "PE"),
            "PE_SELL": (atm - self.spread_distance, "PE"),
        }

        if self.cache_only:
            for strike, opt in leg_specs.values():
                cached = self.nifty_service.get_option_candles(
                    strike=strike, expiry_date=expiry, option_type=opt,
                    start=from_dt, end=to_dt, interval=self.interval, cache_only=True,
                )
                if not cached:
                    print(f"      [cache-only] No cached data for {strike}{opt} — skipping {day}")
                    return None

        result = ORBSpreadDayResult(
            trade_date=day, expiry_date=expiry, atm_strike=atm, nifty_open=nifty_open,
        )

        try:
            spot_candles = self.nifty_service.get_nifty_spot_candles(
                start=from_dt, end=to_dt, interval=self.interval,
            )
            if self.resample_seconds > 1:
                spot_candles = resample_candles(spot_candles, self.resample_seconds)

            leg_prices: dict[str, pd.Series] = {}
            for key, (strike, opt) in leg_specs.items():
                candles = self._fetch_leg(strike, opt, expiry, from_dt, to_dt)
                leg_prices[key] = _price_series(candles)

            trade = self._run_day(day, expiry, spot_candles, leg_prices, atm)
            if trade is not None:
                result.trades.append(trade)
                pnl_sign = "+" if trade.pnl >= 0 else ""
                print(f"      {trade.spread_type}  {trade.buy_strike}/{trade.sell_strike}"
                      f"  pnl {pnl_sign}{trade.pnl:.2f}  ({trade.exit_reason})")
            else:
                print(f"      no breakout / no trade")
        except Exception as exc:
            print(f"      [error] {exc}")

        return result

    # ── Reporting ──────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(day_results: list[ORBSpreadDayResult]) -> None:
        all_trades: list[ORBSpreadTradeResult] = []
        for dr in day_results:
            all_trades.extend(dr.trades)

        sep = "─" * 110
        print(f"\n{'='*110}")
        print("  RESULTS BY DAY")
        print(f"{'='*110}")

        header = (
            f"  {'Date':<12} {'Spread':<18} {'Strikes':>12}"
            f" {'Entry':>15} {'Exit':>15}"
            f" {'Debit':>8} {'ExitVal':>8} {'Qty':>5}"
            f" {'PnL':>11} {'Reason':<14}"
        )
        print(header)
        print(f"  {sep}")

        for t in sorted(all_trades, key=lambda x: x.entry_time):
            pnl_sign = "+" if t.pnl >= 0 else ""
            print(
                f"  {str(t.entry_time.date()):<12}"
                f" {t.spread_type:<18}"
                f" {f'{t.buy_strike}/{t.sell_strike}':>12}"
                f" {t.entry_time.strftime('%d-%b %H:%M'):>15}"
                f" {t.exit_time.strftime('%d-%b %H:%M'):>15}"
                f" {t.net_debit:>8.2f}"
                f" {t.exit_value:>8.2f}"
                f" {t.quantity:>5}"
                f" {pnl_sign+f'{t.pnl:.2f}':>11}"
                f" {t.exit_reason:<14}"
            )

        print(f"\n{'='*110}")
        print("  METRICS")
        print(f"{'='*110}")

        by_dir: dict[str, list[ORBSpreadTradeResult]] = defaultdict(list)
        for t in all_trades:
            by_dir[t.spread_type].append(t)

        col_w = {"lbl": 20, "trades": 7, "wins": 5, "loss": 6,
                 "wr": 6, "pnl": 12, "avg": 10, "pf": 7,
                 "best": 10, "worst": 10, "dur": 7, "cons": 5}

        hdr = (
            f"  {'Spread':<{col_w['lbl']}} {'Trades':>{col_w['trades']}}"
            f" {'Wins':>{col_w['wins']}} {'Loss':>{col_w['loss']}}"
            f" {'Win%':>{col_w['wr']}} {'Total PnL':>{col_w['pnl']}}"
            f" {'Avg PnL':>{col_w['avg']}} {'PF':>{col_w['pf']}}"
            f" {'Best':>{col_w['best']}} {'Worst':>{col_w['worst']}}"
            f" {'AvgMin':>{col_w['dur']}} {'MaxCL':>{col_w['cons']}}"
        )
        print(hdr)
        print(f"  {'─'*108}")

        def _print_row(m: ORBSpreadMetrics) -> None:
            pnl_s = f"{'+' if m.total_pnl >= 0 else ''}{m.total_pnl:.2f}"
            avg_s = f"{'+' if m.avg_pnl >= 0 else ''}{m.avg_pnl:.2f}"
            pf_s  = f"{m.profit_factor:.2f}" if m.profit_factor != float("inf") else "∞"
            print(
                f"  {m.label:<{col_w['lbl']}} {m.total_trades:>{col_w['trades']}}"
                f" {m.wins:>{col_w['wins']}} {m.losses:>{col_w['loss']}}"
                f" {m.win_rate:>{col_w['wr']}.1f} {pnl_s:>{col_w['pnl']}}"
                f" {avg_s:>{col_w['avg']}} {pf_s:>{col_w['pf']}}"
                f" {m.best_trade:>{col_w['best']}.2f} {m.worst_trade:>{col_w['worst']}.2f}"
                f" {m.avg_duration_minutes:>{col_w['dur']}.1f} {m.max_consecutive_losses:>{col_w['cons']}}"
            )

        for label, trades in sorted(by_dir.items()):
            _print_row(_compute_metrics(label, trades))

        print(f"  {'─'*108}")
        _print_row(_compute_metrics("OVERALL", all_trades))
        print(f"{'='*110}\n")
