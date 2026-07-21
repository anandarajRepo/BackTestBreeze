"""
Credit Spread (net-credit vertical spread) strategy for Nifty weekly options
(seconds data).

A credit spread simultaneously SELLS a higher-premium option and BUYS a
lower-premium option of the same type and expiry, collecting a net credit. The
strategy is a defined-risk, theta-positive (time-decay) play:

  • Bull Put Spread  — sell an OTM Put + buy a further-OTM Put (below it).
        Deployed on a bullish / sideways bias: profits while the Nifty stays
        above the short put strike as both puts decay.
  • Bear Call Spread — sell an OTM Call + buy a further-OTM Call (above it).
        Deployed on a bearish / sideways bias: profits while the Nifty stays
        below the short call strike as both calls decay.

Strike selection (per trade, from that day's ATM):
  short strike = ATM ∓ ``short_otm_distance``   (300-500 pts OTM is typical)
  long  strike = short strike ∓ ``spread_width`` (further OTM, protective)

Direction selection (``direction_mode``):
  • "TREND"     — build an opening range from the first ``or_minutes`` of Nifty
                  spot; a break ABOVE the range → bullish → Bull Put Spread,
                  a break BELOW the range → bearish → Bear Call Spread. One
                  position per day, taken on the first breakout.
  • "BULL_PUT"  — always deploy a Bull Put Spread at the opening-range end.
  • "BEAR_CALL" — always deploy a Bear Call Spread at the opening-range end.

Entry: one position per day, no re-entry after an exit within the session.

Exit (combined two-leg P&L monitored continuously on the spot timeline):
  • Combined P&L ≥ ``profit_target_pct`` % of the collected credit → PROFIT_TARGET
  • Combined P&L ≤ −``stop_loss_mult`` × the collected credit      → STOP_LOSS
  • Spot breaches the breakeven (short strike ∓ net credit), when
    ``breakeven_exit_enabled`` is True                             → BREAKEVEN_BREACH
  • Time = 15:25 IST (square off whatever is still open)           → SQUARE_OFF

Combined P&L (per unit) for a credit spread:
    net_credit   = sell_entry − buy_entry           (collected at entry)
    exit_value   = sell_now   − buy_now             (cost to close now)
    per_unit_pnl = net_credit − exit_value
                 = (sell_entry − sell_now) + (buy_now − buy_entry)
    total_pnl    = per_unit_pnl × lot_size × lots
"""

from collections import defaultdict
from datetime import date, datetime, time, timedelta

import pandas as pd

from models.credit_spread_models import (
    CreditSpreadDayResult,
    CreditSpreadMetrics,
    CreditSpreadTradeResult,
)
from services.nifty_option_service import NiftyOptionService
from strategy.orb_option_seconds_strategy import resample_candles


_SQUARE_OFF = time(15, 25)


def _compute_metrics(label: str, trades: list[CreditSpreadTradeResult]) -> CreditSpreadMetrics:
    if not trades:
        return CreditSpreadMetrics(
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

    return CreditSpreadMetrics(
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


class CreditSpreadOptionSecondsStrategy:
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        or_minutes: int = 15,
        direction_mode: str = "TREND",
        short_otm_distance: int = 400,
        spread_width: int = 100,
        profit_target_pct: float = 50.0,
        stop_loss_mult: float = 2.0,
        breakeven_exit_enabled: bool = True,
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
        # Length of the opening range (minutes from 9:15) used to pick direction
        # in "TREND" mode and as the entry timestamp in the forced-side modes.
        self.or_minutes      = or_minutes
        # How the spread side is chosen each day. See module docstring.
        mode = (direction_mode or "TREND").upper()
        if mode not in ("TREND", "BULL_PUT", "BEAR_CALL"):
            raise ValueError(
                "direction_mode must be one of 'TREND', 'BULL_PUT', 'BEAR_CALL' "
                f"(got {direction_mode!r})"
            )
        self.direction_mode  = mode
        # Distance (index points) from the ATM to the SHORT strike. The short
        # leg is placed this far OTM (puts below spot / calls above spot). The
        # execution guidance targets 300-500 points for a high-probability OTM.
        self.short_otm_distance = short_otm_distance
        # Distance (index points) between the short strike and the further-OTM
        # long (protective) strike. This is the spread width and the reference
        # for the maximum defined risk (width − net credit).
        self.spread_width    = spread_width
        # Profit target as a percentage of the collected net credit. The position
        # is closed once the combined P&L reaches this fraction of the credit.
        self.profit_target_pct = profit_target_pct
        # Stop-loss as a multiple of the collected net credit. The position is
        # closed once the combined loss reaches stop_loss_mult × credit.
        self.stop_loss_mult  = stop_loss_mult
        # When True, the position is also closed if the Nifty spot breaches the
        # breakeven (short strike ∓ net credit) — the risk-management exit called
        # out in the execution tips.
        self.breakeven_exit_enabled = breakeven_exit_enabled
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
    ) -> CreditSpreadTradeResult | None:
        """
        Run the credit spread on a single trading day.

        ``leg_prices`` maps a leg key to its close-price Series:
            "PE_SELL", "PE_BUY"  (bull put legs)
            "CE_SELL", "CE_BUY"  (bear call legs)
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

        # Strike geometry for both possible spreads.
        bull_put_sell  = atm_strike - self.short_otm_distance
        bull_put_buy   = bull_put_sell - self.spread_width
        bear_call_sell = atm_strike + self.short_otm_distance
        bear_call_buy  = bear_call_sell + self.spread_width

        in_position = False
        direction   = ""            # "BULL_PUT" or "BEAR_CALL"
        sell_leg = buy_leg = ""
        sell_strike = buy_strike = 0
        sell_entry = buy_entry = 0.0
        net_credit = 0.0
        breakeven  = 0.0
        entry_dt: datetime | None = None
        spot_at_entry = 0.0

        def _open_position(sig: str, i: int, spot_close: float) -> bool:
            """Try to open the spread for signal ``sig`` at post-OR index ``i``."""
            nonlocal in_position, direction, sell_leg, buy_leg
            nonlocal sell_strike, buy_strike, sell_entry, buy_entry
            nonlocal net_credit, breakeven, entry_dt, spot_at_entry

            if sig == "BULL_PUT":
                s_leg, b_leg = "PE_SELL", "PE_BUY"
                s_strike, b_strike = bull_put_sell, bull_put_buy
            else:  # BEAR_CALL
                s_leg, b_leg = "CE_SELL", "CE_BUY"
                s_strike, b_strike = bear_call_sell, bear_call_buy

            s = float(legs[s_leg].iloc[i])
            b = float(legs[b_leg].iloc[i])
            # A credit spread only makes sense when the short leg is richer than
            # the long leg (positive credit collected).
            if pd.isna(s) or pd.isna(b) or (s - b) <= 0:
                return False

            direction    = sig
            sell_leg, buy_leg       = s_leg, b_leg
            sell_strike, buy_strike = s_strike, b_strike
            sell_entry, buy_entry   = s, b
            net_credit   = s - b
            breakeven = (
                s_strike - net_credit if sig == "BULL_PUT"
                else s_strike + net_credit
            )
            entry_dt     = post_df.iloc[i]["datetime"]
            spot_at_entry = spot_close
            in_position  = True
            return True

        for i in range(len(post_df)):
            row = post_df.iloc[i]
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            close        = float(row["close"])

            # ── Exit logic (combined P&L monitored continuously) ──────────
            if in_position:
                sell_now = float(legs[sell_leg].iloc[i])
                buy_now  = float(legs[buy_leg].iloc[i])
                if pd.isna(sell_now) or pd.isna(buy_now):
                    continue

                exit_value = sell_now - buy_now
                per_unit   = net_credit - exit_value
                pnl        = per_unit * self.quantity

                profit_target = self.profit_target_pct / 100.0 * net_credit * self.quantity
                stop_loss     = self.stop_loss_mult * net_credit * self.quantity

                breakeven_breached = self.breakeven_exit_enabled and (
                    (direction == "BULL_PUT"  and close <= breakeven) or
                    (direction == "BEAR_CALL" and close >= breakeven)
                )

                exit_reason = None
                if t >= _SQUARE_OFF:
                    exit_reason = "SQUARE_OFF"
                elif pnl >= profit_target:
                    exit_reason = "PROFIT_TARGET"
                elif pnl <= -stop_loss:
                    exit_reason = "STOP_LOSS"
                elif breakeven_breached:
                    exit_reason = "BREAKEVEN_BREACH"

                if exit_reason:
                    duration = int((dt - entry_dt).total_seconds() / 60)
                    return self._build_trade(
                        direction, expiry, sell_strike, buy_strike,
                        entry_dt, dt, sell_entry, buy_entry, sell_now, buy_now,
                        net_credit, exit_value, breakeven,
                        spot_at_entry, close, exit_reason, duration,
                    )
                continue

            # ── Entry logic (no re-entry once a position has been taken) ──
            if t >= _SQUARE_OFF:
                break

            signal = None
            if self.direction_mode == "TREND":
                # Breakout ABOVE the opening range → bullish → sell puts below.
                # Breakdown BELOW the opening range → bearish → sell calls above.
                if close > or_high:
                    signal = "BULL_PUT"
                elif close < or_low:
                    signal = "BEAR_CALL"
            else:
                # Forced-side modes deploy at the first post-OR bar.
                signal = self.direction_mode

            if signal is None:
                continue

            _open_position(signal, i, close)

        # Force square-off at end of day's data if still open.
        if in_position and entry_dt is not None:
            last_i   = len(post_df) - 1
            row      = post_df.iloc[last_i]
            dt       = row["datetime"]
            close    = float(row["close"])
            sell_now = float(legs[sell_leg].iloc[last_i])
            buy_now  = float(legs[buy_leg].iloc[last_i])
            exit_value = sell_now - buy_now
            duration = int((dt - entry_dt).total_seconds() / 60)
            return self._build_trade(
                direction, expiry, sell_strike, buy_strike,
                entry_dt, dt, sell_entry, buy_entry, sell_now, buy_now,
                net_credit, exit_value, breakeven,
                spot_at_entry, close, "SQUARE_OFF", duration,
            )

        return None

    def _build_trade(
        self,
        direction: str,
        expiry: date,
        sell_strike: int,
        buy_strike: int,
        entry_dt: datetime,
        exit_dt: datetime,
        sell_entry: float,
        buy_entry: float,
        sell_exit: float,
        buy_exit: float,
        net_credit: float,
        exit_value: float,
        breakeven: float,
        spot_at_entry: float,
        spot_at_exit: float,
        exit_reason: str,
        duration: int,
    ) -> CreditSpreadTradeResult:
        per_unit = net_credit - exit_value
        pnl      = per_unit * self.quantity
        return CreditSpreadTradeResult(
            direction=direction,
            spread_type=(
                "BULL_PUT_SPREAD" if direction == "BULL_PUT" else "BEAR_CALL_SPREAD"
            ),
            option_type="PE" if direction == "BULL_PUT" else "CE",
            expiry_date=expiry,
            sell_strike=sell_strike,
            buy_strike=buy_strike,
            entry_time=entry_dt,
            exit_time=exit_dt,
            sell_entry=round(sell_entry, 2),
            buy_entry=round(buy_entry, 2),
            sell_exit=round(sell_exit, 2),
            buy_exit=round(buy_exit, 2),
            net_credit=round(net_credit, 2),
            exit_value=round(exit_value, 2),
            spread_width=abs(sell_strike - buy_strike),
            quantity=self.quantity,
            pnl=round(pnl, 2),
            exit_reason=exit_reason,
            breakeven=round(breakeven, 2),
            spot_at_entry=round(spot_at_entry, 2),
            spot_at_exit=round(spot_at_exit, 2),
            duration_minutes=duration,
        )

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

    def _leg_specs(self, atm: int) -> dict[str, tuple[int, str]]:
        """Return the leg strikes needed for the day's direction mode."""
        specs: dict[str, tuple[int, str]] = {}
        if self.direction_mode in ("TREND", "BULL_PUT"):
            sell = atm - self.short_otm_distance
            specs["PE_SELL"] = (sell, "PE")
            specs["PE_BUY"]  = (sell - self.spread_width, "PE")
        if self.direction_mode in ("TREND", "BEAR_CALL"):
            sell = atm + self.short_otm_distance
            specs["CE_SELL"] = (sell, "CE")
            specs["CE_BUY"]  = (sell + self.spread_width, "CE")
        return specs

    def run_backtest(self) -> list[CreditSpreadDayResult]:
        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        effective_tf = (
            f"{self.resample_seconds}s" if self.resample_seconds > 1 else self.interval
        )
        print(f"\n{'='*70}")
        print(f"  NIFTY CREDIT SPREAD — WEEKLY EXPIRY BACKTEST")
        print(f"  Period    : {self.start_date}  →  {self.end_date}")
        print(f"  Fetch TF  : {self.interval}  |  Effective TF: {effective_tf}")
        print(f"  Mode      : {self.direction_mode}  |  OR window: {self.or_minutes}min"
              f"  |  Short OTM: {self.short_otm_distance}  |  Width: {self.spread_width}")
        print(f"  Exits     : Target {self.profit_target_pct:.0f}% credit"
              f"  |  SL {self.stop_loss_mult:.1f}× credit"
              f"  |  Breakeven exit: {'on' if self.breakeven_exit_enabled else 'off'}")
        print(f"  Lot size  : {self.lot_size} × {self.lots} lot(s) = {self.quantity} units")
        print(f"  Expiries found: {len(wednesdays)}")
        print(f"{'='*70}\n")

        day_results: list[CreditSpreadDayResult] = []

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

    def _run_single_day(self, day: date, expiry: date) -> CreditSpreadDayResult | None:
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

        leg_specs = self._leg_specs(atm)

        if self.cache_only:
            for strike, opt in leg_specs.values():
                cached = self.nifty_service.get_option_candles(
                    strike=strike, expiry_date=expiry, option_type=opt,
                    start=from_dt, end=to_dt, interval=self.interval, cache_only=True,
                )
                if not cached:
                    print(f"      [cache-only] No cached data for {strike}{opt} — skipping {day}")
                    return None

        result = CreditSpreadDayResult(
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
                print(f"      {trade.spread_type}  {trade.sell_strike}/{trade.buy_strike}"
                      f"  credit {trade.net_credit:.2f}"
                      f"  pnl {pnl_sign}{trade.pnl:.2f}  ({trade.exit_reason})")
            else:
                print(f"      no signal / no trade")
        except Exception as exc:
            print(f"      [error] {exc}")

        return result

    # ── Reporting ──────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(day_results: list[CreditSpreadDayResult]) -> None:
        all_trades: list[CreditSpreadTradeResult] = []
        for dr in day_results:
            all_trades.extend(dr.trades)

        sep = "─" * 120
        print(f"\n{'='*120}")
        print("  RESULTS BY DAY")
        print(f"{'='*120}")

        header = (
            f"  {'Date':<12} {'Spread':<18} {'Strikes':>12}"
            f" {'Entry':>15} {'Exit':>15}"
            f" {'Credit':>8} {'ExitVal':>8} {'BrkEven':>9} {'Qty':>5}"
            f" {'PnL':>11} {'Reason':<16}"
        )
        print(header)
        print(f"  {sep}")

        for t in sorted(all_trades, key=lambda x: x.entry_time):
            pnl_sign = "+" if t.pnl >= 0 else ""
            print(
                f"  {str(t.entry_time.date()):<12}"
                f" {t.spread_type:<18}"
                f" {f'{t.sell_strike}/{t.buy_strike}':>12}"
                f" {t.entry_time.strftime('%d-%b %H:%M'):>15}"
                f" {t.exit_time.strftime('%d-%b %H:%M'):>15}"
                f" {t.net_credit:>8.2f}"
                f" {t.exit_value:>8.2f}"
                f" {t.breakeven:>9.2f}"
                f" {t.quantity:>5}"
                f" {pnl_sign+f'{t.pnl:.2f}':>11}"
                f" {t.exit_reason:<16}"
            )

        print(f"\n{'='*120}")
        print("  METRICS")
        print(f"{'='*120}")

        by_dir: dict[str, list[CreditSpreadTradeResult]] = defaultdict(list)
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

        def _print_row(m: CreditSpreadMetrics) -> None:
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
        print(f"{'='*120}\n")
