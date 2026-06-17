"""
Real Strength Histogram strategy for Nifty weekly options on 1-second data.

This is the "seconds" analogue of :class:`RealStrengthOptionStrategy`. The
trading logic — the Real Strength composite oscillator, the dual-SMA trend
filter, the regime-dependent exits (flip vs. peak-drop), the static stop loss
and the re-entry lock — is identical and is reused from the base class' state
machine. What this variant adds is the data-handling machinery shared by the
other *WeeklyOptionsSeconds* runners:

  * `interval` / `resample_seconds` : data is fetched as raw 1-second bars and
    resampled locally to N-second candles before the indicators and the state
    machine run. Both the Nifty spot signal series and the option price series
    are resampled to the same timeframe.
  * `cache_only`                    : when True, candle data is served only from
    the local cache; any expiry/day whose option data is not cached is skipped.
  * `market_holidays`               : a Tuesday expiry that lands on a holiday is
    rolled back to the previous trading day.
  * `per_day_atm`                   : when True a fresh ATM strike is chosen for
    each trading day from that day's Nifty open; when False a single ATM strike
    anchored to the week's Monday open is traded across the whole expiry window.

Signals are generated from Nifty 50 spot candles; trades are executed on the
ATM CE or PE option contract for the expiry week.

Entry (Long -> CE):
  - Real Strength histogram > strength_threshold
  - Histogram rising vs previous bar
  - ADX >= min_adx
  - DI+ > DI-
  - Volume ratio >= vol_ratio_min
  - Optional: SMA(fast) > SMA(slow)

Entry (Short -> PE): mirror conditions on the bearish side.

Exit — regime-dependent:
  - Hard static stop loss (always active)
  - Min hold of min_bars_hold bars before peak/flip exits fire
  - SMA still confirms -> exit when histogram crosses into the opposite zone
    past flip_threshold (+/- flip_threshold)
  - SMA reversed against position -> exit when histogram falls peak_drop_pct%
    from its peak value reached during the trade
  - Square-off at 15:20 IST

Re-entry lock: after a stop loss, no re-entry in the same direction until the
histogram returns to (or crosses) zero.
"""

from datetime import date, datetime

from models.real_strength_models import RSWeeklyExpiryResult
from services.nifty_option_service import NiftyOptionService
from strategy.adx_option_strategy import resample_candles
from strategy.real_strength_option_strategy import (
    RealStrengthOptionStrategy,
    _build_price_map,
    compute_real_strength,
)


class RealStrengthOptionSecondsStrategy(RealStrengthOptionStrategy):
    def __init__(
        self,
        nifty_service: NiftyOptionService,
        capital: float = 100_000.0,
        adx_period: int = 14,
        roc_period: int = 10,
        vol_ma_period: int = 20,
        smooth_period: int = 3,
        strength_threshold: float = 1.0,
        min_adx: float = 14.0,
        vol_ratio_min: float = 1.2,
        sma_fast: int = 30,
        sma_slow: int = 60,
        use_sma_filter: bool = True,
        stop_loss_pct: float = 1.0,
        peak_drop_pct: float = 25.0,
        flip_threshold: float = 0.8,
        min_bars_hold: int = 3,
        start_date: str = "",
        end_date: str = "",
        interval: str = "1second",
        resample_seconds: int = 5,
        print_resampled: bool = False,
        cache_only: bool = False,
        market_holidays: set[date] | None = None,
        per_day_atm: bool = False,
    ):
        super().__init__(
            nifty_service=nifty_service,
            capital=capital,
            adx_period=adx_period,
            roc_period=roc_period,
            vol_ma_period=vol_ma_period,
            smooth_period=smooth_period,
            strength_threshold=strength_threshold,
            min_adx=min_adx,
            vol_ratio_min=vol_ratio_min,
            sma_fast=sma_fast,
            sma_slow=sma_slow,
            use_sma_filter=use_sma_filter,
            stop_loss_pct=stop_loss_pct,
            peak_drop_pct=peak_drop_pct,
            flip_threshold=flip_threshold,
            min_bars_hold=min_bars_hold,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )
        # Candle size (in seconds) used for the strategy. Data is always fetched
        # as raw 1-second bars and resampled locally before the indicators run.
        self.resample_seconds = resample_seconds
        self.print_resampled  = print_resampled
        # When True, candle data is served only from the local cache; any expiry
        # (or day, with per_day_atm) whose option data is not cached is skipped.
        self.cache_only       = cache_only
        self.market_holidays  = set(market_holidays) if market_holidays else set()
        # When True a fresh ATM strike is chosen for each trading day; when False
        # a single ATM strike anchored to the week's Monday open is traded across
        # the whole expiry window.
        self.per_day_atm      = per_day_atm

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _signal_df(self, from_dt: datetime, to_dt: datetime):
        """Fetch spot candles, resample to the strategy timeframe, compute RS."""
        spot_candles = self.nifty_service.get_nifty_spot_candles(
            start=from_dt, end=to_dt, interval=self.interval
        )
        if not spot_candles:
            return None
        if self.resample_seconds > 1:
            spot_candles = resample_candles(spot_candles, self.resample_seconds)
        return compute_real_strength(
            candles=spot_candles,
            adx_period=self.adx_period,
            roc_period=self.roc_period,
            vol_ma_period=self.vol_ma_period,
            smooth_period=self.smooth_period,
            sma_fast=self.sma_fast,
            sma_slow=self.sma_slow,
        )

    def _option_prices(
        self, strike: int, expiry: date, from_dt: datetime, to_dt: datetime
    ) -> dict[str, dict[datetime, float]]:
        """Fetch and resample CE/PE option candles into datetime->close maps."""
        prices: dict[str, dict[datetime, float]] = {"CE": {}, "PE": {}}
        for opt_type in ("CE", "PE"):
            try:
                opt_candles = self.nifty_service.get_option_candles(
                    strike=strike, expiry_date=expiry, option_type=opt_type,
                    start=from_dt, end=to_dt, interval=self.interval,
                    cache_only=self.cache_only,
                ) or []
                if self.resample_seconds > 1:
                    opt_candles = resample_candles(opt_candles, self.resample_seconds)
                prices[opt_type] = _build_price_map(opt_candles)
                print(f"    {opt_type}: {len(opt_candles)} candles fetched")
            except Exception as exc:
                print(f"    [{opt_type}] Fetch error: {exc}")
        return prices

    def _option_data_cached(
        self, strike: int, expiry: date, from_dt: datetime, to_dt: datetime
    ) -> bool:
        """True if both CE and PE option contracts are present in the cache."""
        for opt_type in ("CE", "PE"):
            cached = self.nifty_service.get_option_candles(
                strike=strike, expiry_date=expiry, option_type=opt_type,
                start=from_dt, end=to_dt, interval=self.interval, cache_only=True,
            )
            if not cached:
                return False
        return True

    # ── Weekly expiry orchestration ───────────────────────────────────────────

    def run_weekly_backtest(self) -> list[RSWeeklyExpiryResult]:
        expiry_results: list[RSWeeklyExpiryResult] = []

        wednesdays = NiftyOptionService.weekly_wednesdays(self.start_date, self.end_date)
        effective_tf = (
            f"{self.resample_seconds}s" if self.resample_seconds > 1 else self.interval
        )
        print(f"\n{'='*70}")
        print(f"  NIFTY REAL STRENGTH STRATEGY — WEEKLY EXPIRY BACKTEST (seconds data)")
        print(f"  Period   : {self.start_date}  →  {self.end_date}")
        print(f"  Fetch TF : {self.interval}  |  Effective TF: {effective_tf}")
        print(f"  Capital  : ₹{self.capital:,.0f}")
        print(f"  Threshold: {self.strength_threshold}  |  Min ADX: {self.min_adx}"
              f"  |  Vol Ratio: {self.vol_ratio_min}")
        print(f"  SMA      : {self.sma_fast}/{self.sma_slow}"
              f"  (filter {'ON' if self.use_sma_filter else 'OFF'})")
        print(f"  Stop Loss: {self.stop_loss_pct*100:.1f}%"
              f"  |  Peak Drop: {self.peak_drop_pct*100:.0f}%"
              f"  |  Flip: ±{self.flip_threshold}")
        print(f"  Mode     : {'per-day ATM' if self.per_day_atm else 'weekly ATM'}"
              f"  |  Cache-only: {self.cache_only}")
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
    ) -> RSWeeklyExpiryResult | None:
        try:
            nifty_open = self.nifty_service.get_nifty_open(monday)
        except Exception as exc:
            print(f"  [{expiry}] Could not get Nifty open for {monday}: {exc}")
            return None

        strike = NiftyOptionService.atm_strike(nifty_open)
        print(f"  Expiry {expiry}  |  Monday open {nifty_open:.2f}  |  ATM {strike}")

        from_dt = datetime(win_start.year, win_start.month, win_start.day, 9, 15, 0)
        to_dt   = datetime(win_end.year,   win_end.month,   win_end.day,   15, 30, 0)

        if self.cache_only and not self._option_data_cached(strike, expiry, from_dt, to_dt):
            print(f"    [cache-only] No cached data — skipping expiry {expiry}")
            return None

        signal_df = self._signal_df(from_dt, to_dt)
        if signal_df is None:
            print(f"    No spot candles — skipping")
            return None

        prices = self._option_prices(strike, expiry, from_dt, to_dt)

        week_result = RSWeeklyExpiryResult(
            expiry_date=expiry,
            atm_strike=strike,
            nifty_open=nifty_open,
        )

        ce_trades, pe_trades = self._run_expiry(
            signal_df=signal_df,
            ce_prices=prices["CE"],
            pe_prices=prices["PE"],
            strike=strike,
            expiry=expiry,
        )
        week_result.ce_trades = ce_trades
        week_result.pe_trades = pe_trades

        self._print_week_summary(week_result, ce_trades, pe_trades)
        return week_result

    def _run_expiry_per_day(
        self, expiry: date, win_start: date, win_end: date
    ) -> RSWeeklyExpiryResult | None:
        print(f"  Expiry {expiry}  |  per-day ATM")

        week_result = RSWeeklyExpiryResult(
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

            if self.cache_only and not self._option_data_cached(
                strike, expiry, from_dt, to_dt
            ):
                print(f"      [cache-only] No cached data — skipping {day}")
                continue

            signal_df = self._signal_df(from_dt, to_dt)
            if signal_df is None:
                print(f"      No spot candles — skipping {day}")
                continue

            prices = self._option_prices(strike, expiry, from_dt, to_dt)

            ce_trades, pe_trades = self._run_expiry(
                signal_df=signal_df,
                ce_prices=prices["CE"],
                pe_prices=prices["PE"],
                strike=strike,
                expiry=expiry,
            )
            week_result.ce_trades.extend(ce_trades)
            week_result.pe_trades.extend(pe_trades)
            print(f"      Trades: CE {len(ce_trades)}  PE {len(pe_trades)}")

        if week_result.all_trades:
            self._print_week_summary(week_result, [], [], header=False)
        return week_result

    @staticmethod
    def _print_week_summary(week_result, ce_trades, pe_trades, header=True):
        total = len(week_result.all_trades)
        week_pnl = sum(t.pnl for t in week_result.all_trades)
        pnl_sign = "+" if week_pnl >= 0 else ""
        if header:
            print(f"    Trades: {total} (CE:{len(ce_trades)} PE:{len(pe_trades)})"
                  f"  |  PnL: {pnl_sign}{week_pnl:.2f}")
        else:
            print(f"    Week total: {total} trades  |  PnL: {pnl_sign}{week_pnl:.2f}")
