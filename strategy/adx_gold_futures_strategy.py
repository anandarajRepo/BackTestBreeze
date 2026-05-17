"""
ADX DI+/DI- crossover strategy for GOLD MCX futures (long & short).

Entry rules:
  LONG  — buy futures when DI+ crosses above DI- and ADX >= threshold
  SHORT — sell futures when DI- crosses above DI+ and ADX >= threshold

Exit rules:
  - DI direction reversal (crossover flips) → also opens reverse position
  - Square-off at 23:25 IST (MCX evening session close)
  - Max 5 trades per day
  - No new entries before 09:00 or after 22:45

GOLD futures contract structure (MCX):
  Expiry : 5th of even months (Feb/Apr/Jun/Aug/Oct/Dec), adjusted for
           weekends/holidays per GOLD_FUTURES_EXPIRY_DATES lookup.
  Lot size: 1 kg (100 grams × 10 = 1000 grams? — configurable via LOT_SIZE)
"""

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from math import floor

import pandas as pd

from models.commodity_models import (
    CommoditySymbolMetrics,
    FuturesExpiryResult,
    FuturesTradeResult,
)
from services.commodity_option_service import (
    GOLD_FUTURES_CONTRACT_MONTHS,
    CommodityOptionService,
)


_ENTRY_START        = time(9, 0)
_ENTRY_CUTOFF       = time(22, 45)
_SQUARE_OFF         = time(23, 25)
_MAX_TRADES_PER_DAY = 5

GOLD_LOT_SIZE = 1   # 1 kg per lot on MCX GOLD (standard contract)


def _wilder_ewm(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_adx(candles: list[dict], period: int = 14) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    prev_high  = df["high"].shift(1)
    prev_low   = df["low"].shift(1)
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move   = df["high"] - prev_high
    down_move = prev_low   - df["low"]

    plus_dm  = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move

    smooth_tr       = _wilder_ewm(tr,       period)
    smooth_plus_dm  = _wilder_ewm(plus_dm,  period)
    smooth_minus_dm = _wilder_ewm(minus_dm, period)

    di_plus  = 100 * smooth_plus_dm  / smooth_tr
    di_minus = 100 * smooth_minus_dm / smooth_tr

    dx_denom = di_plus + di_minus
    dx = (100 * (di_plus - di_minus).abs() / dx_denom).where(dx_denom != 0, 0.0)
    adx = _wilder_ewm(dx, period)

    result = df[["datetime"]].copy()
    result["adx"]      = adx.round(4)
    result["di_plus"]  = di_plus.round(4)
    result["di_minus"] = di_minus.round(4)
    result["close"]    = df["close"]
    return result


def _compute_futures_metrics(
    trades: list[FuturesTradeResult],
) -> CommoditySymbolMetrics:
    symbol = "GOLD-FUT"
    if not trades:
        return CommoditySymbolMetrics(
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

    return CommoditySymbolMetrics(
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


class ADXGoldFuturesStrategy:
    def __init__(
        self,
        commodity_service: CommodityOptionService,
        capital: float = 100_000.0,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        start_date: str = "",
        end_date: str = "",
        interval: str = "1minute",
        lot_size: int = GOLD_LOT_SIZE,
    ):
        self.commodity_service = commodity_service
        self.capital           = capital
        self.adx_period        = adx_period
        self.adx_threshold     = adx_threshold
        self.start_date        = datetime.strptime(start_date, "%d-%b-%Y").date()
        self.end_date          = datetime.strptime(end_date,   "%d-%b-%Y").date()
        self.interval          = interval
        self.lot_size          = lot_size

    # ── Core ADX futures engine ───────────────────────────────────────────────

    def _run_futures(
        self,
        candles: list[dict],
        expiry_date: date,
    ) -> list[FuturesTradeResult]:
        """Run ADX strategy on futures candles; returns both LONG and SHORT trades."""
        if not candles:
            return []

        indicators = compute_adx(candles, self.adx_period)
        trades: list[FuturesTradeResult] = []
        daily_trade_count: dict[date, int] = defaultdict(int)

        in_position   = False
        direction: str = ""
        entry_row     = None
        entry_price   = 0.0
        lots          = 0

        prev_di_plus  = None
        prev_di_minus = None

        for _, row in indicators.iterrows():
            dt: datetime = row["datetime"]
            t: time      = dt.time()
            today: date  = dt.date()

            adx      = row["adx"]
            di_plus  = row["di_plus"]
            di_minus = row["di_minus"]
            price    = row["close"]

            if pd.isna(adx) or pd.isna(di_plus) or pd.isna(di_minus):
                prev_di_plus  = di_plus
                prev_di_minus = di_minus
                continue

            # ── Exit / reversal logic ─────────────────────────────────────
            if in_position:
                exit_reason = None

                if t >= _SQUARE_OFF:
                    exit_reason = "SQUARE_OFF"
                elif direction == "LONG" and prev_di_plus is not None:
                    if prev_di_plus >= prev_di_minus and di_plus < di_minus:
                        exit_reason = "ADX_CROSSOVER"
                elif direction == "SHORT" and prev_di_plus is not None:
                    if prev_di_minus >= prev_di_plus and di_minus < di_plus:
                        exit_reason = "ADX_CROSSOVER"

                if exit_reason:
                    exit_price = price
                    duration   = int((dt - entry_row["datetime"]).total_seconds() / 60)
                    multiplier = 1 if direction == "LONG" else -1
                    pnl = round(lots * self.lot_size * multiplier * (exit_price - entry_price), 2)

                    trades.append(FuturesTradeResult(
                        symbol=f"GOLD-FUT-{expiry_date}",
                        commodity="GOLD",
                        direction=direction,
                        expiry_date=expiry_date,
                        entry_time=entry_row["datetime"],
                        exit_time=dt,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        lots=lots,
                        lot_size=self.lot_size,
                        pnl=pnl,
                        exit_reason=exit_reason,
                        adx_at_entry=entry_row["adx"],
                        di_plus_at_entry=entry_row["di_plus"],
                        di_minus_at_entry=entry_row["di_minus"],
                        adx_at_exit=adx,
                        di_plus_at_exit=di_plus,
                        di_minus_at_exit=di_minus,
                        duration_minutes=duration,
                    ))

                    in_position = False
                    entry_row   = None

            # ── Entry logic ───────────────────────────────────────────────
            if (
                not in_position
                and prev_di_plus is not None
                and _ENTRY_START <= t <= _ENTRY_CUTOFF
                and adx >= self.adx_threshold
                and daily_trade_count[today] < _MAX_TRADES_PER_DAY
                and price > 0
            ):
                new_direction = None
                if (prev_di_plus <= prev_di_minus) and (di_plus > di_minus):
                    new_direction = "LONG"
                elif (prev_di_minus <= prev_di_plus) and (di_minus > di_plus):
                    new_direction = "SHORT"

                if new_direction:
                    entry_price = price
                    lots        = max(floor(self.capital / (entry_price * self.lot_size)), 1)
                    direction   = new_direction
                    in_position = True
                    entry_row   = row
                    daily_trade_count[today] += 1

            prev_di_plus  = di_plus
            prev_di_minus = di_minus

        # Force-close any open position at end of data
        if in_position and entry_row is not None:
            last       = indicators.iloc[-1]
            exit_price = last["close"]
            dt         = last["datetime"]
            duration   = int((dt - entry_row["datetime"]).total_seconds() / 60)
            multiplier = 1 if direction == "LONG" else -1
            pnl = round(lots * self.lot_size * multiplier * (exit_price - entry_price), 2)

            trades.append(FuturesTradeResult(
                symbol=f"GOLD-FUT-{expiry_date}",
                commodity="GOLD",
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
                adx_at_entry=entry_row["adx"],
                di_plus_at_entry=entry_row["di_plus"],
                di_minus_at_entry=entry_row["di_minus"],
                adx_at_exit=last["adx"],
                di_plus_at_exit=last["di_plus"],
                di_minus_at_exit=last["di_minus"],
                duration_minutes=duration,
            ))

        return trades

    # ── Main backtest orchestrator ────────────────────────────────────────────

    def run_backtest(self) -> list[FuturesExpiryResult]:
        """
        Iterate over each GOLD futures contract in the backtest window.
        For each contract, fetch daily futures candles and apply the ADX strategy.
        Futures contracts exist for even months only (Feb/Apr/Jun/Aug/Oct/Dec).
        """
        expiry_results: list[FuturesExpiryResult] = []

        print(f"\n{'='*80}")
        print(f"  GOLD FUTURES ADX STRATEGY — MCX")
        print(f"  Period     : {self.start_date}  →  {self.end_date}")
        print(f"  Capital    : ₹{self.capital:,.0f}  |  ADX period: {self.adx_period}"
              f"  |  ADX threshold: {self.adx_threshold}")
        print(f"  Lot size   : {self.lot_size} kg")
        print(f"{'='*80}\n")

        # Build list of (contract_expiry, window_start, window_end) for even months
        contracts = self._gold_futures_contracts()

        for futures_expiry, win_start, win_end in contracts:
            # Clamp to user's requested range
            win_start = max(win_start, self.start_date)
            win_end   = min(win_end,   self.end_date)
            if win_start > win_end:
                continue

            print(f"  Contract expiry {futures_expiry}  |  Window {win_start} → {win_end}")

            result = FuturesExpiryResult(expiry_date=futures_expiry, commodity="GOLD")
            all_trades: list[FuturesTradeResult] = []

            trade_date = win_start
            while trade_date <= win_end:
                if not self.commodity_service.is_mcx_trading_day(trade_date):
                    trade_date += timedelta(days=1)
                    continue

                day_start = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 0, 0)
                day_end   = datetime(trade_date.year, trade_date.month, trade_date.day, 23, 30, 0)

                try:
                    candles = self.commodity_service.get_futures_candles(
                        stock_code="GOLD",
                        expiry_date=futures_expiry,
                        start=day_start,
                        end=day_end,
                        interval=self.interval,
                    )
                    day_trades = self._run_futures(candles, futures_expiry)
                    all_trades.extend(day_trades)

                    if day_trades:
                        day_pnl = sum(t.pnl for t in day_trades)
                        print(
                            f"    {trade_date}: {len(day_trades)} trades"
                            f"  PnL ₹{day_pnl:+.2f}"
                        )
                except Exception as exc:
                    print(f"    [{trade_date}] error: {exc}")

                trade_date += timedelta(days=1)

            result.trades = all_trades
            expiry_results.append(result)

            total_pnl = sum(t.pnl for t in all_trades)
            print(
                f"  → Expiry {futures_expiry}: {len(all_trades)} trades"
                f"  PnL ₹{total_pnl:+.2f}\n"
            )

        return expiry_results

    def _gold_futures_contracts(self) -> list[tuple[date, date, date]]:
        """
        Return (futures_expiry, window_start, window_end) for each GOLD futures
        contract that overlaps the backtest period.

        Window: from 1st of the contract month to the futures expiry date.
        """
        contracts: list[tuple[date, date, date]] = []
        year  = self.start_date.year
        month = self.start_date.month

        # Step back one month to catch contracts that started before start_date
        if month == 1:
            year -= 1
            month = 12
        else:
            month -= 1

        seen: set[date] = set()
        for _ in range(30):  # at most 30 half-years
            if month in GOLD_FUTURES_CONTRACT_MONTHS:
                expiry = CommodityOptionService._nominal_futures_expiry(year, month)
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
    def print_report(expiry_results: list[FuturesExpiryResult]) -> None:
        all_trades: list[FuturesTradeResult] = []
        for er in expiry_results:
            all_trades.extend(er.trades)

        sep = "─" * 105

        print(f"\n{'='*105}")
        print("  RESULTS BY FUTURES CONTRACT")
        print(f"{'='*105}")

        for er in expiry_results:
            if not er.trades:
                continue
            total_pnl = sum(t.pnl for t in er.trades)
            longs  = [t for t in er.trades if t.direction == "LONG"]
            shorts = [t for t in er.trades if t.direction == "SHORT"]
            print(
                f"\n  Contract {er.expiry_date}  |  {er.commodity}"
                f"  |  Trades {len(er.trades)}"
                f" (L:{len(longs)} S:{len(shorts)})"
                f"  |  PnL ₹{total_pnl:+.2f}"
            )
            print(f"  {sep}")
            header = (
                f"  {'Direction':<7} {'Entry':>19} {'Exit':>19}"
                f" {'EntryPx':>9} {'ExitPx':>9} {'Lots':>4} {'LotSz':>5}"
                f" {'PnL':>11} {'Reason':<14}"
                f" {'ADX':>6} {'DI+':>6} {'DI-':>6}"
            )
            print(header)
            print(f"  {sep}")

            for t in sorted(er.trades, key=lambda x: x.entry_time):
                pnl_sign = "+" if t.pnl >= 0 else ""
                print(
                    f"  {t.direction:<7}"
                    f" {t.entry_time.strftime('%d-%b %H:%M'):>19}"
                    f" {t.exit_time.strftime('%d-%b %H:%M'):>19}"
                    f" {t.entry_price:>9.2f}"
                    f" {t.exit_price:>9.2f}"
                    f" {t.lots:>4}"
                    f" {t.lot_size:>5}"
                    f" {pnl_sign+f'{t.pnl:.2f}':>11}"
                    f" {t.exit_reason:<14}"
                    f" {t.adx_at_exit:>6.2f}"
                    f" {t.di_plus_at_exit:>6.2f}"
                    f" {t.di_minus_at_exit:>6.2f}"
                )

        # ── Overall metrics ───────────────────────────────────────────────
        m = _compute_futures_metrics(all_trades)

        print(f"\n{'='*105}")
        print("  OVERALL METRICS — GOLD FUTURES")
        print(f"{'='*105}")
        print(f"  Total trades       : {m.total_trades}")
        print(f"  Wins / Losses      : {m.wins} / {m.losses}")
        print(f"  Win rate           : {m.win_rate:.1f}%")
        print(f"  Total PnL          : ₹{m.total_pnl:+.2f}")
        print(f"  Avg PnL per trade  : ₹{m.avg_pnl:+.2f}")
        print(f"  Profit factor      : {m.profit_factor if m.profit_factor != float('inf') else '∞'}")
        print(f"  Best trade         : ₹{m.best_trade:+.2f}")
        print(f"  Worst trade        : ₹{m.worst_trade:+.2f}")
        print(f"  Avg duration (min) : {m.avg_duration_minutes:.1f}")
        print(f"  Max consec. losses : {m.max_consecutive_losses}")

        longs  = [t for t in all_trades if t.direction == "LONG"]
        shorts = [t for t in all_trades if t.direction == "SHORT"]
        long_pnl  = sum(t.pnl for t in longs)
        short_pnl = sum(t.pnl for t in shorts)
        print(f"\n  LONG  trades: {len(longs):>4}  |  PnL ₹{long_pnl:+.2f}")
        print(f"  SHORT trades: {len(shorts):>4}  |  PnL ₹{short_pnl:+.2f}")
        print(f"{'='*105}\n")
