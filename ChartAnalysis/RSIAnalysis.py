"""
Nifty 50 Index — Intraday RSI Extremes Analysis (1-minute data)
================================================================

Fetches 1-minute Nifty 50 spot (cash) candles for every trading day in the
configured window, computes RSI on the 1-minute closes, and reports — day by
day — how many crossover events occurred with:

  - RSI < 30  (oversold)
  - RSI > 70  (overbought)

Each excursion into an extreme zone is counted as ONE crossover, no matter
how many bars it lasts: e.g. once RSI drops below 30, the whole stretch until
it crosses back above 30 counts as a single oversold event. The same applies
to stretches above 70.

Data retrieval follows the same pattern as the BackTestScripts/
WeeklyOptionsSeconds scripts: a Breeze session is created from .env
credentials and candles are fetched through NiftyOptionService (which
transparently caches them on disk for later runs).

Usage:
  1. Set START_DATE / END_DATE to the desired window (DD-Mon-YYYY).
  2. Optionally adjust RSI_PERIOD / RSI_OVERSOLD / RSI_OVERBOUGHT.
  3. Run:  python -m ChartAnalysis.RSIAnalysis
"""

import os
from datetime import datetime, time

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.rsi_option_strategy import compute_rsi

load_dotenv()

# ── Session ───────────────────────────────────────────────────────────────────

breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN"),
)
print("Session Generated Successfully\n")

# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE     = "01-Jul-2026"   # format: DD-Mon-YYYY
END_DATE       = "10-Jul-2026"   # format: DD-Mon-YYYY

INTERVAL       = "1minute"       # candle size fetched from Breeze
RSI_PERIOD     = 14              # RSI lookback (in 1-minute bars)
RSI_OVERSOLD   = 30.0            # oversold threshold (RSI strictly below this)
RSI_OVERBOUGHT = 70.0            # overbought threshold (RSI strictly above this)

# Regular Nifty market hours (IST) — the fetch window for each trading day.
MARKET_OPEN    = time(9, 15)
MARKET_CLOSE   = time(15, 30)

# ── Analysis ──────────────────────────────────────────────────────────────────


def count_zone_crossovers(rsi, threshold: float, oversold: bool) -> int:
    """Count completed excursions of RSI into an extreme zone.

    A run of consecutive bars inside the zone (RSI < threshold when
    ``oversold``, RSI > threshold otherwise) is combined into a single
    event, counted once RSI crosses back out of the zone. An excursion
    still open at the last bar also counts as one event.
    """
    in_zone = (rsi < threshold) if oversold else (rsi > threshold)
    count = 0
    inside = False
    for flag in in_zone:
        if flag and not inside:
            inside = True          # entered the zone: start of one event
        elif not flag and inside:
            inside = False         # crossed back out: event completed
            count += 1
    if inside:                     # session ended while still in the zone
        count += 1
    return count


def main() -> None:
    service = NiftyOptionService(breeze)

    start = datetime.strptime(START_DATE, "%d-%b-%Y").date()
    end = datetime.strptime(END_DATE, "%d-%b-%Y").date()

    print(
        f"Nifty 50 — 1-minute RSI({RSI_PERIOD}) extremes, "
        f"{START_DATE} to {END_DATE}\n"
    )
    header = (
        f"{'Date':<12} {'Day':<10} {'Bars':>5} "
        f"{'RSI<' + str(int(RSI_OVERSOLD)):>8} "
        f"{'RSI>' + str(int(RSI_OVERBOUGHT)):>8} "
        f"{'Open':>10} {'Close':>10} {'%Chg':>8}"
    )
    print(header)
    print("-" * len(header))

    total_bars = total_oversold = total_overbought = 0
    days_with_data = 0

    # Fetch one trading day at a time: a full 1-minute session is 375 bars,
    # comfortably under Breeze's ~1000-rows-per-request cap.
    for day in NiftyOptionService.trading_days(start, end):
        day_start = datetime.combine(day, MARKET_OPEN)
        day_end = datetime.combine(day, MARKET_CLOSE)

        candles = service.get_nifty_spot_candles(day_start, day_end, INTERVAL)
        if not candles:
            print(f"{day.strftime('%d-%b-%Y'):<12} {day.strftime('%A'):<10}   no data")
            continue

        rsi_df = compute_rsi(candles, RSI_PERIOD)
        rsi = rsi_df["rsi"].dropna()

        bars = len(rsi_df)
        oversold = count_zone_crossovers(rsi, RSI_OVERSOLD, oversold=True)
        overbought = count_zone_crossovers(rsi, RSI_OVERBOUGHT, oversold=False)

        total_bars += bars
        total_oversold += oversold
        total_overbought += overbought
        days_with_data += 1

        day_open = float(candles[0]["open"])
        day_close = float(candles[-1]["close"])
        pct_change = (day_close - day_open) / day_open * 100

        print(
            f"{day.strftime('%d-%b-%Y'):<12} {day.strftime('%A'):<10} "
            f"{bars:>5} {oversold:>8} {overbought:>8} "
            f"{day_open:>10.2f} {day_close:>10.2f} {pct_change:>+8.2f}"
        )

    print("-" * len(header))
    print(
        f"{'TOTAL':<12} {str(days_with_data) + ' days':<10} "
        f"{total_bars:>5} {total_oversold:>8} {total_overbought:>8}"
    )


if __name__ == "__main__":
    main()
