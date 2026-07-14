"""
Nifty 50 Index — Intraday RSI Extremes Analysis (1-minute data)
================================================================

Fetches 1-minute Nifty 50 spot (cash) candles for every trading day in the
configured window, computes RSI on the 1-minute closes, and reports — day by
day — two measures for each extreme zone:

  - RSI < 30  (oversold)
  - RSI > 70  (overbought)

Bars:   how many 1-minute bars closed inside the zone.
Events: how many crossover events occurred. Each excursion into an extreme
zone is counted as ONE crossover, no matter how many bars it lasts: e.g. once
RSI drops below 30, the whole stretch until it crosses back above 30 counts
as a single oversold event. The same applies to stretches above 70.

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


def zone_crossover_events(rsi_df, threshold: float, oversold: bool, label: str):
    """Collect each excursion of RSI into an extreme zone with its timing.

    Returns a list of dicts, one per event: the zone ``label``, the datetime
    of the first bar inside the zone (``start``), the datetime of the bar
    where RSI crossed back out — or the last bar if the excursion never
    closed (``end``) — the close price at each of those bars (``entry_price``
    and ``exit_price``), and the ``duration`` in minutes between the two.
    """
    df = rsi_df.dropna(subset=["rsi"])
    rsi = df["rsi"]
    in_zone = (rsi < threshold) if oversold else (rsi > threshold)

    events = []
    start_ts = start_px = None
    for ts, px, flag in zip(df["datetime"], df["close"], in_zone):
        if flag and start_ts is None:
            start_ts, start_px = ts, px    # entered the zone: start of one event
        elif not flag and start_ts is not None:
            events.append({
                "label": label, "start": start_ts, "end": ts,
                "entry_price": float(start_px), "exit_price": float(px),
            })
            start_ts = start_px = None     # crossed back out: event completed
    if start_ts is not None:               # session ended while still in the zone
        events.append({
            "label": label, "start": start_ts, "end": df["datetime"].iloc[-1],
            "entry_price": float(start_px), "exit_price": float(df["close"].iloc[-1]),
        })

    for event in events:
        event["duration"] = int((event["end"] - event["start"]).total_seconds() // 60)
    return events


CHILD_INDENT = 4      # spaces the child table is indented under its parent row


def format_crossover_events(events, indent: int = CHILD_INDENT) -> str:
    """Render events as a small child table printed under the day's row.

    'Hi' marks an overbought excursion (RSI above the upper threshold),
    'Lo' an oversold one. Each event is one table row showing its entry and
    exit time, duration, and the close price at entry and exit, e.g.:

        Crossovers (Hi=RSI>70, Lo=RSI<30):
          Zone   Entry    Exit     Dur    Entry Px    Exit Px     %Chg
          ------------------------------------------------------------
          Hi      9:40    9:41      1m    23920.10   23925.30    +0.02
    """
    pad = " " * indent
    title = (
        f"{pad}Crossovers (Hi=RSI>{int(RSI_OVERBOUGHT)}, "
        f"Lo=RSI<{int(RSI_OVERSOLD)}):"
    )
    if not events:
        return f"{title} none"

    inner = pad + "  "
    header = (
        f"{inner}{'Zone':<4} {'Entry':>6}  {'Exit':>6}  {'Dur':>5} "
        f"{'Entry Px':>11} {'Exit Px':>10} {'%Chg':>8}"
    )
    lines = [title, header, inner + "-" * (len(header) - len(inner))]
    for event in sorted(events, key=lambda e: e["start"]):
        # f-string formatting instead of strftime('%-H:%M'): the '-' (no-pad)
        # modifier is a glibc extension and raises ValueError on Windows.
        start, end = event["start"], event["end"]
        zone = "Lo" if "<" in event["label"] else "Hi"
        entry_px, exit_px = event["entry_price"], event["exit_price"]
        pct = (exit_px - entry_px) / entry_px * 100
        entry_t = f"{start.hour}:{start.minute:02d}"
        exit_t = f"{end.hour}:{end.minute:02d}"
        dur = f"{event['duration']}m"
        lines.append(
            f"{inner}{zone:<4} {entry_t:>6}  {exit_t:>6}  {dur:>5} "
            f"{entry_px:>11.2f} {exit_px:>10.2f} {pct:>+8.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    service = NiftyOptionService(breeze)

    start = datetime.strptime(START_DATE, "%d-%b-%Y").date()
    end = datetime.strptime(END_DATE, "%d-%b-%Y").date()

    print(
        f"Nifty 50 — 1-minute RSI({RSI_PERIOD}) extremes, "
        f"{START_DATE} to {END_DATE}\n"
    )
    os_label = f"RSI<{int(RSI_OVERSOLD)}"
    ob_label = f"RSI>{int(RSI_OVERBOUGHT)}"
    header = (
        f"{'Date':<12} {'Day':<10} {'Bars':>5} "
        f"{os_label + ' bars':>13} {os_label + ' evts':>13} "
        f"{ob_label + ' bars':>13} {ob_label + ' evts':>13} "
        f"{'Open':>10} {'Close':>10} {'%Chg':>8}"
    )
    print(header)
    print("-" * len(header))

    total_bars = 0
    total_oversold_bars = total_oversold = 0
    total_overbought_bars = total_overbought = 0
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
        oversold_bars = int((rsi < RSI_OVERSOLD).sum())
        overbought_bars = int((rsi > RSI_OVERBOUGHT).sum())
        oversold = count_zone_crossovers(rsi, RSI_OVERSOLD, oversold=True)
        overbought = count_zone_crossovers(rsi, RSI_OVERBOUGHT, oversold=False)
        events = zone_crossover_events(rsi_df, RSI_OVERSOLD, True, os_label)
        events += zone_crossover_events(rsi_df, RSI_OVERBOUGHT, False, ob_label)

        total_bars += bars
        total_oversold_bars += oversold_bars
        total_oversold += oversold
        total_overbought_bars += overbought_bars
        total_overbought += overbought
        days_with_data += 1

        day_open = float(candles[0]["open"])
        day_close = float(candles[-1]["close"])
        pct_change = (day_close - day_open) / day_open * 100

        print(
            f"{day.strftime('%d-%b-%Y'):<12} {day.strftime('%A'):<10} "
            f"{bars:>5} {oversold_bars:>13} {oversold:>13} "
            f"{overbought_bars:>13} {overbought:>13} "
            f"{day_open:>10.2f} {day_close:>10.2f} {pct_change:>+8.2f}"
        )
        # Child table: the day's crossover events, indented under its row.
        print(format_crossover_events(events))
        print()

    print("-" * len(header))
    print(
        f"{'TOTAL':<12} {str(days_with_data) + ' days':<10} "
        f"{total_bars:>5} {total_oversold_bars:>13} {total_oversold:>13} "
        f"{total_overbought_bars:>13} {total_overbought:>13}"
    )


if __name__ == "__main__":
    main()
