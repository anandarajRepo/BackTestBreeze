"""
Nifty 50 Weekly Options — McGinley T3 Flow Campaign Backtest  (1-second data)
=============================================================================

A trend-following strategy built around an adaptive signal trail and a
campaign-style trade management model. It waits for the selected flow engine to
shift direction, opens a long campaign on the option premium, and manages the
trade against a TP1 / TP2 / TP3 target structure.

Signal engine (the adaptive basis), set by ENGINE_MODE:
  • "MCGINLEY" — McGinley Dynamic only (adapts to changes in price speed)
  • "T3"       — Tillson T3 smoothing only (a smoother trend basis)
  • "BLEND"    — average of the McGinley Dynamic and T3 curves (default)

An ATR signal trail is built around the engine basis and locks in a directional
path. A confirmed upward transition of the trail opens a long campaign; a
downward transition (flow flip) closes it. Each option leg (CE & PE) is traded
long on its OWN price's flow signal. Signals are confirmed on closed bars.

Campaign target model, set by TARGET_MODE:
  • "FLOW_TRAIL"   — TP1/TP2/TP3 projected from the distance between price and
                     the active signal trail, so targets expand and contract
                     with the trend structure (default)
  • "ATR_BASELINE" — TP1/TP2/TP3 projected from the ATR at entry

Take-profit exit behaviour, set by TP_EXIT_MODE:
  • "TP1" / "TP2" / "TP3" — exit the FULL position at the selected single target
  • "SCALE_OUT"           — reduce the position across TP1/TP2/TP3 using the
                            TP1_PCT / TP2_PCT settings (remainder exits at TP3)

Other exits:
  - Flow flip (signal trail confirms a downward transition)  → close all
  - Optional campaign stop-loss (SL_MULT × base distance)    → close all
  - Trailing stop-loss (optional, percentage-based)          → close all
  - Square-off at 15:20 IST                                  → close all
  - No new entries before 9:30 or after 14:45
  - Max 5 trades per day per symbol

Usage:
  1. Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  2. Set RESAMPLE_SECONDS to the desired candle size (1, 5, 10, 15, 30, 45, 60…).
  Data is always fetched as 1-second bars from Breeze; resampling is done
  locally before the signal engine and strategy logic run.
"""

import os
from datetime import date

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.mcginley_t3_flow_option_strategy import McGinleyT3FlowOptionStrategy

load_dotenv()

# ── Session ───────────────────────────────────────────────────────────────────

breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN"),
)
print("Session Generated Successfully\n")

# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE        = "01-Jan-2026"   # format: DD-Mon-YYYY
END_DATE          = "09-Jun-2026"   # format: DD-Mon-YYYY

CAPITAL           = 100_000.0       # capital per contract (used for position sizing)

# Signal engine. "MCGINLEY", "T3" or "BLEND". BLEND averages the McGinley
# Dynamic and T3 curves for a balanced engine between responsiveness and
# smoothness.
ENGINE_MODE       = "BLEND"
MCGINLEY_PERIOD   = 14              # McGinley Dynamic lookback period
T3_PERIOD         = 8               # T3 smoothing length
T3_VOLUME_FACTOR  = 0.7             # T3 volume factor (0–1), higher = more responsive

# ATR signal-trail parameters. The trail is built `ATR_MULTIPLIER` × ATR away
# from the engine basis and locks in a directional path.
ATR_PERIOD        = 10
ATR_MULTIPLIER    = 2.0

# Campaign target model. "FLOW_TRAIL" projects the targets from the distance
# between price and the active signal trail (the default, which ties the target
# spacing to the same adaptive trail that produced the signal). "ATR_BASELINE"
# projects them from a standard ATR baseline at entry.
TARGET_MODE       = "FLOW_TRAIL"

# Target multiples of the base distance (price−trail, or ATR) above the entry.
TP1_MULT          = 1.0
TP2_MULT          = 2.0
TP3_MULT          = 3.0

# Take-profit exit behaviour. "TP1" / "TP2" / "TP3" exit the full position at
# the chosen single target. "SCALE_OUT" reduces the position across all three:
# TP1_PCT of the position at TP1, TP2_PCT at TP2, and the remainder at TP3.
TP_EXIT_MODE      = "SCALE_OUT"
TP1_PCT           = 0.40
TP2_PCT           = 0.30

# Optional campaign stop-loss, placed SL_MULT × base distance below the entry.
SL_ENABLED        = True
SL_MULT           = 1.0

# Trailing stop-loss. When enabled, the remaining position is closed if the
# option price falls TRAILING_STOP_PCT percent below the highest price reached
# since entry (the stop ratchets up with the peak, never down).
TRAILING_STOP_ENABLED = True
TRAILING_STOP_PCT     = 5.0

# Always fetch raw 1-second bars from Breeze; resampling is done locally.
INTERVAL          = "1second"

# Candle size (in seconds) used for the strategy. Set to 1 to use raw 1-second
# bars without any resampling.
RESAMPLE_SECONDS  = 60

# Print the final resampled DataFrame (with the signal engine) alongside trades.
PRINT_RESAMPLED   = False

# When True, candle data is served ONLY from the local cache — no Breeze API
# calls are made. Expiries whose data is not already cached are skipped.
CACHE_ONLY        = True

# When True, a fresh ATM strike is chosen for EACH trading day from that day's
# Nifty 9:15 open. When False, a single ATM strike is anchored to the week's
# Monday open and traded across the whole expiry window.
PER_DAY_ATM       = True

# NSE market holidays for 2026. When a Tuesday weekly expiry falls on one of
# these dates, the expiry is rolled back to the previous trading day.
MARKET_HOLIDAYS = {
    date(2026, 1, 26),   # Republic Day            (Monday)
    date(2026, 3, 3),    # Holi                    (Tuesday)
    date(2026, 3, 26),   # Shri Ram Navami         (Thursday)
    date(2026, 3, 31),   # Shri Mahavir Jayanti    (Tuesday)
    date(2026, 4, 3),    # Good Friday             (Friday)
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti    (Tuesday)
    date(2026, 5, 1),    # Maharashtra Day         (Friday)
    date(2026, 5, 28),   # Bakri Id (Eid-al-Adha)  (Thursday)
    date(2026, 6, 26),   # Moharram                (Friday)
    date(2026, 8, 15),   # Independence Day        (Saturday)
    date(2026, 9, 14),   # Ganesh Chaturthi        (Monday)
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti  (Friday)
    date(2026, 11, 8),   # Diwali Laxmi Pujan      (Sunday)
    date(2026, 11, 10),  # Diwali Balipratipada    (Tuesday)
    date(2026, 11, 24),  # Guru Nanak Jayanti      (Tuesday)
    date(2026, 12, 25),  # Christmas               (Friday)
}

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nifty_service = NiftyOptionService(breeze)

    strategy = McGinleyT3FlowOptionStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        engine_mode=ENGINE_MODE,
        mcginley_period=MCGINLEY_PERIOD,
        t3_period=T3_PERIOD,
        t3_volume_factor=T3_VOLUME_FACTOR,
        atr_period=ATR_PERIOD,
        atr_multiplier=ATR_MULTIPLIER,
        target_mode=TARGET_MODE,
        tp1_mult=TP1_MULT,
        tp2_mult=TP2_MULT,
        tp3_mult=TP3_MULT,
        tp_exit_mode=TP_EXIT_MODE,
        tp1_pct=TP1_PCT,
        tp2_pct=TP2_PCT,
        sl_enabled=SL_ENABLED,
        sl_mult=SL_MULT,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        resample_seconds=RESAMPLE_SECONDS,
        print_resampled=PRINT_RESAMPLED,
        cache_only=CACHE_ONLY,
        market_holidays=MARKET_HOLIDAYS,
        per_day_atm=PER_DAY_ATM,
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_pct=TRAILING_STOP_PCT,
    )

    expiry_results = strategy.run_weekly_backtest()
    McGinleyT3FlowOptionStrategy.print_report(expiry_results)
