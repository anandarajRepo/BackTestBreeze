"""
Nifty 50 Weekly Options — Real Strength Histogram Backtest
===========================================================

Strategy Overview
-----------------
A momentum-continuation strategy using the Real Strength histogram, a composite
oscillator that multiplies three independent components:

  Real Strength = EMA( ROC × max(vol / vol_MA, floor) × max(ADX / 20, floor) )

    - ROC  : Rate of Change of price — provides the sign (direction)
    - vol/vol_MA : Volume amplifier — requires above-average participation
    - ADX/20     : Trend-strength amplifier — requires confirmed trend

A high reading requires all three components to align simultaneously.

Entry (Long → CE):
  - Histogram > strength_threshold
  - Histogram rising vs previous bar (no entries on a fading peak)
  - ADX >= min_adx
  - DI+ > DI-
  - Volume ratio >= vol_ratio_min
  - Optional: SMA(fast) > SMA(slow)  [toggleable via USE_SMA_FILTER]

Entry (Short → PE): mirror conditions on bearish side.

Exit — Regime-dependent:
  - Hard stop loss (STOP_LOSS_PCT %) always overrides all exits
  - Minimum hold of MIN_BARS_HOLD bars before peak/flip exits fire
  - While SMA still confirms position:
      Exit when histogram crosses into the opposite zone past ±FLIP_THRESHOLD
  - After SMA reverses against position:
      Exit when histogram drops PEAK_DROP_PCT% from its in-trade peak
  - Square-off at 15:20 IST

Re-entry lock: after a stop loss, no new entry in the same direction until
the histogram returns through zero.

Only one position (CE or PE) is held at a time per expiry week.
Signals come from Nifty spot 5-minute candles; option prices are used
exclusively for entry/exit pricing.

Usage
-----
  Set START_DATE / END_DATE to the desired backtest window (DD-Mon-YYYY).
  Each Tuesday in that range is treated as a Nifty weekly expiry.
  The ATM strike is computed from Monday's Nifty opening price.
  The trade window for each expiry is Wednesday (prior week) → Tuesday.
"""

import os

from breeze_connect import BreezeConnect
from dotenv import load_dotenv

from services.nifty_option_service import NiftyOptionService
from strategy.real_strength_option_strategy import RealStrengthOptionStrategy

load_dotenv()

# ── Session ───────────────────────────────────────────────────────────────────

breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN"),
)
print("Session Generated Successfully\n")

# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE  = "01-Jan-2026"   # format: DD-Mon-YYYY
END_DATE    = "05-May-2026"   # format: DD-Mon-YYYY

CAPITAL     = 100_000.0       # capital per contract

# Signal generation (computed on Nifty spot candles)
INTERVAL        = "5minute"   # candle interval for both spot and option data
ADX_PERIOD      = 14          # lookback for ADX / DI calculation
ROC_PERIOD      = 10          # Rate-of-Change lookback (bars)
VOL_MA_PERIOD   = 20          # volume moving-average lookback (bars)
SMOOTH_PERIOD   = 3           # EMA smoothing for the Raw Strength histogram

# Entry filters
STRENGTH_THRESHOLD = 1.0      # minimum histogram magnitude to enter
MIN_ADX            = 14.0     # minimum ADX required at entry
VOL_RATIO_MIN      = 1.2      # minimum volume / vol_MA ratio at entry
SMA_FAST           = 30       # fast SMA period (bars)
SMA_SLOW           = 60       # slow SMA period (bars)
USE_SMA_FILTER     = True     # require SMA cross to confirm direction

# Risk & exit parameters
STOP_LOSS_PCT  = 1.0          # hard stop: % decline in option price from entry
PEAK_DROP_PCT  = 25.0         # peak-drop exit: % fall from trade's histogram peak
FLIP_THRESHOLD = 0.8          # flip exit: histogram must cross ±this level
MIN_BARS_HOLD  = 3            # bars before peak/flip exits are allowed to fire

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nifty_service = NiftyOptionService(breeze)

    strategy = RealStrengthOptionStrategy(
        nifty_service=nifty_service,
        capital=CAPITAL,
        adx_period=ADX_PERIOD,
        roc_period=ROC_PERIOD,
        vol_ma_period=VOL_MA_PERIOD,
        smooth_period=SMOOTH_PERIOD,
        strength_threshold=STRENGTH_THRESHOLD,
        min_adx=MIN_ADX,
        vol_ratio_min=VOL_RATIO_MIN,
        sma_fast=SMA_FAST,
        sma_slow=SMA_SLOW,
        use_sma_filter=USE_SMA_FILTER,
        stop_loss_pct=STOP_LOSS_PCT,
        peak_drop_pct=PEAK_DROP_PCT,
        flip_threshold=FLIP_THRESHOLD,
        min_bars_hold=MIN_BARS_HOLD,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
    )

    expiry_results = strategy.run_weekly_backtest()
    RealStrengthOptionStrategy.print_report(expiry_results)
