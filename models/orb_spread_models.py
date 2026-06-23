from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class ORBSpreadTradeResult:
    """
    Result of a single Opening-Range-Breakout vertical-spread position.

    A position is always a two-leg debit spread:
      • BULL  → Bull Call Spread : buy ATM CE  + sell higher-strike CE
      • BEAR  → Bear Put  Spread : buy ATM PE  + sell lower-strike  PE

    ``net_debit`` is the per-unit combined premium paid at entry
    (buy_entry − sell_entry). ``exit_value`` is the per-unit combined premium
    at exit (buy_exit − sell_exit). ``pnl`` is the total realized profit/loss
    across the whole position (per-unit move × lot_size × lots).
    """
    direction: str            # "BULL" or "BEAR"
    spread_type: str          # "BULL_CALL_SPREAD" or "BEAR_PUT_SPREAD"
    expiry_date: date
    buy_strike: int
    sell_strike: int
    entry_time: datetime
    exit_time: datetime
    buy_entry: float          # premium paid for the long leg
    sell_entry: float         # premium received for the short leg
    buy_exit: float           # long-leg premium at exit
    sell_exit: float          # short-leg premium at exit
    net_debit: float          # buy_entry − sell_entry (per unit)
    exit_value: float         # buy_exit − sell_exit  (per unit)
    quantity: int             # total units (lot_size × lots)
    pnl: float
    exit_reason: str          # "PROFIT_TARGET" | "STOP_LOSS" | "SQUARE_OFF"
    or_high: float
    or_low: float
    breakout_price: float     # Nifty spot at the breakout
    duration_minutes: int


@dataclass
class ORBSpreadDayResult:
    trade_date: date
    expiry_date: date
    atm_strike: int
    nifty_open: float
    trades: list[ORBSpreadTradeResult] = field(default_factory=list)


@dataclass
class ORBSpreadMetrics:
    label: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    profit_factor: float
    best_trade: float
    worst_trade: float
    avg_duration_minutes: float
    max_consecutive_losses: int
