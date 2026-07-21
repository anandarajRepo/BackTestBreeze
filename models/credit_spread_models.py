from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class CreditSpreadTradeResult:
    """
    Result of a single credit (net-credit) vertical-spread position.

    A position is always a two-leg defined-risk credit spread:
      • BULL_PUT  → Bull Put Spread  : sell OTM PE + buy further-OTM PE (below)
      • BEAR_CALL → Bear Call Spread : sell OTM CE + buy further-OTM CE (above)

    ``net_credit`` is the per-unit premium collected at entry
    (sell_entry − buy_entry). ``exit_value`` is the per-unit cost to close the
    spread at exit (sell_exit − buy_exit). Because the position is opened for a
    credit, the per-unit P&L is (net_credit − exit_value); ``pnl`` is that move
    scaled by the whole position (lot_size × lots).

    Max profit  = net_credit                        (both legs expire worthless)
    Max risk    = spread_width − net_credit          (spot beyond the long strike)
    Breakeven   = short_strike − net_credit  (bull put)
                = short_strike + net_credit  (bear call)
    """
    direction: str            # "BULL_PUT" or "BEAR_CALL"
    spread_type: str          # "BULL_PUT_SPREAD" or "BEAR_CALL_SPREAD"
    option_type: str          # "PE" (bull put) or "CE" (bear call)
    expiry_date: date
    sell_strike: int          # short (higher-premium) leg
    buy_strike: int           # long  (lower-premium, protective) leg
    entry_time: datetime
    exit_time: datetime
    sell_entry: float         # premium received for the short leg
    buy_entry: float          # premium paid for the long leg
    sell_exit: float          # short-leg premium at exit
    buy_exit: float           # long-leg premium at exit
    net_credit: float         # sell_entry − buy_entry (per unit, collected)
    exit_value: float         # sell_exit − buy_exit    (per unit, cost to close)
    spread_width: int         # |sell_strike − buy_strike| (max risk reference)
    quantity: int             # total units (lot_size × lots)
    pnl: float
    exit_reason: str          # "PROFIT_TARGET" | "STOP_LOSS" | "BREAKEVEN_BREACH" | "SQUARE_OFF"
    breakeven: float          # short_strike ∓ net_credit
    spot_at_entry: float      # Nifty spot when the spread was opened
    spot_at_exit: float       # Nifty spot when the spread was closed
    duration_minutes: int


@dataclass
class CreditSpreadDayResult:
    trade_date: date
    expiry_date: date
    atm_strike: int
    nifty_open: float
    trades: list[CreditSpreadTradeResult] = field(default_factory=list)


@dataclass
class CreditSpreadMetrics:
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
