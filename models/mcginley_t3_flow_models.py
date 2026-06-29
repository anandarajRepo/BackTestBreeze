from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class FlowPartialExit:
    """A single partial (or final) exit leg of a McGinley T3 Flow campaign."""
    time: datetime
    price: float
    shares: int
    pnl: float
    reason: str               # "TP1", "TP2", "TP3", "FLOW_FLIP", "STOP_LOSS",
                              # "TRAILING_STOP", "SQUARE_OFF"


@dataclass
class McGinleyT3FlowTradeResult:
    """
    Result of a single McGinley T3 Flow campaign on one option leg.

    A campaign may be reduced across TP1 / TP2 / TP3 (scale-out mode) or closed
    in a single leg at the selected target / by a flow flip, optional stop or
    square-off. `exit_price` is the share-weighted average exit price across all
    legs, `pnl` is the total realized profit/loss, and `exit_reason` reflects
    the reason for the final (campaign-closing) leg.
    """
    symbol: str
    option_type: str          # "CE" or "PE"
    strike: int
    expiry_date: date
    entry_time: datetime
    exit_time: datetime       # time of the final leg that closed the campaign
    entry_price: float
    exit_price: float         # share-weighted average exit price across all legs
    shares: int               # total shares entered
    pnl: float                # total pnl across all partial + final legs
    exit_reason: str          # reason of the final leg
    engine_at_entry: float    # signal-engine basis value at entry
    trail_at_entry: float     # ATR signal-trail value at entry
    atr_at_entry: float
    tp1_price: float
    tp2_price: float
    tp3_price: float
    sl_price: float
    duration_minutes: int
    partials: list[FlowPartialExit] = field(default_factory=list)


@dataclass
class FlowWeeklyExpiryResult:
    expiry_date: date
    atm_strike: int
    nifty_open: float
    ce_trades: list[McGinleyT3FlowTradeResult] = field(default_factory=list)
    pe_trades: list[McGinleyT3FlowTradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[McGinleyT3FlowTradeResult]:
        return self.ce_trades + self.pe_trades


@dataclass
class FlowSymbolMetrics:
    symbol: str
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
