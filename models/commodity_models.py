from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class CommodityTradeResult:
    symbol: str
    commodity: str
    option_type: str          # "CE" or "PE"
    strike: int
    expiry_date: date
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    exit_reason: str          # "ADX_CROSSOVER" or "SQUARE_OFF"
    adx_at_entry: float
    di_plus_at_entry: float
    di_minus_at_entry: float
    adx_at_exit: float
    di_plus_at_exit: float
    di_minus_at_exit: float
    duration_minutes: int


@dataclass
class MonthlyExpiryResult:
    expiry_date: date
    commodity: str
    atm_strike: int
    commodity_open: float
    ce_trades: list[CommodityTradeResult] = field(default_factory=list)
    pe_trades: list[CommodityTradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[CommodityTradeResult]:
        return self.ce_trades + self.pe_trades


@dataclass
class CommoditySymbolMetrics:
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
