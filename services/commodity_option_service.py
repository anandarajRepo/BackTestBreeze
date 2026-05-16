"""
MCX commodity option data service — ATM strike selection, candle fetching,
and monthly expiry calendar via Breeze API.

Supported commodities and their MCX stock codes / strike intervals:
  Gold        → GOLD        (₹100 strike interval)
  Silver      → SILVER      (₹500 strike interval)
  Crude Oil   → CRUDEOIL    (₹50  strike interval)
  Natural Gas → NATURALGAS  (₹10  strike interval)

Monthly expiry on MCX falls on the last Thursday of each month.

GOLD-specific contract structure:
  Futures expiry : 5th of even months only (Feb, Apr, Jun, Aug, Oct, Dec)
  Option expiry  : 27th of every month
  ATM strike     : computed daily from the active futures contract price
"""

from calendar import monthrange
from datetime import date, datetime, timedelta

from breeze_connect import BreezeConnect


# Commodity config: stock_code → (exchange_code, strike_interval)
COMMODITY_CONFIG: dict[str, tuple[str, int]] = {
    "GOLD":        ("MCX", 100),
    "SILVER":      ("MCX", 500),
    "CRUDEOIL":    ("MCX", 50),
    "NATURALGAS":  ("MCX", 10),
}

# GOLD MCX contract expiry days
GOLD_FUTURES_EXPIRY_DAY: int = 5   # futures expire on the 5th of the contract month
GOLD_OPTION_EXPIRY_DAY:  int = 27  # options  expire on the 27th of each month

# MCX GOLD futures only exist for even months (Feb, Apr, Jun, Aug, Oct, Dec)
GOLD_FUTURES_CONTRACT_MONTHS: tuple[int, ...] = (2, 4, 6, 8, 10, 12)

# MCX exchange holidays (market closed; no trading in any commodity)
MCX_HOLIDAYS: frozenset[date] = frozenset({
    # 2025
    date(2025, 1, 26),   # Republic Day
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 4, 10),   # Ram Navami
    date(2025, 4, 14),   # Dr. Ambedkar Jayanti / Good Friday
    date(2025, 4, 18),   # Good Friday (MCX)
    date(2025, 5, 12),   # Buddha Purnima
    date(2025, 6, 7),    # Eid ul-Adha (Id-ul-Zuha)
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 27),   # Ganesh Chaturthi
    date(2025, 10, 2),   # Gandhi Jayanti / Mahatma Gandhi Jayanti
    date(2025, 10, 2),   # Dussehra (if same date)
    date(2025, 10, 20),  # Diwali / Laxmi Puja (Muhurat trading day — full holiday)
    date(2025, 10, 21),  # Diwali (Balipratipada)
    date(2025, 11, 5),   # Gurunanak Jayanti
    date(2025, 12, 25),  # Christmas
    # 2026
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 3),    # Mahashivratri (Feb 3, 2026)
    date(2026, 2, 4),    # MCX holiday (Mahashivratri observed / bridge day)
    date(2026, 3, 3),    # Holi
    date(2026, 3, 30),   # Ram Navami
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 4, 30),   # Buddha Purnima
    date(2026, 8, 15),   # Independence Day
    date(2026, 9, 19),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 10, 19),  # Dussehra
    date(2026, 11, 8),   # Diwali / Laxmi Puja
    date(2026, 11, 9),   # Balipratipada
    date(2026, 11, 24),  # Gurunanak Jayanti
    date(2026, 12, 25),  # Christmas
})


class CommodityOptionService:
    def __init__(self, breeze: BreezeConnect):
        self.breeze = breeze

    @staticmethod
    def is_mcx_trading_day(d: date) -> bool:
        """Return True if MCX trades on this date (not a weekend or known holiday)."""
        return d.weekday() != 6 and d not in MCX_HOLIDAYS

    # ── ATM helpers ───────────────────────────────────────────────────────────

    def get_commodity_open(
        self, stock_code: str, trade_date: date, expiry_date: date | None = None
    ) -> float:
        """Return the first 1-minute candle open for a commodity futures on trade_date.

        expiry_date: the futures contract expiry; required by the Breeze API for MCX.
        When omitted, defaults to the last Thursday of trade_date's month.
        """
        exchange_code, _ = COMMODITY_CONFIG[stock_code]
        from_dt = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 0, 0)
        to_dt   = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 30, 0)

        if expiry_date is None:
            expiry_date = self.last_thursday(trade_date.year, trade_date.month)

        expiry_str = datetime(
            expiry_date.year, expiry_date.month, expiry_date.day, 6, 0, 0
        ).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        resp = self.breeze.get_historical_data_v2(
            interval="1minute",
            from_date=from_dt,
            to_date=to_dt,
            stock_code=stock_code,
            exchange_code=exchange_code,
            product_type="futures",
            expiry_date=expiry_str,
        )

        candles = resp.get("Success") or []
        if not candles:
            raise ValueError(f"No data for {stock_code} on {trade_date}: {resp}")

        return float(candles[0]["open"])

    @staticmethod
    def atm_strike(price: float, strike_interval: int) -> int:
        """Round price to the nearest strike_interval."""
        return int(round(price / strike_interval) * strike_interval)

    # ── Option candle fetching ────────────────────────────────────────────────

    def get_option_candles(
        self,
        stock_code: str,
        strike: int,
        expiry_date: date,
        option_type: str,
        start: datetime,
        end: datetime,
        interval: str = "1minute",
    ) -> list[dict]:
        """
        Fetch intraday candles for an MCX commodity option contract.

        option_type: "CE" (call) or "PE" (put)
        """
        exchange_code, _ = COMMODITY_CONFIG[stock_code]
        right = "call" if option_type == "CE" else "put"
        expiry_str = datetime(
            expiry_date.year, expiry_date.month, expiry_date.day, 6, 0, 0
        ).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        resp = self.breeze.get_historical_data_v2(
            interval=interval,
            from_date=start,
            to_date=end,
            stock_code=stock_code,
            exchange_code=exchange_code,
            product_type="WeeklyOptions",
            expiry_date=expiry_str,
            right=right,
            strike_price=str(strike),
        )

        return resp.get("Success") or []

    # ── Monthly expiry calendar ───────────────────────────────────────────────

    @staticmethod
    def last_thursday(year: int, month: int) -> date:
        """Return the last Thursday of the given month."""
        last_day = monthrange(year, month)[1]
        d = date(year, month, last_day)
        # weekday(): Monday=0, Thursday=3
        offset = (d.weekday() - 3) % 7
        return d - timedelta(days=offset)

    @classmethod
    def monthly_expiries(cls, start: date, end: date) -> list[date]:
        """
        Return all MCX monthly expiries (last Thursday of each month)
        between start and end inclusive.
        """
        expiries: list[date] = []
        year, month = start.year, start.month
        while True:
            expiry = cls.last_thursday(year, month)
            if expiry > end:
                break
            if expiry >= start:
                expiries.append(expiry)
            # advance to next month
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
        return expiries

    @classmethod
    def month_window(cls, expiry: date) -> tuple[date, date]:
        """
        Trading window for a monthly expiry:
        from the 1st of the expiry month through the expiry day.
        """
        month_start = date(expiry.year, expiry.month, 1)
        return month_start, expiry

    # ── GOLD-specific contract helpers ────────────────────────────────────────

    @staticmethod
    def _nominal_futures_expiry(year: int, month: int) -> date:
        """Return the nominal MCX GOLD futures expiry date (always the 5th)."""
        return date(year, month, GOLD_FUTURES_EXPIRY_DAY)

    @staticmethod
    def _last_trading_day(nominal: date) -> date:
        """
        Return the last trading day for a contract with the given nominal expiry.
        If the 5th falls on Saturday or Sunday, the last trading day is the preceding Friday.
        """
        if nominal.weekday() == 5:   # Saturday → Friday
            return nominal - timedelta(days=1)
        if nominal.weekday() == 6:   # Sunday → Friday
            return nominal - timedelta(days=2)
        return nominal

    @staticmethod
    def gold_futures_expiry(trade_date: date) -> date:
        """
        Return the nominal MCX GOLD futures expiry date for the active contract on
        trade_date. The contract is considered expired once its last trading day has
        passed (weekends shift the last trading day to the preceding Friday, but the
        nominal contract date stays on the 5th and is what the Breeze API expects).
        """
        year, month = trade_date.year, trade_date.month
        for _ in range(13):
            if month in GOLD_FUTURES_CONTRACT_MONTHS:
                nominal = CommodityOptionService._nominal_futures_expiry(year, month)
                last_trading = CommodityOptionService._last_trading_day(nominal)
                if last_trading >= trade_date:
                    return nominal   # always return the nominal 5th for the API
            month += 1
            if month > 12:
                month = 1
                year += 1
        raise ValueError(f"Could not find GOLD futures expiry for {trade_date}")

    @staticmethod
    def gold_option_expiry(trade_date: date) -> date:
        """
        Return the active GOLD option expiry (27th of month) for trade_date.
        If trade_date is after the 27th, rolls to the next month's 27th.
        """
        candidate = date(trade_date.year, trade_date.month, GOLD_OPTION_EXPIRY_DAY)
        if trade_date > candidate:
            if trade_date.month == 12:
                candidate = date(trade_date.year + 1, 1, GOLD_OPTION_EXPIRY_DAY)
            else:
                candidate = date(trade_date.year, trade_date.month + 1, GOLD_OPTION_EXPIRY_DAY)
        return candidate

    @classmethod
    def gold_option_expiries(cls, start: date, end: date) -> list[date]:
        """
        Return all GOLD option expiries (27th of each month) between start and end inclusive.
        """
        expiries: list[date] = []
        year, month = start.year, start.month
        while True:
            expiry = date(year, month, GOLD_OPTION_EXPIRY_DAY)
            if expiry > end:
                break
            if expiry >= start:
                expiries.append(expiry)
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
        return expiries

    @classmethod
    def gold_option_window(cls, option_expiry: date) -> tuple[date, date]:
        """
        Trading window for a GOLD option expiry:
        from the 28th of the previous month through the 27th (option_expiry).
        """
        if option_expiry.month == 1:
            win_start = date(option_expiry.year - 1, 12, 28)
        else:
            win_start = date(option_expiry.year, option_expiry.month - 1, 28)
        return win_start, option_expiry

    def get_gold_futures_price(self, trade_date: date) -> float:
        """
        Return the opening GOLD futures price on trade_date using the active
        futures contract (expiry = 5th of the appropriate month).
        """
        futures_expiry = self.gold_futures_expiry(trade_date)
        return self.get_commodity_open("GOLD", trade_date, expiry_date=futures_expiry)

    def get_gold_daily_atm(self, trade_date: date) -> tuple[float, int]:
        """
        Fetch GOLD futures price for trade_date and return (futures_price, atm_strike).
        ATM is rounded to the nearest ₹100 interval.
        """
        _, strike_interval = COMMODITY_CONFIG["GOLD"]
        price = self.get_gold_futures_price(trade_date)
        strike = self.atm_strike(price, strike_interval)
        return price, strike
