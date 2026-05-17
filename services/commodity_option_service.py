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
  Futures expiry : 5th of even months (Feb/Apr/Jun/Aug/Oct/Dec), adjusted for
                   weekends/holidays per GOLD_FUTURES_EXPIRY_DATES lookup.
  Option expiry  : official MCX dates per GOLD_OPTION_EXPIRY_DATES lookup;
                   falls back to 27th adjusted for weekends/holidays.
  ATM strike     : computed daily from the active futures contract price

SILVER-specific contract structure:
  Futures expiry : 5th of every month, adjusted for weekends/holidays per
                   SILVER_FUTURES_EXPIRY_DATES lookup.
  Option expiry  : official MCX dates per SILVER_OPTION_EXPIRY_DATES lookup;
                   falls back to 27th adjusted for weekends/holidays.
  ATM strike     : computed daily from the active futures contract price
                   (rounded to nearest ₹500 interval)
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

# SILVER MCX contract expiry days
SILVER_FUTURES_EXPIRY_DAY: int = 5   # futures expire on the 5th of every month
SILVER_OPTION_EXPIRY_DAY:  int = 27  # options  expire on the 27th of each month

# MCX SILVER futures exist for all calendar months
SILVER_FUTURES_CONTRACT_MONTHS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

# Explicit MCX GOLD futures expiry dates for 2026 (source: Groww / MCX official calendar).
# Key: (year, month) — months are even only (Feb=2, Apr=4, Jun=6, Aug=8, Oct=10, Dec=12).
# These are the dates the Breeze API expects as the contract expiry parameter.
GOLD_FUTURES_EXPIRY_DATES: dict[tuple[int, int], date] = {
    (2026, 2):  date(2026, 2,  5),   # February  5, 2026
    (2026, 4):  date(2026, 4,  2),   # April     2, 2026  (Apr 5 is Sunday; Apr 3 Good Friday)
    (2026, 6):  date(2026, 6,  5),   # June      5, 2026
    (2026, 8):  date(2026, 8,  5),   # August    5, 2026
    (2026, 10): date(2026, 10, 5),   # October   5, 2026
    (2026, 12): date(2026, 12, 4),   # December  4, 2026  (Dec 5 is Saturday)
}

# Explicit MCX SILVER futures expiry dates for 2026 as expected by the Breeze API.
# Key: (year, month) — all calendar months.
# These are the 5th-of-month dates the Breeze API uses, adjusted back past weekends/holidays.
SILVER_FUTURES_EXPIRY_DATES: dict[tuple[int, int], date] = {
    (2026, 1):  date(2026, 1,  5),   # January   5, 2026 (Monday)
    (2026, 2):  date(2026, 2,  5),   # February  5, 2026 (Thursday)
    (2026, 3):  date(2026, 3,  5),   # March     5, 2026 (Thursday)
    (2026, 4):  date(2026, 4,  2),   # April     5 is Sunday; Apr 3 Good Friday → Apr 2
    (2026, 5):  date(2026, 5,  5),   # May       5, 2026 (Tuesday)
    (2026, 6):  date(2026, 6,  5),   # June      5, 2026 (Friday)
    (2026, 7):  date(2026, 7,  3),   # July      5 is Sunday; Jul 4 Saturday → Jul 3
    (2026, 8):  date(2026, 8,  5),   # August    5, 2026 (Wednesday)
    (2026, 9):  date(2026, 9,  4),   # September 5 is Saturday → Sep 4
    (2026, 10): date(2026, 10, 5),   # October   5, 2026 (Monday)
    (2026, 11): date(2026, 11, 5),   # November  5, 2026 (Thursday)
    (2026, 12): date(2026, 12, 4),   # December  5 is Saturday → Dec 4
}

# Explicit MCX SILVER option expiry dates for 2025-2026 (source: Groww / MCX official calendar).
# Key: (year, month) — every calendar month.
SILVER_OPTION_EXPIRY_DATES: dict[tuple[int, int], date] = {
    (2025, 12): date(2025, 12, 31),  # December 31, 2025
    (2026, 1):  date(2026, 1,  27),  # January  27, 2026
    (2026, 2):  date(2026, 2,  24),  # February 24, 2026
    (2026, 3):  date(2026, 3,  26),  # March    26, 2026
    (2026, 4):  date(2026, 4,  24),  # April    24, 2026
    (2026, 5):  date(2026, 5,  26),  # May      26, 2026
    (2026, 6):  date(2026, 6,  25),  # June     25, 2026
    (2026, 7):  date(2026, 7,  28),  # July     28, 2026
    (2026, 8):  date(2026, 8,  26),  # August   26, 2026
    (2026, 9):  date(2026, 9,  24),  # September 24, 2026
    (2026, 10): date(2026, 10, 27),  # October  27, 2026
    (2026, 11): date(2026, 11, 24),  # November 24, 2026
    (2026, 12): date(2026, 12, 24),  # December 24, 2026
}

# Explicit MCX GOLD option expiry dates for 2025-2026 (source: Groww / MCX official calendar).
# Key: (year, month) — every calendar month.
GOLD_OPTION_EXPIRY_DATES: dict[tuple[int, int], date] = {
    (2025, 12): date(2025, 12, 31),  # December 31, 2025
    (2026, 1):  date(2026, 1,  27),  # January  27, 2026
    (2026, 2):  date(2026, 2,  27),  # February 27, 2026
    (2026, 3):  date(2026, 3,  24),  # March    24, 2026
    (2026, 4):  date(2026, 4,  30),  # April    30, 2026
    (2026, 5):  date(2026, 5,  27),  # May      27, 2026
    (2026, 6):  date(2026, 6,  30),  # June     30, 2026
    (2026, 7):  date(2026, 7,  27),  # July     27, 2026
    (2026, 8):  date(2026, 8,  31),  # August   31, 2026
    (2026, 9):  date(2026, 9,  23),  # September 23, 2026
    (2026, 10): date(2026, 10, 30),  # October  30, 2026
    (2026, 11): date(2026, 11, 25),  # November 25, 2026
}

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
        return d.weekday() not in (5, 6) and d not in MCX_HOLIDAYS

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
            product_type="options",
            expiry_date=expiry_str,
            right=right,
            strike_price=str(strike),
        )

        return resp.get("Success") or []

    # ── Monthly expiry calendar ───────────────────────────────────────────────

    @staticmethod
    def last_thursday(year: int, month: int) -> date:
        """Return the last Thursday of the given month, adjusted back if it falls on a holiday."""
        last_day = monthrange(year, month)[1]
        d = date(year, month, last_day)
        # weekday(): Monday=0, Thursday=3
        offset = (d.weekday() - 3) % 7
        d = d - timedelta(days=offset)
        # Roll back past any MCX holidays (weekends already excluded by Thursday logic)
        while d in MCX_HOLIDAYS:
            d -= timedelta(days=1)
        return d

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
        """
        Return the MCX GOLD futures expiry date for the given contract month.
        Uses the explicit 2026 lookup table when available; otherwise falls back
        to the generic 5th-of-month rule adjusted for weekends/holidays.
        """
        if (year, month) in GOLD_FUTURES_EXPIRY_DATES:
            return GOLD_FUTURES_EXPIRY_DATES[(year, month)]
        return date(year, month, GOLD_FUTURES_EXPIRY_DAY)

    @staticmethod
    def _last_trading_day(nominal: date) -> date:
        """
        Return the last trading day for a contract with the given nominal expiry.
        Rolls back past weekends and MCX holidays.
        """
        d = nominal
        while d.weekday() == 5 or d.weekday() == 6 or d in MCX_HOLIDAYS:
            d -= timedelta(days=1)
        return d

    @staticmethod
    def gold_futures_expiry(trade_date: date) -> date:
        """
        Return the MCX GOLD futures expiry date for the active contract on trade_date.
        For months covered by GOLD_FUTURES_EXPIRY_DATES the exact official date is used;
        for all other months the expiry is the 5th adjusted for weekends/holidays.
        The contract is considered active as long as its expiry date >= trade_date.
        """
        year, month = trade_date.year, trade_date.month
        for _ in range(13):
            if month in GOLD_FUTURES_CONTRACT_MONTHS:
                expiry = CommodityOptionService._nominal_futures_expiry(year, month)
                if expiry >= trade_date:
                    return expiry
            month += 1
            if month > 12:
                month = 1
                year += 1
        raise ValueError(f"Could not find GOLD futures expiry for {trade_date}")

    @staticmethod
    def _adjust_option_expiry(nominal: date) -> date:
        """
        Return the effective option expiry for the given (year, month).
        Uses GOLD_OPTION_EXPIRY_DATES when available; otherwise rolls the 27th
        back past weekends and MCX holidays.
        """
        key = (nominal.year, nominal.month)
        if key in GOLD_OPTION_EXPIRY_DATES:
            return GOLD_OPTION_EXPIRY_DATES[key]
        d = date(nominal.year, nominal.month, GOLD_OPTION_EXPIRY_DAY)
        while d.weekday() == 5 or d.weekday() == 6 or d in MCX_HOLIDAYS:
            d -= timedelta(days=1)
        return d

    @staticmethod
    def gold_option_expiry(trade_date: date) -> date:
        """
        Return the active GOLD option expiry for trade_date.
        Uses the official MCX dates from GOLD_OPTION_EXPIRY_DATES when available;
        falls back to 27th-of-month adjusted for weekends/holidays otherwise.
        Rolls forward to the next month if trade_date is past the current expiry.
        """
        candidate = CommodityOptionService._adjust_option_expiry(
            date(trade_date.year, trade_date.month, GOLD_OPTION_EXPIRY_DAY)
        )
        if trade_date > candidate:
            if trade_date.month == 12:
                next_nominal = date(trade_date.year + 1, 1, GOLD_OPTION_EXPIRY_DAY)
            else:
                next_nominal = date(trade_date.year, trade_date.month + 1, GOLD_OPTION_EXPIRY_DAY)
            candidate = CommodityOptionService._adjust_option_expiry(next_nominal)
        return candidate

    @classmethod
    def gold_option_expiries(cls, start: date, end: date) -> list[date]:
        """
        Return all GOLD option expiries between start and end inclusive.
        Uses official MCX dates from GOLD_OPTION_EXPIRY_DATES when available;
        falls back to 27th-of-month adjusted for weekends/holidays otherwise.
        """
        expiries: list[date] = []
        year, month = start.year, start.month
        while True:
            nominal = date(year, month, GOLD_OPTION_EXPIRY_DAY)
            if nominal > end:
                break
            expiry = cls._adjust_option_expiry(nominal)
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

    # ── SILVER-specific contract helpers ──────────────────────────────────────

    @staticmethod
    def _nominal_silver_futures_expiry(year: int, month: int) -> date:
        """
        Return the MCX SILVER futures expiry date for the given contract month.
        Uses the explicit lookup table when available; otherwise falls back to
        the generic 5th-of-month rule adjusted for weekends/holidays.
        """
        if (year, month) in SILVER_FUTURES_EXPIRY_DATES:
            return SILVER_FUTURES_EXPIRY_DATES[(year, month)]
        return date(year, month, SILVER_FUTURES_EXPIRY_DAY)

    @staticmethod
    def silver_futures_expiry(trade_date: date) -> date:
        """
        Return the MCX SILVER futures expiry date for the active contract on trade_date.
        SILVER futures trade every calendar month (unlike GOLD which is even months only).
        The contract is considered active as long as its expiry date >= trade_date.
        """
        year, month = trade_date.year, trade_date.month
        for _ in range(13):
            expiry = CommodityOptionService._nominal_silver_futures_expiry(year, month)
            if expiry >= trade_date:
                return expiry
            month += 1
            if month > 12:
                month = 1
                year += 1
        raise ValueError(f"Could not find SILVER futures expiry for {trade_date}")

    @staticmethod
    def _adjust_silver_option_expiry(nominal: date) -> date:
        """
        Return the effective SILVER option expiry for the given (year, month).
        Uses SILVER_OPTION_EXPIRY_DATES when available; otherwise rolls the 27th
        back past weekends and MCX holidays.
        """
        key = (nominal.year, nominal.month)
        if key in SILVER_OPTION_EXPIRY_DATES:
            return SILVER_OPTION_EXPIRY_DATES[key]
        d = date(nominal.year, nominal.month, SILVER_OPTION_EXPIRY_DAY)
        while d.weekday() == 5 or d.weekday() == 6 or d in MCX_HOLIDAYS:
            d -= timedelta(days=1)
        return d

    @staticmethod
    def silver_option_expiry(trade_date: date) -> date:
        """
        Return the active SILVER option expiry for trade_date.
        Uses official MCX dates from SILVER_OPTION_EXPIRY_DATES when available;
        falls back to 27th-of-month adjusted for weekends/holidays otherwise.
        Rolls forward to the next month if trade_date is past the current expiry.
        """
        candidate = CommodityOptionService._adjust_silver_option_expiry(
            date(trade_date.year, trade_date.month, SILVER_OPTION_EXPIRY_DAY)
        )
        if trade_date > candidate:
            if trade_date.month == 12:
                next_nominal = date(trade_date.year + 1, 1, SILVER_OPTION_EXPIRY_DAY)
            else:
                next_nominal = date(trade_date.year, trade_date.month + 1, SILVER_OPTION_EXPIRY_DAY)
            candidate = CommodityOptionService._adjust_silver_option_expiry(next_nominal)
        return candidate

    @classmethod
    def silver_option_expiries(cls, start: date, end: date) -> list[date]:
        """
        Return all SILVER option expiries between start and end inclusive.
        Uses official MCX dates from SILVER_OPTION_EXPIRY_DATES when available;
        falls back to 27th-of-month adjusted for weekends/holidays otherwise.
        """
        expiries: list[date] = []
        year, month = start.year, start.month
        while True:
            nominal = date(year, month, SILVER_OPTION_EXPIRY_DAY)
            if nominal > end:
                break
            expiry = cls._adjust_silver_option_expiry(nominal)
            if expiry >= start:
                expiries.append(expiry)
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
        return expiries

    @classmethod
    def silver_option_window(cls, option_expiry: date) -> tuple[date, date]:
        """
        Trading window for a SILVER option expiry:
        from the 28th of the previous month through the option expiry date.
        """
        if option_expiry.month == 1:
            win_start = date(option_expiry.year - 1, 12, 28)
        else:
            win_start = date(option_expiry.year, option_expiry.month - 1, 28)
        return win_start, option_expiry

    def get_silver_futures_price(self, trade_date: date) -> float:
        """
        Return the opening SILVER futures price on trade_date using the active
        futures contract (expiry = 5th of the appropriate month).
        """
        futures_expiry = self.silver_futures_expiry(trade_date)
        return self.get_commodity_open("SILVER", trade_date, expiry_date=futures_expiry)

    def get_silver_daily_atm(self, trade_date: date) -> tuple[float, int]:
        """
        Fetch SILVER futures price for trade_date and return (futures_price, atm_strike).
        ATM is rounded to the nearest ₹500 interval.
        """
        _, strike_interval = COMMODITY_CONFIG["SILVER"]
        price = self.get_silver_futures_price(trade_date)
        strike = self.atm_strike(price, strike_interval)
        return price, strike
