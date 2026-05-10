"""
MCX commodity option data service — ATM strike selection, candle fetching,
and monthly expiry calendar via Breeze API.

Supported commodities and their MCX stock codes / strike intervals:
  Gold        → GOLD        (₹100 strike interval)
  Silver      → SILVER      (₹500 strike interval)
  Crude Oil   → CRUDEOIL    (₹50  strike interval)
  Natural Gas → NATURALGAS  (₹10  strike interval)

Monthly expiry on MCX falls on the last Thursday of each month.
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


class CommodityOptionService:
    def __init__(self, breeze: BreezeConnect):
        self.breeze = breeze

    # ── ATM helpers ───────────────────────────────────────────────────────────

    def get_commodity_open(self, stock_code: str, trade_date: date) -> float:
        """Return the first 1-minute candle open for a commodity on trade_date."""
        exchange_code, _ = COMMODITY_CONFIG[stock_code]
        from_dt = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 0, 0)
        to_dt   = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 30, 0)

        resp = self.breeze.get_historical_data_v2(
            interval="1minute",
            from_date=from_dt,
            to_date=to_dt,
            stock_code=stock_code,
            exchange_code=exchange_code,
            product_type="futures",
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
