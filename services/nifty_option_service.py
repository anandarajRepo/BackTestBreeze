"""
Nifty option data service — ATM strike selection and candle fetching via Breeze API.
"""

from datetime import date, datetime, timedelta

from breeze_connect import BreezeConnect


class NiftyOptionService:
    STRIKE_INTERVAL = 50

    def __init__(self, breeze: BreezeConnect):
        self.breeze = breeze

    # ── ATM helpers ───────────────────────────────────────────────────────────

    def get_nifty_open(self, trade_date: date) -> float:
        """Return the first 1-minute candle open for Nifty on trade_date."""
        from_dt = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 15, 0)
        to_dt = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 30, 0)

        resp = self.breeze.get_historical_data_v2(
            interval="1minute",
            from_date=from_dt,
            to_date=to_dt,
            stock_code="NIFTY",
            exchange_code="NSE",
            product_type="cash",
        )

        candles = resp.get("Success") or []
        if not candles:
            raise ValueError(f"No Nifty data for {trade_date}: {resp}")

        return float(candles[0]["open"])

    @classmethod
    def atm_strike(cls, nifty_open: float) -> int:
        """Round nifty_open to nearest STRIKE_INTERVAL."""
        return int(round(nifty_open / cls.STRIKE_INTERVAL) * cls.STRIKE_INTERVAL)

    # ── Option candle fetching ────────────────────────────────────────────────

    def get_option_candles(
        self,
        strike: int,
        expiry_date: date,
        option_type: str,
        start: datetime,
        end: datetime,
        interval: str = "1minute",
    ) -> list[dict]:
        """
        Fetch intraday candles for a Nifty option contract.

        option_type: "CE" (call) or "PE" (put)
        """
        right = "call" if option_type == "CE" else "put"
        expiry_str = datetime(
            expiry_date.year, expiry_date.month, expiry_date.day, 6, 0, 0
        ).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        resp = self.breeze.get_historical_data_v2(
            interval=interval,
            from_date=start,
            to_date=end,
            stock_code="NIFTY",
            exchange_code="NFO",
            product_type="options",
            expiry_date=expiry_str,
            right=right,
            strike_price=str(strike),
        )

        candles = resp.get("Success") or []
        return candles

    # ── Weekly expiry calendar ────────────────────────────────────────────────

    @staticmethod
    def weekly_tuesdays(start: date, end: date) -> list[date]:
        """Return all Tuesdays between start and end (inclusive)."""
        tuesdays: list[date] = []
        d = start
        # advance to first Tuesday
        while d.weekday() != 1:  # 1 = Tuesday
            d += timedelta(days=1)
        while d <= end:
            tuesdays.append(d)
            d += timedelta(weeks=1)
        return tuesdays

    @staticmethod
    def monday_of_week(tuesday: date) -> date:
        """Return the Monday immediately before a given Tuesday."""
        return tuesday - timedelta(days=1)

    @staticmethod
    def week_window(tuesday: date) -> tuple[date, date]:
        """
        Trading window for an expiry week: Wednesday of the previous week → Tuesday expiry.
        Previous Wednesday = tuesday - 6 days.
        """
        prev_wednesday = tuesday - timedelta(days=6)
        return prev_wednesday, tuesday
