"""
Nifty option data service — ATM strike selection and candle fetching via Breeze API.
"""

from datetime import date, datetime, time, timedelta

from breeze_connect import BreezeConnect

from services.candle_cache import CandleCache


class NiftyOptionService:
    STRIKE_INTERVAL = 50

    # Breeze's get_historical_data_v2 returns at most ~1000 records per call and,
    # when more exist, hands back only the tail of the window. To fetch a full
    # window for fine-grained intervals we must page the request into chunks that
    # comfortably stay under that cap.
    #
    # Approx rows-per-second for each interval, used to size each fetch chunk so a
    # single call returns well under ~1000 rows.
    _INTERVAL_SECONDS = {
        "1second": 1,
        "1minute": 60,
        "5minute": 300,
        "30minute": 1800,
        "1day": 86400,
    }
    # Keep each request to at most this many rows (margin under the ~1000 cap).
    _MAX_ROWS_PER_REQUEST = 800
    # Regular Nifty market hours (IST). We only request data within this window.
    _MARKET_OPEN = time(9, 15)
    _MARKET_CLOSE = time(15, 30)

    def __init__(self, breeze: BreezeConnect, cache: CandleCache | None = None):
        self.breeze = breeze
        # File-based cache so the same (symbol, interval, date range) is fetched
        # from the Breeze API only once and reused on later backtest runs.
        self.cache = cache if cache is not None else CandleCache()

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

    @classmethod
    def _chunk_windows(
        cls, start: datetime, end: datetime, interval: str
    ) -> list[tuple[datetime, datetime]]:
        """
        Split [start, end] into market-hours sub-windows, each sized to return at
        most ``_MAX_ROWS_PER_REQUEST`` rows for the given interval.

        This works around Breeze's per-request row cap (~1000 records), which
        otherwise silently truncates a long/fine-grained request to just the tail
        of the window.
        """
        secs = cls._INTERVAL_SECONDS.get(interval, 60)
        chunk = timedelta(seconds=secs * cls._MAX_ROWS_PER_REQUEST)

        windows: list[tuple[datetime, datetime]] = []
        day = start.date()
        last_day = end.date()
        while day <= last_day:
            # Clamp each day to regular market hours and to the overall request range.
            day_open = datetime.combine(day, cls._MARKET_OPEN)
            day_close = datetime.combine(day, cls._MARKET_CLOSE)
            seg_start = max(day_open, start)
            seg_end = min(day_close, end)

            cursor = seg_start
            while cursor < seg_end:
                nxt = min(cursor + chunk, seg_end)
                windows.append((cursor, nxt))
                cursor = nxt

            day += timedelta(days=1)
        return windows

    def get_option_candles(
        self,
        strike: int,
        expiry_date: date,
        option_type: str,
        start: datetime,
        end: datetime,
        interval: str = "1minute",
        cache_only: bool = False,
    ) -> list[dict] | None:
        """
        Fetch intraday candles for a Nifty option contract.

        option_type: "CE" (call) or "PE" (put)

        The request is paged into market-hours chunks (see ``_chunk_windows``) so
        that fine-grained intervals over multi-day windows are fetched in full
        rather than being truncated to Breeze's per-request row cap.

        When ``cache_only`` is True, data is served exclusively from the local
        cache; if the contract is not cached, ``None`` is returned (no API call)
        so the caller can skip it.
        """
        right = "call" if option_type == "CE" else "put"
        expiry_str = datetime(
            expiry_date.year, expiry_date.month, expiry_date.day, 6, 0, 0
        ).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        def _fetch() -> list[dict]:
            merged: dict[str, dict] = {}
            for chunk_start, chunk_end in self._chunk_windows(start, end, interval):
                resp = self.breeze.get_historical_data_v2(
                    interval=interval,
                    from_date=chunk_start,
                    to_date=chunk_end,
                    stock_code="NIFTY",
                    exchange_code="NFO",
                    product_type="options",
                    expiry_date=expiry_str,
                    right=right,
                    strike_price=str(strike),
                )
                for candle in resp.get("Success") or []:
                    # De-dupe on the candle timestamp in case chunk edges overlap.
                    merged[candle["datetime"]] = candle
            return [merged[k] for k in sorted(merged)]

        return self.cache.get_or_fetch(
            _fetch,
            cache_only=cache_only,
            stock_code="NIFTY",
            exchange_code="NFO",
            product_type="options",
            interval=interval,
            expiry=expiry_str,
            right=right,
            strike=strike,
            start=start,
            end=end,
        )

    def get_nifty_spot_candles(
        self,
        start: datetime,
        end: datetime,
        interval: str = "5minute",
    ) -> list[dict]:
        """Fetch Nifty 50 spot (cash) candles for signal generation."""

        def _fetch() -> list[dict]:
            resp = self.breeze.get_historical_data_v2(
                interval=interval,
                from_date=start,
                to_date=end,
                stock_code="NIFTY",
                exchange_code="NSE",
                product_type="cash",
            )
            return resp.get("Success") or []

        return self.cache.get_or_fetch(
            _fetch,
            stock_code="NIFTY",
            exchange_code="NSE",
            product_type="cash",
            interval=interval,
            start=start,
            end=end,
        )

    # ── Weekly expiry calendar ────────────────────────────────────────────────

    @staticmethod
    def weekly_wednesdays(start: date, end: date) -> list[date]:
        """Return all Tuesdays (Nifty weekly expiry day) between start and end (inclusive)."""
        wednesdays: list[date] = []
        d = start
        # advance to first Tuesday
        while d.weekday() != 1:  # 1 = Tuesday
            d += timedelta(days=1)
        while d <= end:
            wednesdays.append(d)
            d += timedelta(weeks=1)
        return wednesdays

    @staticmethod
    def adjust_expiry_for_holidays(
        expiry: date, holidays: set[date] | frozenset[date] | None
    ) -> date:
        """
        If a Tuesday expiry falls on a market holiday, roll it back to the
        previous working day. The replacement is moved back one day at a time,
        skipping weekends (Saturday/Sunday) and any further holidays, so the
        returned date is always a valid trading day.

        Non-holiday expiries are returned unchanged.
        """
        if not holidays:
            return expiry
        adjusted = expiry
        # Only Tuesday expiries are rolled back, per the trading calendar rule.
        while adjusted in holidays or adjusted.weekday() >= 5:
            adjusted -= timedelta(days=1)
        return adjusted

    @staticmethod
    def is_trading_day(
        day: date, holidays: set[date] | frozenset[date] | None = None
    ) -> bool:
        """True if ``day`` is a weekday (Mon–Fri) and not a market holiday."""
        if day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False
        if holidays and day in holidays:
            return False
        return True

    @classmethod
    def trading_days(
        cls,
        start: date,
        end: date,
        holidays: set[date] | frozenset[date] | None = None,
    ) -> list[date]:
        """
        Return every trading day in the inclusive range [start, end], skipping
        weekends and market holidays.
        """
        days: list[date] = []
        d = start
        while d <= end:
            if cls.is_trading_day(d, holidays):
                days.append(d)
            d += timedelta(days=1)
        return days

    @staticmethod
    def monday_of_week(expiry: date) -> date:
        """Return the Monday immediately before a given Tuesday expiry."""
        return expiry - timedelta(days=1)

    @staticmethod
    def week_window(expiry: date) -> tuple[date, date]:
        """
        Trading window for an expiry week: Wednesday (prior week) → Tuesday expiry.
        Previous Wednesday = expiry - 6 days.
        """
        prev_wednesday = expiry - timedelta(days=6)
        return prev_wednesday, expiry
