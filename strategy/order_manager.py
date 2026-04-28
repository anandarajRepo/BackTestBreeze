from breeze_connect import BreezeConnect


class OrderManager:
    def __init__(self, breeze: BreezeConnect):
        self.breeze = breeze

    def place_market_order(
        self,
        action: str,
        stock_code: str,
        exchange_code: str,
        quantity: int,
    ) -> dict:
        return self.breeze.place_order(
            stock_code=stock_code,
            exchange_code=exchange_code,
            product="cash",
            action=action,
            order_type="market",
            quantity=str(quantity),
            price="0",
            validity="day",
        )
