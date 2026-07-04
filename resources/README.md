# Resources

Central place for backtest input data so stock universes and contract
details are no longer hardcoded inside strategy scripts.

## Layout

```
resources/
├── stocks/                  # stocks to be backtested, one txt file per group
│   ├── banking.txt          # one Breeze scrip code per line, e.g. NSE:HDFBAN-EQ
│   ├── it.txt
│   └── ...
├── futures_contracts.txt    # futures contract details for all available stocks
├── resource_loader.py       # helpers to load these files in scripts
└── README.md
```

## Stock lists (`stocks/*.txt`)

- One symbol per line in the Breeze format used across the repo: `NSE:<SCRIP>-EQ`.
- Blank lines and lines starting with `#` are ignored.
- Add a new watchlist by simply dropping a new `.txt` file into `stocks/`.

## Futures contracts (`futures_contracts.txt`)

Tab-separated file with a header row:

| Column     | Meaning                                        |
|------------|------------------------------------------------|
| Scrip      | Breeze scrip code (without exchange/`-EQ`)     |
| Expiry     | Contract expiry, `DD-Mon-YYYY`                 |
| Lot Size   | Contract lot size (shares per lot)             |
| Margin/Lot | Approximate margin required per lot (INR)      |

> **Note:** Lot sizes and margins change with NSE circulars and broker
> policy. The values here are indicative starting points — refresh them
> from your broker's margin file / NSE F&O lot size circular before
> relying on them for position sizing. The `Expiry` column should be
> rolled forward each month.

## Usage in scripts

```python
from resources.resource_loader import load_stocks, load_all_stocks, load_futures_contracts

SYMBOLS = load_stocks("banking")          # a single list
SYMBOLS = load_all_stocks()               # every list merged, de-duplicated

contracts = load_futures_contracts()
lot = contracts["TCS"].lot_size
margin = contracts["TCS"].margin_per_lot
```
