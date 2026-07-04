# Resources

Central place for backtest input data so stock universes and contract
details are no longer hardcoded inside strategy scripts.

## Layout

```
resources/
├── stocks.json              # all stock groups in one file, keyed by group name
├── futures_contracts.json   # futures contract details for all available stocks
├── indices.json             # index definitions (NIFTY, BANKNIFTY, SENSEX, ...)
├── resource_loader.py       # helpers to load these files in scripts
└── README.md
```

## Stock lists (`stocks.json`)

- One JSON object; each key is a group (sector / watchlist), each value a list
  of symbols in the Breeze format used across the repo: `NSE:<SCRIP>-EQ`.
- Add a new watchlist by simply adding a new key with its symbol list.

```json
{
  "banking": ["NSE:HDFBAN-EQ", "NSE:ICIBAN-EQ"],
  "it": ["NSE:TCS-EQ", "NSE:INFTEC-EQ"]
}
```

## Futures contracts (`futures_contracts.json`)

A JSON list of contract objects:

| Key            | Meaning                                        |
|----------------|------------------------------------------------|
| scrip          | Breeze scrip code (without exchange/`-EQ`)     |
| expiry         | Contract expiry, `DD-Mon-YYYY`                 |
| lot_size       | Contract lot size (shares per lot)             |
| margin_per_lot | Approximate margin required per lot (INR)      |

> **Note:** Lot sizes and margins change with NSE circulars and broker
> policy. The values here are indicative starting points — refresh them
> from your broker's margin file / NSE F&O lot size circular before
> relying on them for position sizing. The `expiry` value should be
> rolled forward each month.

## Indices (`indices.json`)

A JSON object with an `indices` list; each entry has:

| Key         | Meaning                                  |
|-------------|------------------------------------------|
| name        | Human-readable index name                |
| symbol      | `<EXCHANGE>:<BREEZE_CODE>`               |
| breeze_code | Breeze scrip code for the index          |
| exchange    | NSE / BSE                                |

## Usage in scripts

```python
from resources.resource_loader import (
    load_stocks, load_all_stocks, load_futures_contracts, load_indices
)

SYMBOLS = load_stocks("banking")          # a single list
SYMBOLS = load_all_stocks()               # every list merged, de-duplicated

contracts = load_futures_contracts()
lot = contracts["TCS"].lot_size
margin = contracts["TCS"].margin_per_lot

indices = load_indices()
nifty = indices["NIFTY"].symbol           # "NSE:NIFTY"
```
