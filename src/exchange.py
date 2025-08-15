# Created by LP
# Date: 2025-08-13
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

"""
Output in logs folder:

2025-08-13 18:34:28 INFO: CCXT Version: 4.4.34
2025-08-13 18:34:28 INFO: === Available Exchanges (first 5) ===
2025-08-13 18:34:28 INFO: ['ace', 'alpaca', 'ascendex', 'bequant', 'bigone']
2025-08-13 18:34:28 INFO: Testing if BTC/USDT exists on bybit...
2025-08-13 18:34:30 INFO: ✅ BTC/USDT is available on bybit
2025-08-13 18:34:30 INFO: OHLCV data:
[[1755061200000, 119338.4, 119687.3, 118933.7, 119071.6, 277.571659],
 [1755064800000, 119071.6, 119432.5, 118945.5, 119416.0, 220.281558],
 [1755068400000, 119416.0, 119631.6, 119270.2, 119571.8, 197.690076],
 [1755072000000, 119571.8, 120040.8, 119440.1, 120039.6, 312.915963],
 [1755075600000, 120039.6, 120167.6, 119788.4, 119889.1, 155.948586]]
2025-08-13 18:34:30 INFO: === Testing multiple exchanges for BTC/USDT ===
2025-08-13 18:34:35 WARNING: ⚠️  ace: error ace GET https://ace.io/polarisex/oapi/v2/list/marketPair
2025-08-13 18:34:35 WARNING: ⚠️  alpaca: error alpaca requires "apiKey" credential
2025-08-13 18:34:37 INFO: ✅ ascendex: supports BTC/USDT (spot)
2025-08-13 18:34:39 INFO: ✅ bequant: supports BTC/USDT (spot)
2025-08-13 18:34:40 INFO: ✅ bigone: supports BTC/USDT (spot)
2025-08-13 18:34:40 INFO: ✅ binance: supports BTC/USDT (spot)
2025-08-13 18:34:41 WARNING: ⚠️  binancecoinm: error BTC/USDT not listed on binancecoinm in spot or swap or future
2025-08-13 18:34:41 INFO: ✅ binanceus: supports BTC/USDT (spot)
2025-08-13 18:34:42 WARNING: ⚠️  binanceusdm: error BTC/USDT not listed on binanceusdm in spot or swap or future
2025-08-13 18:34:42 INFO: ✅ bingx: supports BTC/USDT (spot)
"""
import os
import time
import logging
from pprint import pformat
import ccxt

# --- Logging Setup ---
os.makedirs("logs", exist_ok=True)  # create folder if not exists

logging.basicConfig(
    filename=os.path.join(
        "logs", "exchange.log"
    ),  # saves log file in the current working directory
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("exchange")

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logging.getLogger().addHandler(console)


def make_exchange(name: str, default_type: str = None, **kwargs):
    """
    Create a CCXT exchange instance with rate limiting and optional defaultType.
    """
    try:
        kwargs = {
            "enableRateLimit": True,
            "timeout": kwargs.get("timeout", 10000),  # default 10 seconds,
            **kwargs,
        }
        ex = getattr(ccxt, name)(kwargs)
        if default_type is not None:
            ex.options = {**getattr(ex, "options", {}), "defaultType": default_type}
        return ex
    except AttributeError:
        raise ValueError(f"Exchange '{name}' not found in CCXT.")


def assert_symbol_multi_type(
    exchange_or_name, symbol, default_type: str = None, **kwargs
):
    """
    Try spot first, then swap if symbol not found.
    Accepts either an exchange object or exchange name string.
    """
    if default_type:
        ex = make_exchange(exchange_or_name, default_type=default_type, **kwargs)
        ex.load_markets()
        if symbol not in ex.symbols:
            raise ValueError(f"{symbol} not listed on {ex.id} ({default_type})")
        return ex, default_type
    else:
        for market_type in ["spot", "swap", "future"]:
            if isinstance(exchange_or_name, str):
                ex = make_exchange(exchange_or_name, default_type=market_type, **kwargs)
            else:
                ex = exchange_or_name
                ex.options = {**getattr(ex, "options", {}), "defaultType": market_type}

            ex.load_markets()
            if symbol in ex.symbols:
                return ex, market_type
    raise ValueError(
        f"{symbol} not listed on {getattr(ex, 'id', exchange_or_name)} in spot or swap or future"
    )


def retry(
    func,
    retries=1,
    delay=0.5,
    exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout),
):
    """Retry a function if it hits a transient CCXT network issue."""
    for attempt in range(retries + 1):
        try:
            return func()
        except exceptions as e:
            if attempt < retries:
                time.sleep(delay)
            else:
                raise e


def main():
    log.info("CCXT Version: %s", ccxt.__version__)

    # Example: list first 5 exchanges
    log.info("=== Available Exchanges (first 5) ===")
    log.info(pformat(ccxt.exchanges[:5]))

    # Test a single exchange
    exchange_name = "bybit"
    ex = make_exchange(exchange_name)
    symbol_to_check = "BTC/USDT"

    log.info("Testing if %s exists on %s...", symbol_to_check, exchange_name)
    if assert_symbol_multi_type(ex, symbol_to_check):
        log.info("✅ %s is available on %s", symbol_to_check, exchange_name)
        ohlcv = retry(lambda: ex.fetch_ohlcv(symbol_to_check, timeframe="1h", limit=5))
        log.info("OHLCV data:\n%s", pformat(ohlcv))

    # Test multiple exchanges
    log.info("=== Testing multiple exchanges for %s ===", symbol_to_check)
    for ex_id in ccxt.exchanges[:10]:
        try:
            ex, mtype = assert_symbol_multi_type(ex_id, "BTC/USDT", timeout=5000)
            log.info("✅ %s: supports BTC/USDT (%s)", ex_id, mtype)
        except Exception as e:
            log.warning("⚠️  %s: error %s", ex_id, e)


if __name__ == "__main__":
    main()
