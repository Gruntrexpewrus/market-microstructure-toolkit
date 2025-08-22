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
# src/market_microstructure_toolkit/exchange.py
# Created by LP
# Date: 2025-08-13

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Tuple, TypeVar

import ccxt

log = logging.getLogger(__name__)  # no handlers here

T = TypeVar("T")


def make_exchange(name: str, default_type: Optional[str] = None, **kwargs) -> ccxt.Exchange:
    """
    Create a CCXT exchange with rate limiting and optional defaultType.

    Parameters
    ----------
    name : str
        ccxt exchange id, e.g. "bybit", "binance".
    default_type : {"spot","swap","future"} | None
        Sets exchange.options["defaultType"] when provided.
    **kwargs :
        Passed to the ccxt exchange constructor (e.g., timeout=10000).

    Returns
    -------
    ccxt.Exchange
    """
    try:
        params = {
            "enableRateLimit": True,
            "timeout": kwargs.get("timeout", 10_000),
            **kwargs,
        }
        ex = getattr(ccxt, name)(params)
    except AttributeError as e:
        raise ValueError(f"Exchange '{name}' not found in ccxt.") from e

    if default_type is not None:
        ex.options = {**getattr(ex, "options", {}), "defaultType": default_type}
    return ex


def assert_symbol_multi_type(
    exchange_or_name: ccxt.Exchange | str,
    symbol: str,
    default_type: Optional[str] = None,
    **kwargs,
) -> Tuple[ccxt.Exchange, str]:
    """
    Ensure `symbol` exists on the exchange, trying market types.

    If `default_type` is given, only that type is checked. Otherwise tries
    in order: spot → swap → future.

    Returns
    -------
    (exchange, market_type)

    Raises
    ------
    ValueError if the symbol is not listed under the tried market types.
    """
    # Single-type path
    if default_type:
        ex = (
            make_exchange(exchange_or_name, default_type=default_type, **kwargs)
            if isinstance(exchange_or_name, str)
            else exchange_or_name
        )
        if default_type and not isinstance(exchange_or_name, str):
            ex.options = {**getattr(ex, "options", {}), "defaultType": default_type}
        ex.load_markets()
        if symbol not in ex.symbols:
            raise ValueError(f"{symbol} not listed on {ex.id} ({default_type})")
        return ex, default_type

    # Multi-try path
    for market_type in ("spot", "swap", "future"):
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
    func: Callable[[], T],
    retries: int = 1,
    delay: float = 0.5,
    exceptions: tuple[type[BaseException], ...] = (
        ccxt.NetworkError,
        ccxt.ExchangeNotAvailable,
        ccxt.RequestTimeout,
    ),
) -> T:
    """Retry a callable on transient ccxt network errors."""
    for attempt in range(retries + 1):
        try:
            return func()
        except exceptions:
            if attempt < retries:
                time.sleep(delay)
            else:
                raise


if __name__ == "__main__":
    # Demo only: configure logging here, not on import.
    from .setup_log import setup_logging  # package-local helper

    log = setup_logging(name="exchange")
    log.info("CCXT Version: %s", ccxt.__version__)

    ex = make_exchange("bybit", default_type=None, timeout=10000)
    symbol = "BTC/USDT"

    try:
        ex2, mtype = assert_symbol_multi_type(ex, symbol)
        log.info("✅ %s available on %s (%s)", symbol, ex2.id, mtype)
        data = retry(lambda: ex2.fetch_ohlcv(symbol, timeframe="1h", limit=3))
        log.info("Sample OHLCV: %s", data)
    except Exception as e:
        log.warning("Symbol check failed: %s", e)
