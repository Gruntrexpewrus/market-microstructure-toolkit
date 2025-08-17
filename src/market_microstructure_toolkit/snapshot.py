# Created by LP
# Date: 2025-08-14
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

# src/snapshot.py
# Created by LP
# T3: Normalized order-book snapshot (L1/L2/L3)

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Optional
import logging

from .setup_log import setup_logging

log = logging.getLogger(__name__)  # no handlers here


def _norm(
    levels_list: Optional[List[List[float]]], n: int
) -> List[Tuple[float, float]]:
    """Normalize raw [price, size] into [(float(px), float(sz))], truncated to n."""
    out: List[Tuple[float, float]] = []
    if not levels_list:
        return out
    for lvl in levels_list[:n]:
        try:
            px, sz = float(lvl[0]), float(lvl[1])
            out.append((px, sz))
        except Exception as e:
            log.warning("Skipping malformed level %s (err=%s)", lvl, e)
    return out


def _iso_utc(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def fetch_order_book_snapshot(
    ex,
    symbol: str,
    depth: int = 10,
    book_level: str = "L2",  # "L1" | "L2" | "L3"
    params: Optional[dict] = None,  # exchange-specific params if needed
) -> Dict[str, Any]:
    """
    Return a normalized snapshot dict.

    Keys:
      ts_ms: int (ms)
      iso: str (UTC ISO8601)
      best_bid, best_ask: float | None
      bids, asks: list[(price, size)] length ≤ depth (≤1 for L1)
      symbol: str (from exchange or input)
      exchange_id: str
      raw_nonce: int | None
      book_level: "L1" | "L2" | "L3"

    Notes:
      - L2 uses unified fetch_l2_order_book when available, else fetch_order_book.
      - L3 via REST is generally unsupported; pass venue params or use websockets.
      - We always truncate to enforce deterministic length.
    """
    params = params or {}

    if book_level == "L1":
        ob = ex.fetch_order_book(symbol, limit=max(1, depth), params=params)
        slice_n = 1
    elif book_level == "L2":
        if hasattr(ex, "fetch_l2_order_book"):
            ob = ex.fetch_l2_order_book(symbol, limit=depth, params=params)
        else:
            ob = ex.fetch_order_book(symbol, limit=depth, params=params)
        slice_n = depth
    elif book_level == "L3":
        if params:
            ob = ex.fetch_order_book(symbol, limit=depth, params=params)
        else:
            raise NotImplementedError(
                "L3 (per-order) is not unified in ccxt REST. "
                "Pass venue-specific params (e.g., {'level': 3}) if supported, "
                "or use the exchange websocket for raw book."
            )
        slice_n = depth
    else:
        raise ValueError(f"Unknown book_level: {book_level}")

    # Defensive access (some venues omit timestamp/symbol/nonce)
    ts_ms = ob.get("timestamp") or int(time.time() * 1000)
    bids_raw = ob.get("bids") or []
    asks_raw = ob.get("asks") or []
    symbol_ex = ob.get("symbol") or symbol

    bids = _norm(bids_raw, slice_n)
    asks = _norm(asks_raw, slice_n)

    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None

    if best_bid is not None and best_ask is not None and best_bid >= best_ask:
        log.warning(
            "Crossed/locked book: best_bid=%.8f best_ask=%.8f (%s %s, %s)",
            best_bid,
            best_ask,
            ex.id,
            symbol_ex,
            book_level,
        )

    return {
        "ts_ms": ts_ms,
        "iso": _iso_utc(ts_ms),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bids": bids,
        "asks": asks,
        "symbol": symbol_ex,
        "exchange_id": getattr(ex, "id", None),
        "raw_nonce": ob.get("nonce"),
        "book_level": book_level,
    }


# Optional: local smoke test. Prefer a separate script in scripts/ for cleanliness.
if __name__ == "__main__":
    from .exchange import assert_symbol_multi_type

    # Configure the named logger used in this module
    log = setup_logging(name="snapshot")
    ex, mtype = assert_symbol_multi_type(
        "bybit", "BTC/USDT:USDT", timeout=10000, default_type="swap"
    )
    log.info("Using %s (%s)", ex.id, mtype)

    s1 = fetch_order_book_snapshot(ex, "BTC/USDT:USDT", depth=1, book_level="L1")
    log.info("L1 best_bid=%s best_ask=%s", s1["best_bid"], s1["best_ask"])

    s2 = fetch_order_book_snapshot(ex, "BTC/USDT:USDT", depth=10, book_level="L2")
    log.info("L2 bids[:2]=%s asks[:2]=%s", s2["bids"][:2], s2["asks"][:2])
