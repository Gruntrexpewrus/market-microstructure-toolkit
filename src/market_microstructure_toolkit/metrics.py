# Created by LP
# Date: 2025-08-17
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

# Phase 1 microstructure metrics: spread, midprice, depth imbalance (L1 and depth-K)

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional


def _parse_level(
    row: Dict[str, Any], side: str, i: int
) -> Tuple[Optional[float], Optional[float]]:
    """Extract (price, size) for side='bid'|'ask' at level i (1-based). Returns (None, None) if blank."""
    px_key, sz_key = f"{side}{i}_price", f"{side}{i}_size"
    px = row.get(px_key, "")
    sz = row.get(sz_key, "")
    try:
        px_val = float(px) if px != "" else None
    except Exception:
        px_val = None
    try:
        sz_val = float(sz) if sz != "" else None
    except Exception:
        sz_val = None
    return px_val, sz_val


def spread_and_mid(
    best_bid: Optional[float], best_ask: Optional[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Return (spread, mid). If either side missing, returns (None, None) for spread and mid respectively."""
    spread = None
    mid = None
    if best_bid is not None and best_ask is not None:
        spread = best_ask - best_bid
        mid = 0.5 * (best_bid + best_ask)
    return spread, mid


def imbalance_l1(
    best_bid_size: Optional[float], best_ask_size: Optional[float]
) -> Optional[float]:
    """Depth imbalance at level-1 only: bidSize / (bidSize + askSize). Returns None if both missing or zero."""
    b = best_bid_size or 0.0
    a = best_ask_size or 0.0
    denom = b + a
    if denom <= 0:
        return None
    return b / denom


def imbalance_depth_k(row: Dict[str, Any], depth: int) -> Optional[float]:
    """
    Depth imbalance aggregated across the top-K: sum(bid sizes) / (sum(bid sizes) + sum(ask sizes)).
    Returns None if both sums are zero/missing.
    """
    bid_sum = 0.0
    ask_sum = 0.0
    for i in range(1, depth + 1):
        _, bsz = _parse_level(row, "bid", i)
        _, asz = _parse_level(row, "ask", i)
        if bsz is not None:
            bid_sum += bsz
        if asz is not None:
            ask_sum += asz
    denom = bid_sum + ask_sum
    if denom <= 0:
        return None
    return bid_sum / denom


def compute_row_metrics(row: Dict[str, Any], depth: int) -> Dict[str, Optional[float]]:
    """
    Compute spread, mid, L1 imbalance, and depth-K imbalance for a single flat row from record.py.
    Expects columns: best_bid, best_ask, bid{i}_size, ask{i}_size...
    """
    # best bid/ask
    try:
        bb = float(row.get("best_bid", "")) if row.get("best_bid", "") != "" else None
    except Exception:
        bb = None
    try:
        ba = float(row.get("best_ask", "")) if row.get("best_ask", "") != "" else None
    except Exception:
        ba = None

    # best sizes (for L1 imbalance)
    _, bid1_sz = _parse_level(row, "bid", 1)
    _, ask1_sz = _parse_level(row, "ask", 1)

    spread, mid = spread_and_mid(bb, ba)
    imb_l1 = imbalance_l1(bid1_sz, ask1_sz)
    imb_k = imbalance_depth_k(row, depth)

    return {
        "spread": spread,
        "mid": mid,
        "imbalance_l1": imb_l1,
        "imbalance_k": imb_k,
    }


if __name__ == "__main__":
    """
    Demo: run metrics on a fake row (no files, no exchange).
    Usage: python -m src.metrics
    """
    # Demo only: configure logging here, not on import.
    from .setup_log import setup_logging  # package-local helper

    log = setup_logging(name="metrics")

    fake_row = {
        "best_bid": "100.0",
        "best_ask": "100.1",
        "bid1_price": "100.0",
        "bid1_size": "1.0",
        "bid2_price": "99.9",
        "bid2_size": "0.5",
        "ask1_price": "100.1",
        "ask1_size": "0.8",
        "ask2_price": "100.2",
        "ask2_size": "0.6",
    }

    depth = 2
    m = compute_row_metrics(fake_row, depth)
    log.info("Input row:", fake_row)
    log.info(f"Metrics at depth={depth}:")
    for k, v in m.items():
        log.info(f"  {k}: {v}")
