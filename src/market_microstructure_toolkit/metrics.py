# Created by LP
# Date: 2025-08-17
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

r"""
### Phase-2 metrics (formulas)

- **Relative spread (bps)**
  \( \text{mid}=\frac{b+a}{2} \), \( \text{rel\_spread\_bps}=10^4\frac{a-b}{\text{mid}} \)

- **Microprice**
  \( \text{mp}=\frac{a\cdot \text{bidSize} + b\cdot \text{askSize}}{\text{bidSize} + \text{askSize}} \)
  **Microprice imbalance (bps):** \( 10^4\frac{\text{mp}-\text{mid}}{\text{mid}} \)

- **Rolling realized variance (RV)** over window \(W\) (using log returns):
  \( r_t=\ln\left(\frac{p_t}{p_{t-1}}\right),\quad \text{RV}_T=\sum_{t=T-W+1}^{T} r_t^2 \)

- **Notional depth (top-K)**
  bid/ask notional = \( \sum_{i=1}^K p_i \cdot q_i \)

- **Book slope (toy proxy)**
  OLS slope of cumulative size \( y_i=\sum_{j\le i}q_j \) vs. price distance \( x_i=|p_i-p_1| \).

- **OFI (L1)** (Cont-style proxy): reacts to changes at best bid/ask across consecutive rows.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _parse_level(row: Dict[str, Any], side: str, i: int) -> Tuple[Optional[float], Optional[float]]:
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


def imbalance_l1(best_bid_size: Optional[float], best_ask_size: Optional[float]) -> Optional[float]:
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


def relative_spread_bps(best_bid, best_ask) -> Optional[float]:
    """
    Relative spread in basis points: 10_000 * (ask - bid) / mid.
    Returns None if inputs are missing/invalid or if ask <= bid.
    """
    try:
        bb = float(best_bid)
        ba = float(best_ask)
    except (TypeError, ValueError):
        return None

    # Reject non-finite or non-positive prices
    if not (math.isfinite(bb) and math.isfinite(ba)):
        return None
    if bb <= 0 or ba <= 0:
        return None
    if ba <= bb:
        return None

    mid = 0.5 * (bb + ba)
    if not math.isfinite(mid) or mid <= 0:
        return None

    return 10_000.0 * (ba - bb) / mid


def microprice(
    best_bid: Optional[float],
    best_ask: Optional[float],
    bid_size: Optional[float],
    ask_size: Optional[float],
) -> Optional[float]:
    """
    Microprice (queue-size–weighted mid).
    Common definition:
        mp = (best_ask * bid_size + best_bid * ask_size) / (bid_size + ask_size)
    Intuition: price leans toward the side with *less* queue (more pressure).
    Returns None if any of the required inputs are missing or denom <= 0.
    """
    if (best_bid is None) or (best_ask is None) or (bid_size is None) or (ask_size is None):
        return None
    denom = bid_size + ask_size
    if denom <= 0:
        return None
    return (best_ask * bid_size + best_bid * ask_size) / denom


def microprice_imbalance(
    best_bid: Optional[float],
    best_ask: Optional[float],
    bid_size: Optional[float],
    ask_size: Optional[float],
) -> Optional[float]:
    """
    Microprice imbalance: normalized deviation of microprice vs mid in bps.
    Formula:
        mid = (bb + ba)/2
        mp  = microprice(bb, ba, bsz, asz)
        mpi_bps = 10_000 * (mp - mid) / mid
    Returns None if mid or mp unavailable or mid<=0.
    """
    if (best_bid is None) or (best_ask is None):
        return None
    mid = 0.5 * (best_bid + best_ask)
    if mid <= 0:
        return None
    mp = microprice(best_bid, best_ask, bid_size, ask_size)
    if mp is None:
        return None
    return 10_000.0 * (mp - mid) / mid


# TODO not in unit test
# --- Rolling realized variance (sum of squared returns) ---
def realized_var(price, window: int = 20, use_log: bool = True) -> pd.Series:
    """
    Rolling realized variance over 'window' samples.

    Args
    ----
    price : sequence-like of floats (e.g., mid)
    window : int
        Rolling window length (in rows, not seconds).
    use_log : bool
        If True, use log-returns; else simple pct-change.

    Returns
    -------
    pd.Series aligned to 'price' with NaN until window is filled.
    """
    s = pd.Series(price, dtype="float64")
    if use_log:
        r = np.log(s).diff()
    else:
        r = s.pct_change()
    return r.pow(2).rolling(window=window, min_periods=window).sum()


def rolling_realized_variance(prices: Iterable[float], window: int) -> List[Optional[float]]:
    """
    Realized variance over a rolling window using simple log-returns.
    For a series p_t, define r_t = ln(p_t / p_{t-1}).
    Windowed RV_T = sum_{t=T-window+1..T} r_t^2
    Returns RV series aligned with input length; first window-1 values are None.
    """
    prices = list(prices)
    n = len(prices)
    if window <= 1:
        return [None] * n
    # compute returns (None for the first)
    rets: List[Optional[float]] = [None]
    for t in range(1, n):
        p0, p1 = prices[t - 1], prices[t]
        if p0 is None or p1 is None or p0 <= 0 or p1 <= 0:
            rets.append(None)
        else:
            rets.append(math.log(p1 / p0))
    # rolling sum of r^2
    out: List[Optional[float]] = []
    acc = 0.0
    q = []  # last (window-1) return squares
    for t in range(n):
        r = rets[t]
        if r is None:
            out.append(None)
            q.clear()
            acc = 0.0
            continue
        r2 = r * r
        q.append(r2)
        acc += r2
        if len(q) > window:
            acc -= q.pop(0)
        out.append(acc if len(q) == window else None)
    return out


def notional_depth(row: Dict[str, Any], depth: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Sum of (price * size) for top-K per side (notional in quote currency).
    Returns (bid_notional, ask_notional). Missing/blank levels ignored.
    """
    bid_notional = 0.0
    ask_notional = 0.0
    any_bid = False
    any_ask = False
    for i in range(1, depth + 1):
        bpx, bsz = _parse_level(row, "bid", i)
        apx, asz = _parse_level(row, "ask", i)
        if bpx is not None and bsz is not None:
            bid_notional += bpx * bsz
            any_bid = True
        if apx is not None and asz is not None:
            ask_notional += apx * asz
            any_ask = True
    return (bid_notional if any_bid else None, ask_notional if any_ask else None)


def book_slope(row: Dict[str, Any], depth: int, side: str = "bid") -> Optional[float]:
    """
    Very simple book 'slope' proxy via linear fit of cumulative size vs. price delta.
    For side ∈ {'bid','ask'}:
       x_i = |p_i - p_1|   (ticks or price units)
       y_i = sum_{j=1..i} size_j
    slope = cov(x,y)/var(x)  (ordinary least squares through the mean)
    Returns None if fewer than 2 valid levels.
    """
    assert side in {"bid", "ask"}
    prices = []
    sizes = []
    for i in range(1, depth + 1):
        px, sz = _parse_level(row, side, i)
        if px is not None and sz is not None:
            prices.append(px)
            sizes.append(sz)
    if len(prices) < 2:
        return None
    # build (x, y)
    p0 = prices[0]
    x = [abs(p - p0) for p in prices]
    y = []
    csum = 0.0
    for s in sizes:
        csum += s
        y.append(csum)
    # OLS slope
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den = sum((xi - mean_x) ** 2 for xi in x)
    if den <= 0:
        return None
    return num / den


def ofi_l1(prev_row: Dict[str, Any], row: Dict[str, Any]) -> Optional[float]:
    """
    Order Flow Imbalance (best quotes), contiguous rows only (L1 approximation).
    One popular L1 proxy (Cont et al.):
      OFI_t =
        + Δbid_size if bid_price_t > bid_price_{t-1}
        - bid_size_{t-1} if bid_price_t < bid_price_{t-1}
        + Δask_size if ask_price_t < ask_price_{t-1}
        - ask_size_{t-1} if ask_price_t > ask_price_{t-1}
    Returns None if any best levels are missing.
    """
    b0, a0 = row.get("bid1_price", ""), row.get("ask1_price", "")
    s0b, s0a = row.get("bid1_size", ""), row.get("ask1_size", "")
    b1, a1 = prev_row.get("bid1_price", ""), prev_row.get("ask1_price", "")
    s1b, s1a = prev_row.get("bid1_size", ""), prev_row.get("ask1_size", "")
    try:
        b0 = float(b0) if b0 != "" else None
        a0 = float(a0) if a0 != "" else None
        s0b = float(s0b) if s0b != "" else None
        s0a = float(s0a) if s0a != "" else None
        b1 = float(b1) if b1 != "" else None
        a1 = float(a1) if a1 != "" else None
        s1b = float(s1b) if s1b != "" else None
        s1a = float(s1a) if s1a != "" else None
    except Exception:
        return None
    if None in (b0, a0, s0b, s0a, b1, a1, s1b, s1a):
        return None

    out = 0.0
    # bid side
    if b0 > b1:
        out += s0b - s1b
    elif b0 < b1:
        out -= s1b
    # ask side
    if a0 < a1:
        out += s0a - s1a
    elif a0 > a1:
        out -= s1a
    return out


def compute_row_metrics(
    row: Dict[str, Any], depth: int, rv_window: int | None = None
) -> Dict[str, Optional[float]]:
    """
    Compute row-wise metrics using *only this row* (except RV which needs a series; see CLI).
    Returns:
      spread, mid, relative_spread_bps, microprice, microprice_imbalance,
      imbalance_l1, imbalance_k,
      notional_bid_k, notional_ask_k
    """
    # parse bests
    try:
        bb = float(row.get("best_bid", "")) if row.get("best_bid", "") != "" else None
    except Exception:
        bb = None
    try:
        ba = float(row.get("best_ask", "")) if row.get("best_ask", "") != "" else None
    except Exception:
        ba = None

    # sizes at level-1
    _, bid1_sz = _parse_level(row, "bid", 1)
    _, ask1_sz = _parse_level(row, "ask", 1)

    sp, mid = spread_and_mid(bb, ba)
    rel_bps = relative_spread_bps(bb, ba)
    mp = microprice(bb, ba, bid1_sz, ask1_sz)
    mpi = microprice_imbalance(bb, ba, bid1_sz, ask1_sz)
    imb_l1 = imbalance_l1(bid1_sz, ask1_sz)
    imb_k = imbalance_depth_k(row, depth)
    nb, na = notional_depth(row, depth)

    return {
        "spread": sp,
        "mid": mid,
        "relative_spread_bps": rel_bps,
        "microprice": mp,
        "microprice_imbalance_bps": mpi,
        "imbalance_l1": imb_l1,
        "imbalance_k": imb_k,
        "notional_bid_k": nb,
        "notional_ask_k": na,
        # rolling RV is added in the CLI where we have a time series
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
