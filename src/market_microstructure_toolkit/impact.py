# Created by LP
# Date: 2025-08-22
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

# impact.py simulates naive TWAP/VWAP execution by “walking” your recorded order book (L2) at each slice

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# ----- Public types

Side = str  # "buy" | "sell"
Fill = Tuple[float, float]  # (price, qty)


@dataclass
class ImpactResult:
    """Container for execution simulation outcome.

    Notes
    -----
    - `vwap` is the achieved price-weighted average (sum(p*q)/sum(q))
    - Slippage is reported in **bps** (1e4 * (achieved/reference - 1)).
      For sells we flip the comparison so that *worse* execution is positive slippage.
    """

    side: Side
    target_qty: float
    filled_qty: float
    vwap: Optional[float]
    notional: float
    slippage_bps_vs_mid_open: Optional[float]
    slippage_bps_vs_mid_close: Optional[float]
    slices: int
    fills: List[Dict[str, Any]]  # per-slice rollup rows


# ===== Internal helpers ======================================================


def _col(px_or_sz: str, side: Side, level: int) -> str:
    """Build a flattened column name for the side+level.

    Parameters
    ----------
    px_or_sz : {"price", "size"}
        Which field to read.
    side : {"buy","sell"}
        Execution side; determines which book side to consume (buy hits asks, sell hits bids).
    level : int
        1-based depth level.

    Returns
    -------
    str
        e.g. 'ask1_price' for buy@L1, 'bid2_size' for sell@L2.

    Implementation note
    -------------------
    The “trick” here is simply mapping execution side → opposing book side:
    buyers lift the **asks**, sellers hit the **bids**.
    """
    book_side = "ask" if side == "buy" else "bid"
    return f"{book_side}{level}_{px_or_sz}"


def _row_ladder(row: Dict[str, Any], side: Side, depth: int) -> List[Tuple[float, float]]:
    """Extract a (price, size) ladder from a flat row.

    Parameters
    ----------
    row : dict
        A single flattened snapshot row.
    side : {"buy","sell"}
        Execution side (selects asks for buys, bids for sells).
    depth : int
        Max levels to read (top-K).

    Returns
    -------
    list[tuple[float,float]]
        Cleaned list of (price, size), skipping blanks.

    Notes
    -----
    - Any non-numeric or missing values are ignored.
    - For buy-side execution, the natural book order is ascending asks (as stored).
      For sell-side execution, we use the best-to-worse bids as stored.
    """
    ladder: List[Tuple[float, float]] = []
    for i in range(1, depth + 1):
        p = row.get(_col("price", side, i))
        s = row.get(_col("size", side, i))
        if p in ("", None) or s in ("", None):
            continue
        try:
            ladder.append((float(p), float(s)))
        except Exception:
            # Drop malformed cells silently; snapshots can be sparse
            continue
    return ladder


def _walk_book(
    side: Side, qty: float, ladder: List[Tuple[float, float]]
) -> Tuple[List[Fill], float]:
    """Walk the ladder to fill up to `qty`.

    Parameters
    ----------
    side : {"buy","sell"}
        Only used for semantics; not needed by the math here.
    qty : float
        Target base qty for this slice.
    ladder : list[(price, size)]
        Levels available to consume.

    Returns
    -------
    (fills, filled_qty) : (list[(px,qty)], float)
        The list of price-level fills and the actually filled quantity.

    Notes
    -----
    - Partial fills at the last level are handled via `min(remaining, level_size)`.
    - No crossing beyond the provided `ladder` (no hidden liquidity).
    """
    remaining = qty
    fills: List[Fill] = []
    for price, avail in ladder:
        if remaining <= 0:
            break
        take = min(remaining, max(0.0, avail))
        if take > 0:
            fills.append((price, take))
            remaining -= take
    filled = qty - remaining
    return fills, filled


def _fills_vwap(fills: Iterable[Fill]) -> Optional[float]:
    """Compute VWAP across `fills` (sum(p*q)/sum(q)).

    Returns
    -------
    float | None
        None if total quantity == 0.
    """
    nq = 0.0
    pq = 0.0
    for px, q in fills:
        pq += px * q
        nq += q
    if nq <= 0:
        return None
    return pq / nq


def _mid_from_row(row: Dict[str, Any]) -> Optional[float]:
    """Return mid price for a row (prefer precomputed `mid`, else derive from best bid/ask)."""
    mid = row.get("mid")
    if mid not in ("", None):
        try:
            return float(mid)
        except Exception:
            pass
    bb, ba = row.get("best_bid"), row.get("best_ask")
    if bb in ("", None) or ba in ("", None):
        return None
    try:
        return (float(bb) + float(ba)) / 2.0
    except Exception:
        return None


def _bps(achieved: Optional[float], reference: Optional[float]) -> Optional[float]:
    """Return slippage in basis points: 1e4 * (achieved / reference - 1).

    Returns
    -------
    float | None
        None if inputs invalid.
    """
    if achieved is None or reference is None or reference == 0:
        return None
    return 1e4 * (achieved / reference - 1.0)


def _slice_indices_by_time(df: pd.DataFrame, slices: int) -> List[int]:
    """Pick evenly spaced indices over time for TWAP.

    Parameters
    ----------
    df : pd.DataFrame
        Recording ordered by time.
    slices : int
        Number of slices to select.

    Returns
    -------
    list[int]
        Positional indices to use for each slice.

    Implementation trick
    --------------------
    Uses `numpy.linspace` over integer positions to guarantee:
    - Inclusion of the *last* index (endpoint=True),
    - Roughly even spacing even if rows are irregular in wall-clock time.
    """
    if len(df) == 0:
        return []
    if slices <= 1:
        return [df.index[-1]]
    import numpy as np

    idx = np.linspace(0, len(df) - 1, num=slices, endpoint=True).round().astype(int)
    return list(sorted(set(idx.tolist())))


# ===== Public simulators =====================================================


def simulate_twap(
    df: pd.DataFrame,
    side: Side,
    total_qty: float,
    slices: int,
    depth: int,
) -> ImpactResult:
    """Naive TWAP: split total qty into equal time slices; walk L2 at each slice.

    Parameters
    ----------
    df : pd.DataFrame
        Flat L2 snapshots as produced by your recorders.
    side : {"buy","sell"}
        Execution side; buy consumes asks, sell consumes bids.
    total_qty : float
        Total base quantity to execute.
    slices : int
        Number of equal time slices (child orders).
    depth : int
        Max levels to walk at each slice.

    Returns
    -------
    ImpactResult
        Summary + per-slice rollup.

    Slippage convention
    -------------------
    - For **buy**: slippage_bps = achieved_vwap / mid - 1  (positive = worse)
    - For **sell**: slippage_bps = mid / achieved_vwap - 1 (positive = worse)
    """
    assert side in ("buy", "sell"), "side must be 'buy' or 'sell'"
    assert total_qty > 0 and slices >= 1 and depth >= 1

    picks = _slice_indices_by_time(df, slices)
    if not picks:
        return ImpactResult(side, total_qty, 0.0, None, 0.0, None, None, 0, [])

    slice_qty = total_qty / len(picks)

    fills_all: List[Dict[str, Any]] = []
    all_fills: List[Fill] = []
    mid_open = _mid_from_row(df.iloc[picks[0]].to_dict())

    for i, idx in enumerate(picks, 1):
        row = df.iloc[idx].to_dict()
        ladder = _row_ladder(row, side=side, depth=depth)
        fills, filled = _walk_book(side, qty=slice_qty, ladder=ladder)
        vwap_slice = _fills_vwap(fills)

        all_fills.extend(fills)
        fills_all.append(
            {
                "slice": i,
                "ts_ms": row.get("ts_ms"),
                "iso": row.get("iso"),
                "filled_qty": filled,
                "slice_target_qty": slice_qty,
                "slice_vwap": vwap_slice,
                "levels_touched": len(fills),
            }
        )

    filled_qty = sum(q for _, q in all_fills)
    vwap = _fills_vwap(all_fills)
    notional = sum(px * q for px, q in all_fills)

    mid_close = _mid_from_row(df.iloc[picks[-1]].to_dict())

    if vwap is None:
        s_open = s_close = None
    else:
        if side == "buy":
            s_open = _bps(vwap, mid_open)
            s_close = _bps(vwap, mid_close)
        else:
            # For sells, report "worse is positive": mid / vwap - 1
            s_open = _bps(mid_open, vwap)
            s_close = _bps(mid_close, vwap)

    return ImpactResult(
        side=side,
        target_qty=total_qty,
        filled_qty=filled_qty,
        vwap=vwap,
        notional=notional,
        slippage_bps_vs_mid_open=s_open,
        slippage_bps_vs_mid_close=s_close,
        slices=len(picks),
        fills=fills_all,
    )


def simulate_vwap_onbook(
    df: pd.DataFrame,
    side: Side,
    total_qty: float,
    depth: int,
) -> ImpactResult:
    """Static on-book VWAP using the **first** snapshot as if we crossed immediately.

    Parameters
    ----------
    df : pd.DataFrame
        Recording table (flat L2).
    side : {"buy","sell"}
        Execution side.
    total_qty : float
        Target base quantity to execute.
    depth : int
        Max levels to consume in the first row.

    Returns
    -------
    ImpactResult
        Summary for a one-shot crossing baseline.

    Why keep this?
    --------------
    Acts as a “now” baseline to compare TWAP performance vs crossing outright.
    """
    assert side in ("buy", "sell")
    assert total_qty > 0 and depth >= 1
    if len(df) == 0:
        return ImpactResult(side, total_qty, 0.0, None, 0.0, None, None, 0, [])

    row0 = df.iloc[0].to_dict()
    ladder = _row_ladder(row0, side=side, depth=depth)
    fills, filled = _walk_book(side, total_qty, ladder)
    vwap = _fills_vwap(fills)
    notional = sum(px * q for px, q in fills)

    mid_open = _mid_from_row(row0)
    mid_close = _mid_from_row(df.iloc[-1].to_dict())

    if vwap is None:
        s_open = s_close = None
    else:
        if side == "buy":
            s_open = _bps(vwap, mid_open)
            s_close = _bps(vwap, mid_close)
        else:
            s_open = _bps(mid_open, vwap)
            s_close = _bps(mid_close, vwap)

    return ImpactResult(
        side=side,
        target_qty=total_qty,
        filled_qty=filled,
        vwap=vwap,
        notional=notional,
        slippage_bps_vs_mid_open=s_open,
        slippage_bps_vs_mid_close=s_close,
        slices=1,
        fills=[
            {
                "slice": 1,
                "ts_ms": row0.get("ts_ms"),
                "iso": row0.get("iso"),
                "filled_qty": filled,
                "slice_target_qty": total_qty,
                "slice_vwap": vwap,
                "levels_touched": len(fills),
            }
        ],
    )
