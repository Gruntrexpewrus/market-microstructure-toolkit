# Created by LP
# Date: 2025-08-22
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

# realist vwap

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .setup_log import setup_logging

Side = Literal["buy", "sell"]
Proxy = Literal["topk_sum", "topk_notional"]


@dataclass
class VwapConfig:
    """
    Configuration for the VWAP-style scheduler + depth-walking simulator.

    Notes
    -----
    - `depth_k`: top-K levels to look at when computing the proxy and when walking depth.
    - `fee_bps`: taker fee in basis points (5 = 0.05%).
    - `proxy`: how to estimate “volume profile” across time windows.
    - `timestamp_col` / `iso_col`: column names in the input frame.
    """

    side: Side
    target_qty: float
    slices: int
    fee_bps: float = 0.0
    proxy: Proxy = "topk_sum"
    depth_k: int = 10  # ← NEW: the code uses this internally
    timestamp_col: str = "ts_ms"
    iso_col: str = "iso"

    # Optional back-compat knobs; if someone passed these elsewhere, map them.
    depth: Optional[int] = field(default=None, repr=False)
    depth_cap: Optional[int] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Back-compat: if another caller set depth/depth_cap, prefer that over default
        if self.depth is not None:
            self.depth_k = int(self.depth)
        if self.depth_cap is not None:
            self.depth_k = int(self.depth_cap)


@dataclass
class ExecConfig:
    """
    Execution configuration for the VWAP simulator.

    Parameters
    ----------
    side : {"buy","sell"}
        Parent order direction.
    target_qty : float
        Total base-asset quantity to execute.
    slices : int
        Number of time buckets (child orders).
    depth_k : int
        Maximum book depth to walk (top-K levels).
    fee_bps : float
        Taker fee in basis points, added to buy cost / subtracted from sell proceeds.
    proxy : {"topk_sum","l1_sum"}
        Volume proxy:
        - "l1_sum": use L1 size on the relevant side (ask for buy, bid for sell).
        - "topk_sum": use sum of sizes over top-K on the relevant side.
    min_slice_qty : float | None
        Optional minimum child slice quantity (pre-carry), applied after proportional sizing.
    """

    side: Side
    target_qty: float
    slices: int = 20
    depth_k: int = 10
    fee_bps: float = 0.0
    proxy: Literal["topk_sum", "l1_sum"] = "topk_sum"
    min_slice_qty: float | None = None


def _read_any(path: Path) -> pd.DataFrame:
    """Read CSV or Parquet by extension, preserving column names as in the recorder."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input type: {path.suffix}")


def _extract_side_levels(row: pd.Series, side: Side, depth_k: int) -> List[Tuple[float, float]]:
    """
    Extract (price, size) pairs for the active side up to depth_k.

    Notes
    -----
    • For a BUY we cross the ASK side (ask1_price/ask1_size, ...).
    • For a SELL we cross the BID side (bid1_price/bid1_size, ...).
    • Missing / blank values are skipped.
    """
    levels: List[Tuple[float, float]] = []
    prefix = "ask" if side == "buy" else "bid"
    for i in range(1, depth_k + 1):
        p = row.get(f"{prefix}{i}_price")
        q = row.get(f"{prefix}{i}_size")
        if p in ("", None) or q in ("", None):
            continue
        try:
            pf = float(p)
            qf = float(q)
        except Exception:
            continue
        if qf <= 0:
            continue
        levels.append((pf, qf))
    return levels


def _mid_from_row(row: pd.Series) -> float | None:
    """Compute midquote from best bid/ask if available."""
    bb = row.get("best_bid")
    ba = row.get("best_ask")
    try:
        bbf = float(bb)
        baf = float(ba)
        if bb is None or ba is None:
            return None
        return 0.5 * (bbf + baf)
    except Exception:
        return None


def _proxy_for_row(row: pd.Series, side: Side, depth_k: int, kind: str) -> float:
    """
    Volume proxy per row used to allocate child sizes.

    Heuristics
    ----------
    • "l1_sum": use L1 displayed size on the *passive* side we cross
        - buy → ask1_size
        - sell → bid1_size
    • "topk_sum": sum of displayed sizes across top-K on the passive side.
    """
    levels = _extract_side_levels(row, side, depth_k)
    if not levels:
        return 0.0
    if kind == "l1_sum":
        return levels[0][1]
    # default → topk_sum
    return float(sum(sz for _, sz in levels))


def _allocate_child_sizes(df: pd.DataFrame, cfg: ExecConfig) -> List[float]:
    """
    Allocate child order quantities across the chosen time buckets, proportional to
    the volume proxy in each bucket. Remainder is distributed greedily to largest buckets.

    Returns
    -------
    List[float]
        Child sizes (length == cfg.slices). May contain zeros if proxy=0 for some bucket.
    """
    # pick evenly spaced rows over the dataset to get 'slices' time buckets
    if cfg.slices <= 0:
        raise ValueError("slices must be positive")
    idx = (pd.Index(range(len(df))) * (cfg.slices / len(df))).round().astype(int)
    # robust clip (works across pandas versions)
    idx = np.clip(idx.to_numpy(), 0, len(df) - 1).tolist()
    # if dataset is short, ensure we still produce exactly cfg.slices buckets by sampling with replacement at the tail
    while len(idx) < cfg.slices:
        idx = idx.append(pd.Index([len(df) - 1]))
    idx = idx[: cfg.slices]

    proxies = [max(0.0, _proxy_for_row(df.iloc[i], cfg.side, cfg.depth_k, cfg.proxy)) for i in idx]
    total_proxy = sum(proxies)
    if total_proxy <= 0:
        # fallback: equal sizing if proxy is flat/zero
        base = cfg.target_qty / cfg.slices
        sizes = [base] * cfg.slices
    else:
        sizes = [cfg.target_qty * (w / total_proxy) for w in proxies]

    # enforce optional min_slice_qty
    if cfg.min_slice_qty is not None:
        sizes = [max(s, float(cfg.min_slice_qty)) for s in sizes]
        # re-normalize to target
        scale = cfg.target_qty / sum(sizes)
        sizes = [s * scale for s in sizes]

    # numerical cleanup so the sum is exact
    err = cfg.target_qty - sum(sizes)
    if abs(err) > 1e-9:
        # adjust the largest bucket to absorb rounding drift
        j = int(max(range(len(sizes)), key=lambda k: sizes[k]))
        sizes[j] += err
    return sizes


def _fill_slice(levels: List[Tuple[float, float]], desired_qty: float) -> Tuple[float, float, int]:
    """
    Walk the provided price levels to fill 'desired_qty'.

    Returns
    -------
    filled_qty : float
        Quantity actually filled (may be < desired if the book is thin).
    vwap : float
        Volume-weighted price achieved for the filled part only.
    levels_touched : int
        How many levels were consumed (1..K). 0 if nothing filled.
    """
    if desired_qty <= 0 or not levels:
        return 0.0, math.nan, 0

    remain = desired_qty
    cost = 0.0
    levels_touched = 0

    for px, avail in levels:
        if remain <= 0:
            break
        take = min(remain, avail)
        cost += take * float(px)
        remain -= take
        levels_touched += 1 if take > 0 else 0

    filled = desired_qty - remain
    if filled <= 0:
        return 0.0, math.nan, 0

    vwap = cost / filled
    return filled, vwap, levels_touched


def simulate_vwap_execution(
    df: pd.DataFrame,
    cfg: ExecConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate VWAP execution over 'cfg.slices' time buckets.

    Strategy
    --------
    1) Choose 'slices' time buckets over the dataset (evenly spaced indices).
    2) Compute a volume proxy per bucket (top-K or L1 depth on the passive side).
    3) Allocate child sizes ∝ proxy weights.
    4) For each slice, walk the passive side at that bucket up to depth-K.
       Any unfilled remainder is carried into the next slice.

    Returns
    -------
    summary_df, per_slice_df
    """
    log = setup_logging(name="impact_vwap")

    if len(df) == 0:
        raise ValueError("Empty dataframe")

    # choose slice indices (even spread over the entire period)
    if cfg.slices > len(df):
        log.warning("slices > rows; sampling with replacement at the tail")
    idx = (pd.Index(range(len(df))) * (cfg.slices / len(df))).round().astype(int)
    # robust clip (works across pandas versions)
    idx = np.clip(idx.to_numpy(), 0, len(df) - 1).tolist()
    while len(idx) < cfg.slices:
        idx.append(len(df) - 1)
    idx = idx[: cfg.slices]

    # sizes per bucket using proxy
    child_sizes = _allocate_child_sizes(df, cfg)

    # for slippage references
    mid_open = _mid_from_row(df.iloc[idx[0]])
    mid_close = _mid_from_row(df.iloc[idx[-1]])

    # simulate
    filled_total = 0.0
    notional_total = 0.0
    fee_total = 0.0
    per_rows: List[Dict[str, Any]] = []

    carry = 0.0  # remainder rolling forward

    for s, (row_i, slice_target) in enumerate(zip(idx, child_sizes), start=1):
        row = df.iloc[row_i]
        ts_ms = int(row.get("ts_ms", 0)) if pd.notna(row.get("ts_ms", 0)) else 0
        iso = str(row.get("iso", ""))

        # child desired includes any carry
        desired = float(slice_target) + carry

        # extract the passive side
        levels = _extract_side_levels(row, cfg.side, cfg.depth_k)

        # BUY must consume asks in ascending price; SELL bids in descending.
        # Our extractor already returns in the natural order recorded by the file:
        # ask1..askK are ascending; bid1..bidK are descending best→worse (as recorded).
        # If your recorder flips order, sort here accordingly.

        filled, vwap, levels_touched = _fill_slice(levels, desired)

        # fees
        # buy → add fees to notional cost; sell → subtract from proceeds
        notional = filled * vwap if math.isfinite(vwap) else 0.0
        fee = abs(notional) * (cfg.fee_bps / 10_000.0)
        if cfg.side == "buy":
            notional_with_fee = notional + fee
        else:
            notional_with_fee = notional - fee

        filled_total += filled
        notional_total += notional_with_fee
        fee_total += fee

        # compute slice mid and slippage for diagnostics
        mid = _mid_from_row(row)
        slip_bps = None
        if mid and filled > 0:
            # positive bps = paid above mid if buy; earned above mid if sell
            signed = (vwap - mid) / mid * 10_000.0
            slip_bps = float(signed if cfg.side == "buy" else -signed)

        # carry remainder forward
        carry = max(0.0, desired - filled)

        per_rows.append(
            {
                "slice": s,
                "ts_ms": ts_ms,
                "iso": iso,
                "slice_target_qty": float(slice_target),
                "desired_qty_incl_carry": desired,
                "filled_qty": filled,
                "slice_vwap": float(vwap) if math.isfinite(vwap) else "",
                "levels_touched": levels_touched,
                "mid": "" if mid is None else float(mid),
                "slippage_bps_vs_mid": "" if slip_bps is None else slip_bps,
                "carry_to_next": carry,
            }
        )

    # summary
    vwap_all = (notional_total / filled_total) if filled_total > 0 else math.nan

    def _slip_vs(ref_mid: float | None) -> float | str:
        if ref_mid is None or not math.isfinite(vwap_all):
            return ""
        signed = (vwap_all - ref_mid) / ref_mid * 10_000.0
        return float(signed if cfg.side == "buy" else -signed)

    summary = {
        "side": cfg.side,
        "target_qty": cfg.target_qty,
        "filled_qty": filled_total,
        "vwap": float(vwap_all) if math.isfinite(vwap_all) else "",
        "notional": float(notional_total),
        "fees_paid": float(fee_total),
        "slippage_bps_vs_mid_open": _slip_vs(mid_open),
        "slippage_bps_vs_mid_close": _slip_vs(mid_close),
        "slices": cfg.slices,
        "depth_k": cfg.depth_k,
        "proxy": cfg.proxy,
    }

    return pd.DataFrame([summary]), pd.DataFrame(per_rows)
