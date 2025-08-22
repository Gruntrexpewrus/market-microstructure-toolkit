# Created by LP
# Date: 2025-08-22
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

# realistic twap simulator

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Tuple

import pandas as pd

# Local imports
from .setup_log import setup_logging

Side = Literal["buy", "sell"]
ScheduleMode = Literal["equal", "stochastic"]


@dataclass
class TWAPConfig:
    """
    Container for TWAP execution parameters.

    Notes
    -----
    - `fee_bps`: taker fee in basis points (5 = 0.05%). Applied to notional.
    - `depth_cap`: only consume top-K levels; None means "no cap"
    - `allow_residual`: if True, a slice may finish partially filled
    - `jitter_ms`: random uniform jitter in ±jitter_ms/2 around ideal slice times
    - `schedule`: 'equal' → equal sized slices; 'stochastic' → Dirichlet noise around equal
    - `seed`: RNG seed for reproducibility
    """

    side: Side
    target_qty: float
    slices: int
    fee_bps: float = 0.0
    depth_cap: int | None = None
    allow_residual: bool = False
    jitter_ms: int = 0
    schedule: ScheduleMode = "equal"
    seed: int | None = None


def _prefer_parquet(path: Path) -> bool:
    """Return True if input is parquet, else CSV."""
    return path.suffix.lower() in (".parquet", ".pq")


def _read_any(path: Path) -> pd.DataFrame:
    """Read CSV/Parquet into a DataFrame without changing columns."""
    if _prefer_parquet(path):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _select_book_side(
    df_row: pd.Series, side: Side, depth_cap: int | None
) -> List[Tuple[float, float]]:
    """
    Extract price/size ladder on the execution side of the book.

    Tricks
    ------
    - We parse the flat schema bid{i}_price/size or ask{i}_price/size in order.
    - Missing/blank entries are skipped.
    """
    ladder: List[Tuple[float, float]] = []
    prefix = "ask" if side == "buy" else "bid"
    i = 1
    while True:
        p_key = f"{prefix}{i}_price"
        s_key = f"{prefix}{i}_size"
        if p_key not in df_row or s_key not in df_row:
            break
        p, s = df_row[p_key], df_row[s_key]
        if p in ("", None) or s in ("", None):
            break
        try:
            price = float(p)
            size = float(s)
        except Exception:
            break
        if size <= 0:
            break
        ladder.append((price, size))
        i += 1
        if depth_cap is not None and len(ladder) >= depth_cap:
            break
    return ladder


def _apply_fee(notional: float, fee_bps: float, side: Side) -> float:
    """
    Apply taker fees to notional.

    For buys, you *pay* fee on the notional → effective spend increases.
    For sells, you *receive* notional net of fee → effective proceeds decrease.
    """
    fee = notional * (fee_bps / 1e4)
    return notional + fee if side == "buy" else max(0.0, notional - fee)


def _slice_sizes(cfg: TWAPConfig) -> List[float]:
    """
    Build slice sizes according to schedule.

    Tricks
    ------
    - 'equal' returns N equal parts (last slice adjusted to hit target exactly).
    - 'stochastic' draws weights from a Dirichlet to introduce realistic unevenness.
    """
    if cfg.slices <= 0:
        return []
    if cfg.schedule == "equal":
        per = cfg.target_qty / cfg.slices
        sizes = [per] * cfg.slices
        # adjust last slice to remove possible FP drift
        sizes[-1] = cfg.target_qty - sum(sizes[:-1])
        return sizes

    # stochastic: mildly noisy around equal with Dirichlet
    rng = random.Random(cfg.seed)
    alpha = [3.0] * cfg.slices  # higher alpha → closer to equal
    # Simple Dirichlet via Gamma draws
    draws = [rng.gammavariate(a, 1.0) for a in alpha]
    total = sum(draws)
    weights = [d / total for d in draws]
    return [cfg.target_qty * w for w in weights]


def _timestamp_schedule(
    ts_ms: Iterable[int], slices: int, jitter_ms: int, seed: int | None
) -> List[int]:
    """
    Pick slice timestamps across the input window, with optional jitter.

    Implementation
    --------------
    - Evenly spaced indexes across [0, len(ts)-1] → round to nearest row
    - Jitter: add ±jitter_ms/2 around chosen ts to avoid robotic timing
    - We then map the jittered timestamp back to the *nearest* row by absolute delta
    """
    if slices <= 0:
        return []
    ts = list(ts_ms)
    n = len(ts)
    if n == 0:
        return []

    idxs = [round(i * (n - 1) / max(1, slices - 1)) for i in range(slices)]
    chosen = [ts[i] for i in idxs]

    if jitter_ms <= 0:
        return chosen

    rng = random.Random(seed)
    half = jitter_ms / 2.0
    jittered: List[int] = []
    for t in chosen:
        dt = rng.uniform(-half, half)
        jittered.append(int(t + dt))
    return jittered


def _nearest_rows_for_times(df: pd.DataFrame, target_ts: List[int]) -> List[int]:
    """
    Map target timestamps to DataFrame row indices by nearest absolute delta.
    """
    if not target_ts:
        return []
    ts_arr = df["ts_ms"].astype("int64").to_numpy()
    out: List[int] = []
    j = 0
    for t in target_ts:
        # advance pointer to improve performance assuming monotone ts
        while j + 1 < len(ts_arr) and abs(ts_arr[j + 1] - t) <= abs(ts_arr[j] - t):
            j += 1
        out.append(j)
    return out


def simulate_twap(path: Path, cfg: TWAPConfig, per_slice_out: Path | None = None) -> Dict[str, Any]:
    """
    Simulate a TWAP execution against the flat L2 snapshot file.

    What it does
    ------------
    - Chooses `slices` timestamps across the recording (with optional jitter).
    - For each timestamp, takes the book *on your side* (asks for buy, bids for sell),
      optionally capped at `depth_cap`, and "walks" it until the slice size is filled.
    - If `allow_residual=False`, the slice is forced to fill by walking deeper
      (if the file has deeper levels). If `True`, the slice may be partially filled.
    - Fees (bps) applied to notional for effective VWAP.

    Limitations
    -----------
    - Slices do **not** deplete the book for subsequent timestamps (no feedback).
      Impact is measured *within each snapshot* by walking depth at that instant.
    """
    log = setup_logging(name="impact_twap")

    df = _read_any(path).copy()
    if "ts_ms" not in df.columns:
        raise ValueError("Input must contain 'ts_ms' column (millisecond timestamps).")

    if cfg.seed is not None:
        random.seed(cfg.seed)

    # Build slice sizes + their timestamps
    sizes = _slice_sizes(cfg)
    if not sizes:
        raise ValueError("No slices generated (check target_qty and slices).")

    target_times = _timestamp_schedule(df["ts_ms"].tolist(), cfg.slices, cfg.jitter_ms, cfg.seed)
    row_idxs = _nearest_rows_for_times(df, target_times)

    per_rows: List[Dict[str, Any]] = []
    total_filled = 0.0
    total_notional = 0.0  # pre-fee
    levels_agg = 0

    for k, (slice_qty, row_idx) in enumerate(zip(sizes, row_idxs), start=1):
        row = df.iloc[row_idx]
        ladder = _select_book_side(row, cfg.side, cfg.depth_cap)
        remain = slice_qty
        notional = 0.0
        touched = 0

        for price, size in ladder:
            if remain <= 0:
                break
            take = size if cfg.allow_residual else min(size, remain)
            # If allow_residual, we can take full size even if > remain (we'll cap below)
            take = min(take, remain)
            if take <= 0:
                continue
            notional += price * take
            remain -= take
            touched += 1
            # If we are not allowing residuals and ladder exhausted without filling → continue
        filled = slice_qty - remain
        total_filled += filled
        total_notional += notional
        levels_agg += touched

        eff_notional = _apply_fee(notional, cfg.fee_bps, cfg.side)
        slice_vwap = (eff_notional / filled) if filled > 0 else math.nan

        per_rows.append(
            {
                "slice": k,
                "ts_ms": int(row["ts_ms"]),
                "iso": row.get("iso", ""),
                "filled_qty": filled,
                "slice_target_qty": slice_qty,
                "slice_vwap": slice_vwap,
                "levels_touched": touched,
            }
        )

    # Summary
    eff_total_notional = _apply_fee(total_notional, cfg.fee_bps, cfg.side)
    vwap = (eff_total_notional / total_filled) if total_filled > 0 else math.nan
    mid_open = _row_mid(df.iloc[row_idxs[0]])
    mid_close = _row_mid(df.iloc[row_idxs[-1]])
    slip_open = _slip_bps(vwap, mid_open, cfg.side) if mid_open is not None else None
    slip_close = _slip_bps(vwap, mid_close, cfg.side) if mid_close is not None else None

    summary = {
        "side": cfg.side,
        "target_qty": cfg.target_qty,
        "filled_qty": total_filled,
        "vwap": vwap,
        "notional": eff_total_notional,
        "fee_bps": cfg.fee_bps,
        "depth_cap": cfg.depth_cap if cfg.depth_cap is not None else -1,
        "allow_residual": cfg.allow_residual,
        "slices": cfg.slices,
        "schedule": cfg.schedule,
        "slippage_bps_vs_mid_open": slip_open,
        "slippage_bps_vs_mid_close": slip_close,
        "levels_touched_avg": (levels_agg / max(1, cfg.slices)),
        "input": str(path),
    }

    if per_slice_out is not None:
        out_df = pd.DataFrame(per_rows)
        per_slice_out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(per_slice_out, index=False)
        log.info("Wrote per-slice detail → %s", per_slice_out)

    log.info(
        "TWAP done side=%s qty=%.6f filled=%.6f slices=%d vwap=%.6f fee_bps=%.2f cap=%s",
        cfg.side,
        cfg.target_qty,
        total_filled,
        cfg.slices,
        vwap,
        cfg.fee_bps,
        cfg.depth_cap,
    )
    return summary


def _row_mid(row: pd.Series) -> float | None:
    """Compute midprice from best bid/ask; return None if missing."""
    try:
        bb = float(row.get("best_bid"))
        ba = float(row.get("best_ask"))
        return (bb + ba) / 2.0
    except Exception:
        return None


def _slip_bps(exec_px: float, ref_mid: float | None, side: Side) -> float | None:
    """
    Slippage vs a reference mid in basis points.

    Definition
    ----------
    - buy: bps = 10_000 * (exec_px / ref_mid - 1)
    - sell: bps = 10_000 * (1 - exec_px / ref_mid)
    """
    if ref_mid is None or not math.isfinite(exec_px) or ref_mid <= 0:
        return None
    if side == "buy":
        return 1e4 * (exec_px / ref_mid - 1.0)
    return 1e4 * (1.0 - exec_px / ref_mid)


# --- Public API shims for library/tests --------------------------------------

__all__ = [
    "TWAPConfig",
    "TwapConfig",
    "simulate_twap",
    "simulate_twap_execution",
]

# CamelCase alias to match external expectations
TwapConfig = TWAPConfig  # noqa: N816


def simulate_twap_execution(df: pd.DataFrame, cfg: TWAPConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    In-memory TWAP simulator (no I/O). Mirrors `simulate_twap` but:
      - Takes a DataFrame instead of a file path
      - Returns (summary_df, per_slice_df) DataFrames

    It reuses the same helpers:
    _slice_sizes, _timestamp_schedule, _nearest_rows_for_times,
    _select_book_side, _apply_fee, _row_mid, _slip_bps.
    """
    if "ts_ms" not in df.columns:
        raise ValueError("DataFrame must contain 'ts_ms' column (millisecond timestamps).")

    sizes = _slice_sizes(cfg)
    if not sizes:
        raise ValueError("No slices generated (check target_qty and slices).")

    target_times = _timestamp_schedule(df["ts_ms"].tolist(), cfg.slices, cfg.jitter_ms, cfg.seed)
    row_idxs = _nearest_rows_for_times(df, target_times)

    per_rows: list[dict[str, Any]] = []
    total_filled = 0.0
    total_notional = 0.0
    levels_agg = 0

    for k, (slice_qty, row_idx) in enumerate(zip(sizes, row_idxs), start=1):
        row = df.iloc[row_idx]
        ladder = _select_book_side(row, cfg.side, cfg.depth_cap)
        remain = slice_qty
        notional = 0.0
        touched = 0

        for price, size in ladder:
            if remain <= 0:
                break
            take = size if cfg.allow_residual else min(size, remain)
            take = min(take, remain)
            if take <= 0:
                continue
            notional += price * take
            remain -= take
            touched += 1

        filled = slice_qty - remain
        total_filled += filled
        total_notional += notional
        levels_agg += touched

        eff_notional = _apply_fee(notional, cfg.fee_bps, cfg.side)
        slice_vwap = (eff_notional / filled) if filled > 0 else math.nan

        per_rows.append(
            {
                "slice": k,
                "ts_ms": int(row["ts_ms"]),
                "iso": row.get("iso", ""),
                "filled_qty": filled,
                "slice_target_qty": slice_qty,
                "slice_vwap": slice_vwap,
                "levels_touched": touched,
            }
        )

    eff_total_notional = _apply_fee(total_notional, cfg.fee_bps, cfg.side)
    vwap = (eff_total_notional / total_filled) if total_filled > 0 else math.nan
    mid_open = _row_mid(df.iloc[row_idxs[0]])
    mid_close = _row_mid(df.iloc[row_idxs[-1]])
    slip_open = _slip_bps(vwap, mid_open, cfg.side) if mid_open is not None else None
    slip_close = _slip_bps(vwap, mid_close, cfg.side) if mid_close is not None else None

    summary_df = pd.DataFrame(
        [
            {
                "side": cfg.side,
                "target_qty": cfg.target_qty,
                "filled_qty": total_filled,
                "vwap": vwap,
                "notional": eff_total_notional,
                "fee_bps": cfg.fee_bps,
                "depth_cap": cfg.depth_cap if cfg.depth_cap is not None else -1,
                "allow_residual": cfg.allow_residual,
                "slices": cfg.slices,
                "schedule": cfg.schedule,
                "slippage_bps_vs_mid_open": slip_open,
                "slippage_bps_vs_mid_close": slip_close,
                "levels_touched_avg": (levels_agg / max(1, cfg.slices)),
                "input": "",
            }
        ]
    )
    per_slice_df = pd.DataFrame.from_records(per_rows)
    return summary_df, per_slice_df
