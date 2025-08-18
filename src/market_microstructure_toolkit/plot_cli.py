# Created by LP
# Date: 2025-08-18
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Iterable, Optional
import math


def _load_rows(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield rows as dicts from Parquet (preferred) or CSV."""
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd

            df = pd.read_parquet(path)
            for _, r in df.iterrows():
                # replace NaNs with None to match CSV path behavior
                yield {
                    k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                    for k, v in r.to_dict().items()
                }
            return
        except Exception:
            pass  # fall through to CSV

    import csv

    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            yield r


def _to_dataframe(rows: Iterable[Dict[str, Any]]):
    import pandas as pd

    rows = list(rows)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # timestamps
    if "ts_ms" in df.columns:
        df["ts_ms"] = pd.to_numeric(df["ts_ms"], errors="coerce")
        df = df.dropna(subset=["ts_ms"])
        df["ts_ms"] = df["ts_ms"].astype("int64")
        df = df.sort_values("ts_ms").reset_index(drop=True)
        df["t"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert("UTC")

    # coerce best prices to numeric (CSV path may be strings)
    for c in ("best_bid", "best_ask"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # also coerce L1 sizes if present (needed for microprice/OFI)
    for c in ("bid1_size", "ask1_size"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _sum_depth_sizes(df, side: str, K: int):
    """Sum of sizes across the top-K for one side."""
    import pandas as pd

    cols = [
        f"{side}{i}_size" for i in range(1, K + 1) if f"{side}{i}_size" in df.columns
    ]
    if not cols:
        return None
    S = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in cols})
    return S.sum(axis=1)


def _sum_depth_notional(df, side: str, K: int):
    """Sum of price*size across the top-K for one side."""
    import pandas as pd

    acc = None
    for i in range(1, K + 1):
        pcol, scol = f"{side}{i}_price", f"{side}{i}_size"
        if pcol in df.columns and scol in df.columns:
            p = pd.to_numeric(df[pcol], errors="coerce")
            q = pd.to_numeric(df[scol], errors="coerce")
            term = p * q
            acc = term if acc is None else (acc + term)
    return acc  # may be None if no columns


def _compute_metrics(df, depth: int):
    """
    Add: spread, mid, imbalance_l1, imbalance_k, relative_spread_bps,
         microprice, OFI(L1), cumulative OFI, RV(20).
    Uses vectorized formulas to work on pandas Series.
    """
    from .metrics import (
        compute_row_metrics,  # per-row dict (spread, mid, imbalance_l1, imbalance_k)
        realized_var,  # vector-friendly
        # relative_spread_bps, # we’ll compute vectorized inline for robustness
    )
    import pandas as pd
    import numpy as np

    # per-row metrics (spread, mid, imbalance_l1, imbalance_k) using your existing helper
    mdf = df.apply(
        lambda r: compute_row_metrics(r.to_dict(), depth),
        axis=1,
        result_type="expand",
    )
    df = pd.concat([df.reset_index(drop=True), mdf], axis=1)

    # Ensure numeric Series
    def num(s):
        return pd.to_numeric(s, errors="coerce")

    bb = num(df.get("best_bid"))
    ba = num(df.get("best_ask"))
    b1 = num(df.get("bid1_size"))
    a1 = num(df.get("ask1_size"))

    # relative spread (bps) = 20000 * (ask - bid) / (ask + bid)
    with np.errstate(invalid="ignore", divide="ignore"):
        denom_mid = ba + bb
        rel_bps = 20000.0 * (ba - bb) / denom_mid
        rel_bps[(denom_mid <= 0) | denom_mid.isna()] = np.nan
    df["relative_spread_bps"] = rel_bps

    # microprice (vectorized): (ask*bidSize + bid*askSize) / (bidSize + askSize)
    if bb is not None and ba is not None and b1 is not None and a1 is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            denom_sz = b1 + a1
            micro = (ba * b1 + bb * a1) / denom_sz
            micro[(denom_sz <= 0) | denom_sz.isna()] = np.nan
        df["microprice"] = micro

    # OFI (L1) vectorized, following LOBSTER / Cont et al. style:
    # OFI_t = 1{p^b_t > p^b_{t-1}} * q^b_t - 1{p^b_t < p^b_{t-1}} * q^b_{t-1}
    #       - 1{p^a_t > p^a_{t-1}} * q^a_{t-1} + 1{p^a_t < p^a_{t-1}} * q^a_t
    if bb is not None and ba is not None and b1 is not None and a1 is not None:
        bb_prev = bb.shift(1)
        ba_prev = ba.shift(1)
        b1_prev = b1.shift(1)
        a1_prev = a1.shift(1)

        up_bid = (bb > bb_prev).astype(float)
        dn_bid = (bb < bb_prev).astype(float)
        up_ask = (ba > ba_prev).astype(float)
        dn_ask = (ba < ba_prev).astype(float)

        ofi = up_bid * b1 - dn_bid * b1_prev - up_ask * a1_prev + dn_ask * a1
        ofi = ofi.fillna(0.0)  # first row / missing prev as zero impact
        df["ofi_l1"] = ofi
        df["ofi_l1_cum"] = ofi.cumsum()

    # realized variance on mid (rolling 20, log)
    if "mid" in df.columns:
        df["rv_20"] = realized_var(df["mid"], window=20, use_log=True)

        # ---- Depth-K Size OFI ----
    K = min(depth, 10)  # pick a default window for plotting
    Bsz = _sum_depth_sizes(df, "bid", K)
    Asz = _sum_depth_sizes(df, "ask", K)
    if Bsz is not None and Asz is not None:
        ofi_k = Bsz.diff().fillna(0.0) - Asz.diff().fillna(0.0)
        df[f"ofi_k{K}_size"] = ofi_k
        df[f"ofi_k{K}_size_cum"] = ofi_k.cumsum()

    # ---- Depth-K Notional OFI ----
    Bnot = _sum_depth_notional(df, "bid", K)
    Anot = _sum_depth_notional(df, "ask", K)
    if Bnot is not None and Anot is not None:
        ofi_k_not = Bnot.diff().fillna(0.0) - Anot.diff().fillna(0.0)
        df[f"ofi_k{K}_notional"] = ofi_k_not
        df[f"ofi_k{K}_notional_cum"] = ofi_k_not.cumsum()

    return df


def _plot(df, out: Optional[Path] = None):
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Decide base output folder (if saving)
    save_dir = None
    if out:
        dataset_name = out.stem  # e.g. "ETH_bybit_L2_60s"
        save_dir = Path("plots") / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Mid vs Microprice
    if {"t", "mid"} <= set(df.columns) or {"t", "microprice"} <= set(df.columns):
        plt.figure(figsize=(12, 4))
        if "mid" in df.columns:
            plt.plot(df["t"], df["mid"], label="Mid")
        if "microprice" in df.columns:
            plt.plot(df["t"], df["microprice"], label="Microprice")
        plt.title("Mid vs Microprice")
        plt.legend()
        if save_dir:
            plt.savefig(save_dir / "mid_micro.png", bbox_inches="tight")

    # 2) Spread (bps)
    if {"t", "relative_spread_bps"} <= set(df.columns):
        plt.figure(figsize=(12, 3))
        plt.plot(df["t"], df["relative_spread_bps"])
        plt.title("Relative Spread (bps)")
        if save_dir:
            plt.savefig(save_dir / "spread_bps.png", bbox_inches="tight")

    # 3) OFI (per step)
    if {"t", "ofi_l1"} <= set(df.columns):
        plt.figure(figsize=(12, 3))
        plt.plot(df["t"], df["ofi_l1"])
        plt.title("OFI (L1)")
        if save_dir:
            plt.savefig(save_dir / "ofi.png", bbox_inches="tight")

    # 4) OFI cumulative
    if {"t", "ofi_l1_cum"} <= set(df.columns):
        plt.figure(figsize=(12, 3))
        plt.plot(df["t"], df["ofi_l1_cum"])
        plt.title("OFI (L1) — cumulative")
        if save_dir:
            plt.savefig(save_dir / "ofi_cum.png", bbox_inches="tight")

    # 5) Realized variance (rolling)
    if {"t", "rv_20"} <= set(df.columns):
        plt.figure(figsize=(12, 3))
        plt.plot(df["t"], df["rv_20"], label="RV(20)")
        plt.title("Realized Variance (rolling)")
        plt.legend()
        if save_dir:
            plt.savefig(save_dir / "rv.png", bbox_inches="tight")

    # 6) Depth-K OFI (size & notional)
    for tag in ("size", "notional"):
        col = f"ofi_k10_{tag}"
        col_cum = f"{col}_cum"
        if col in df.columns:
            plt.figure(figsize=(12, 3))
            plt.plot(df["t"], df[col])
            plt.title(f"Depth-K OFI (K=10, {tag})")
            if save_dir:
                plt.savefig(save_dir / f"ofi_k10_{tag}.png", bbox_inches="tight")
        if col_cum in df.columns:
            plt.figure(figsize=(12, 3))
            plt.plot(df["t"], df[col_cum])
            plt.title(f"Depth-K OFI (K=10, {tag}) — cumulative")
            if save_dir:
                plt.savefig(save_dir / f"ofi_k10_{tag}_cum.png", bbox_inches="tight")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot metrics from a recorded CSV/Parquet."
    )
    parser.add_argument(
        "input", type=Path, help="Path to CSV or Parquet file recorded by the toolkit"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=50,
        help="Depth used when recording (for imbalance_k)",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save PNGs next to the input file"
    )
    args = parser.parse_args()

    rows = _load_rows(args.input)
    df = _to_dataframe(rows)
    if df.empty:
        print("No rows found.")
        return

    df = _compute_metrics(df, depth=args.depth)
    _plot(df, out=(args.input if args.save else None))


if __name__ == "__main__":
    main()
