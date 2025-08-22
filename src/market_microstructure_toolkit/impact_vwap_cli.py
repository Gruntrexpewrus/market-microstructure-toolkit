# Created by LP
# Date: 2025-08-22
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

from __future__ import annotations

import argparse
from pathlib import Path

from .impact_vwap import ExecConfig, _read_any, simulate_vwap_execution
from .setup_log import setup_logging

"""
Execution Simulator — VWAP (Volume-Weighted Average Price)

What this does
--------------
Simulates executing a parent order using a VWAP schedule: child order sizes
are allocated proportionally to a *volume proxy* per time bucket, then each
child fills against displayed book depth (taker) up to top-K.

Volume proxy (no trades data? no problem)
----------------------------------------
We approximate per-bucket activity by displayed depth on the passive side:
• --proxy l1_sum  → use L1 size (ask1_size for buys, bid1_size for sells)
• --proxy topk_sum → sum of sizes over top-K on the passive side (default)

Realism knobs
-------------
• --fee-bps: taker fee in bps.
• --depth: walk the book up to top-K.
• --min-slice-qty: floor per child (renormalized to target).
• Leftover carry: any unfilled portion rolls to the next slice.

Outputs
-------
1) Summary CSV: total filled, overall VWAP, notional, fees, slippage vs start/end mid.
2) Per-slice CSV: timestamp, allocated qty, filled qty, slice VWAP, levels touched.

Examples
--------
# VWAP buy of 50 ETH over 24 buckets, top-25 depth, 5 bps fee:
mmt-impact-vwap data/ETH_bybit_L2_60s.parquet \
  --side buy --target-qty 50 --slices 24 --depth 25 --fee-bps 5 --proxy topk_sum
"""


def main() -> None:
    p = argparse.ArgumentParser(description="VWAP execution simulator over historical books")
    p.add_argument("path", help="Input CSV/Parquet file from the recorder (flat schema)")
    p.add_argument("--side", choices=["buy", "sell"], required=True)
    p.add_argument(
        "--target-qty", type=float, required=True, help="Parent order quantity (base units)"
    )
    p.add_argument(
        "--slices", type=int, default=20, help="Number of child orders / time buckets (default: 20)"
    )
    p.add_argument("--depth", type=int, default=10, help="Walk book to top-K levels (default: 10)")
    p.add_argument(
        "--fee-bps", type=float, default=0.0, help="Taker fee in basis points (default: 0)"
    )
    p.add_argument(
        "--proxy",
        choices=["topk_sum", "l1_sum"],
        default="topk_sum",
        help="Per-bucket volume proxy (default: topk_sum)",
    )
    p.add_argument(
        "--min-slice-qty",
        type=float,
        default=None,
        help="Optional floor per child slice (renormalized to target)",
    )
    p.add_argument(
        "--outdir", default="", help="Directory to write CSVs (default: alongside input)"
    )
    p.add_argument("--log-name", default="impact_vwap", help="Logger name (default: impact_vwap)")
    args = p.parse_args()

    log = setup_logging(name=args.log_name)

    inp = Path(args.path).expanduser().resolve()
    df = _read_any(inp)

    cfg = ExecConfig(
        side=args.side,
        target_qty=args.target_qty,
        slices=args.slices,
        depth_k=args.depth,
        fee_bps=args.fee_bps,
        proxy=args.proxy,
        min_slice_qty=args.min_slice_qty,
    )

    summary_df, per_slice_df = simulate_vwap_execution(df, cfg)

    # output paths
    outdir = Path(args.outdir) if args.outdir else inp.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = inp.stem
    summary_path = outdir / f"{stem}_impact_vwap_summary.csv"
    slices_path = outdir / f"{stem}_impact_vwap_slices.csv"

    summary_df.to_csv(summary_path, index=False)
    per_slice_df.to_csv(slices_path, index=False)

    log.info("VWAP summary  → %s", summary_path)
    log.info("VWAP per-slice→ %s", slices_path)
    print(f"✅ VWAP summary  → {summary_path}")
    print(f"✅ VWAP per-slice→ {slices_path}")


if __name__ == "__main__":
    main()
