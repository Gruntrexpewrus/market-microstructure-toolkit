# Created by LP
# Date: 2025-08-22
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3
"""
Execution Simulator — TWAP (Time-Weighted Average Price)

What this does
--------------
Simulates splitting a large parent order into equal child orders (slices) and
executing them sequentially against historical order-book snapshots. Each slice
consumes displayed liquidity (taker), computes slice VWAP, levels touched, and
slippage vs mid.

TWAP vs VWAP
------------
• TWAP: equal-sized slices at fixed time steps (ignores actual traded flow).
• VWAP: sizes adapt to market volume (or a volume proxy), trading more when
  the market is active.

Realism knobs
-------------
• --fee-bps: taker fee in basis points (applied per slice).
• Depth-K filling: walk the book across top-K levels.
• Carry leftover: if a slice can’t fully fill, the remainder rolls to the next slice.

Outputs
-------
1) Summary CSV: total filled, overall VWAP, notional, slippage vs start/end mid.
2) Per-slice CSV: timestamp, filled qty, slice VWAP, levels touched.

Example
-------
mmt-impact-twap data/ETH_bybit_L2_60s.parquet \
  --side buy --target-qty 50 --slices 20 --depth 25 --fee-bps 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .impact_twap import TWAPConfig, simulate_twap
from .setup_log import setup_logging


def _write_summary_csv(summary: dict, out: Path) -> None:
    """Write one-line summary CSV (create header if new)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([summary])
    header = not out.exists()
    df.to_csv(out, index=False, mode="a", header=header)


def main() -> None:
    """CLI to run a TWAP impact simulation with realistic knobs."""
    p = argparse.ArgumentParser(
        description=__doc__,  # uses the docstring above
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("path", help="Input CSV/Parquet (flat schema with L2 levels)")
    p.add_argument("--side", choices=["buy", "sell"], default="buy")
    p.add_argument("--target-qty", type=float, required=True, help="Total quantity to execute")
    p.add_argument("--slices", type=int, default=20, help="Number of child orders (default 20)")
    p.add_argument(
        "--schedule", choices=["equal", "stochastic"], default="equal", help="Child sizing rule"
    )
    p.add_argument(
        "--fee-bps", type=float, default=0.0, help="Taker fee in basis points (e.g., 5 = 0.05%)"
    )
    p.add_argument(
        "--depth-cap", type=int, default=-1, help="Only consume top-K levels; -1 for no cap"
    )
    p.add_argument(
        "--allow-residual", action="store_true", help="Allow unfilled remainder for a slice"
    )
    p.add_argument(
        "--jitter-ms", type=int, default=0, help="Uniform jitter around slice times (±jitter/2)"
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument(
        "--summary-out", default="", help="Optional CSV path to append the run-level summary"
    )
    p.add_argument(
        "--per-slice-out", default="", help="Optional CSV path for detailed child order fills"
    )
    p.add_argument("--log-name", default="impact_twap", help="Logger name (default: impact_twap)")
    args = p.parse_args()

    setup_logging(name=args.log_name)

    depth_cap = None if args.depth_cap is None or args.depth_cap < 1 else int(args.depth_cap)

    cfg = TWAPConfig(
        side=args.side,  # type: ignore[arg-type]
        target_qty=float(args.target_qty),
        slices=int(args.slices),
        fee_bps=float(args.fee_bps),
        depth_cap=depth_cap,
        allow_residual=bool(args.allow_residual),
        jitter_ms=int(args.jitter_ms),
        schedule=args.schedule,  # type: ignore[arg-type]
        seed=args.seed,
    )

    per_slice_path = Path(args.per_slice_out).expanduser().resolve() if args.per_slice_out else None
    summary = simulate_twap(
        Path(args.path).expanduser().resolve(), cfg, per_slice_out=per_slice_path
    )

    if args.summary_out:
        _write_summary_csv(summary, Path(args.summary_out).expanduser().resolve())

    # Human-friendly echo:
    print(
        f"✅ TWAP summary — side={summary['side']} qty={summary['target_qty']} "
        f"filled={summary['filled_qty']:.6f} vwap={summary['vwap']:.6f} "
        f"slip_open_bps={summary['slippage_bps_vs_mid_open']:.3f} "
        f"slip_close_bps={summary['slippage_bps_vs_mid_close']:.3f}"
    )


if __name__ == "__main__":
    main()
