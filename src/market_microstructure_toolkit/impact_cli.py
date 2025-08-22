# Created by LP
# Date: 2025-08-22
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

# This is a book that is static and only walks the book, other codes have realistic settings
# like fees etc., see impact_twap.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .impact import simulate_twap, simulate_vwap_onbook
from .setup_log import setup_logging


def _read_any(path: Path) -> pd.DataFrame:
    """Load CSV or Parquet by extension (raises on unknown types)."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input type: {ext}")


def _default_outdir(inp: Path, tag: str) -> Path:
    """Build a neat output path for results, namespaced under `impact/`.

    Implementation trick
    --------------------
    We strip the extension and attach a readable tag describing the simulation
    (e.g., `buy_twap_s20_q50_d25`) to make folders self-describing.
    """
    base = inp.with_suffix("")  # drop .csv/.parquet
    return Path(f"impact/{base.name}_{tag}")


def main() -> None:
    """CLI entry point: run TWAP or baseline VWAP on-book and write CSV outputs.

    Writes
    ------
    - `slices.csv`  : per-slice fills summary
    - `summary.csv` : global result summary (VWAP, slippage, notional, etc.)

    Logging
    -------
    Uses your standard `setup_logging` (default logger name: `impact_cli`).
    """
    p = argparse.ArgumentParser(
        description="Naive TWAP/VWAP simulators that walk recorded L2 books."
    )
    p.add_argument("input", help="Input CSV/Parquet produced by record.py or ws_record.py")
    p.add_argument("--side", choices=["buy", "sell"], required=True)
    p.add_argument("--qty", type=float, required=True, help="Total base quantity to execute")
    p.add_argument("--depth", type=int, default=10, help="Max book depth to walk (default 10)")

    # Mutually exclusive modes
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--twap-slices", type=int, help="Number of equal time slices (TWAP)")
    g.add_argument("--vwap-now", action="store_true", help="One-shot on-book VWAP using first row")

    # I/O + logging
    p.add_argument("--outdir", default="", help="Output folder (default: impact/<stem>_<mode>/)")
    p.add_argument("--log-name", default="impact_cli", help="Logger name (default: impact_cli)")
    args = p.parse_args()

    log = setup_logging(name=args.log_name)

    inp = Path(args.input).expanduser().resolve()
    if not inp.exists():
        log.error("Input file not found: %s", inp)
        raise SystemExit(2)

    df = _read_any(inp)
    if len(df) == 0:
        log.error("Empty input: %s", inp)
        raise SystemExit(2)

    # Choose mode and tag
    if args.vwap_now:
        tag = f"{args.side}_vwapnow_q{args.qty:g}_d{args.depth}"
        res = simulate_vwap_onbook(df, side=args.side, total_qty=args.qty, depth=args.depth)
    else:
        tag = f"{args.side}_twap_s{args.twap_slices}_q{args.qty:g}_d{args.depth}"
        res = simulate_twap(
            df, side=args.side, total_qty=args.qty, slices=args.twap_slices, depth=args.depth
        )

    outdir = Path(args.outdir) if args.outdir else _default_outdir(inp, tag)
    outdir.mkdir(parents=True, exist_ok=True)

    # Per-slice rollup
    rollup_path = outdir / "slices.csv"
    pd.DataFrame(res.fills).to_csv(rollup_path, index=False)

    # Summary
    summary = {
        "side": res.side,
        "target_qty": res.target_qty,
        "filled_qty": res.filled_qty,
        "vwap": res.vwap,
        "notional": res.notional,
        "slippage_bps_vs_mid_open": res.slippage_bps_vs_mid_open,
        "slippage_bps_vs_mid_close": res.slippage_bps_vs_mid_close,
        "slices": res.slices,
        "input": str(inp),
    }
    summary_path = outdir / "summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    log.info("Wrote slice rollup  → %s", rollup_path)
    log.info("Wrote summary      → %s", summary_path)
    print(f"✅ VWAP/TWAP results saved in: {outdir}")


if __name__ == "__main__":
    main()

# TWAP: buy 50 across 20 slices, walk 25 levels
# mmt-impact data/ETH_bybit_L2_60s.parquet --side buy --qty 50 --twap-slices 20 --depth 25

# VWAP-now: sell 25 crossing the first snapshot
# mmt-impact data/ETH_bybit_L2_60s.parquet --side sell --qty 25 --vwap-now --depth 50
