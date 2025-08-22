# Created by LP
# Date: 2025-08-22
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .advanced_plots import run_advanced_plots
from .setup_log import setup_logging


def _default_outdir(inp: Path) -> Path:
    # keep it obvious and consistent
    return Path("plots") / f"{inp.stem}_adv"


def main() -> None:
    """CLI entry point for advanced visualization & analytics."""
    p = argparse.ArgumentParser(
        description="Advanced visualization & analytics for order-book recordings"
    )
    p.add_argument("path", help="Input CSV or Parquet file (flat schema)")
    p.add_argument(
        "--outdir", default="", help="Directory to save PNGs (default: plots/<stem>_adv/)"
    )
    p.add_argument(
        "--no-save", action="store_true", help="Do not save PNGs; show interactively instead"
    )
    p.add_argument("--depth", type=int, default=10, help="Top-K depth to use for depth plots")
    p.add_argument("--rv-window", type=int, default=100, help="Event-time RV window")
    p.add_argument(
        "--corr-window", type=int, default=100, help="Rolling correlation window for OFI vs returns"
    )
    p.add_argument(
        "--tail", type=int, default=500, help="Use only the last N rows (speed up plotting)"
    )
    p.add_argument("--log-name", default="plot_adv", help="Logger name (default: plot_adv)")
    args = p.parse_args()

    # Standard logging init (matches the rest of your toolkit)
    setup_logging(name=args.log_name)
    log = logging.getLogger(args.log_name)

    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        log.error("Input not found: %s", path)
        raise SystemExit(2)

    # Decide output directory (also used for logging below)
    user_outdir = Path(args.outdir).expanduser().resolve() if args.outdir else None
    final_outdir = user_outdir or _default_outdir(path)

    log.info(
        "Starting advanced plots | input=%s save=%s outdir=%s depth=%d rv_window=%d corr_window=%d tail=%d",
        path,
        (not args.no_save),
        final_outdir,
        args.depth,
        args.rv_window,
        args.corr_window,
        args.tail,
    )

    # run_advanced_plots returns the directory it wrote to (Path) or None if not saving.
    returned_outdir = run_advanced_plots(
        path=path,
        outdir=user_outdir,  # pass through user choice; function will default if None
        save=not args.no_save,
        depth_k=args.depth,
        rv_window=args.rv_window,
        corr_window=args.corr_window,
        tail=args.tail,
    )

    if not args.no_save:
        # Prefer what the function returns; fall back to our computed default.
        outdir_to_report = returned_outdir or final_outdir
        log.info("Saved plots → %s", outdir_to_report)
        print(f"📁 Saved plots → {outdir_to_report}")
    else:
        log.info("Interactive display requested (no files saved).")


if __name__ == "__main__":
    main()

# usage
# mmt-plot-adv data/ETH_bybit_L2_60s.parquet --depth 25 --rv-window 200 --tail 2000
# → PNGs saved to: plots/ETH_bybit_L2_60s_adv/
