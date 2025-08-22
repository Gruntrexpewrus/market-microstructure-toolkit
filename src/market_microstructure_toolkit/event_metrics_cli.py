# Created by LP
# Date: 2025-08-22
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

# event metrics cli
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .event_metrics import compute_event_time_metrics
from .setup_log import setup_logging

log = logging.getLogger(__name__)  # handlers configured in main()


def _read_any(path: Path) -> pd.DataFrame:
    """Read CSV or Parquet by extension."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input type: {ext}")


def _default_out_path(inp: Path, rv_window: int) -> Path:
    """Default: <input_basename>_evtmetrics_w<rv>.csv (next to the input)."""
    base = inp.with_suffix("")  # drop .csv/.parquet
    return Path(f"{base}_evtmetrics_w{rv_window}.csv")


def main() -> None:
    """CLI entrypoint: compute event-time (per-update) metrics and save CSV."""
    parser = argparse.ArgumentParser(
        description="Compute event-time (per-update) metrics from an order-book snapshot file."
    )
    parser.add_argument(
        "input",
        help="Input CSV/Parquet produced by `record` or `ws_record` (flat schema).",
    )
    parser.add_argument(
        "--rv-window",
        type=int,
        default=20,
        help="Event-time window (updates) for realized variance (default: 20).",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output CSV path (default: '<input>_evtmetrics_w<rv>.csv').",
    )
    args = parser.parse_args()

    # initialize project logging
    setup_logging(name="event_metrics_cli")

    try:
        inp = Path(args.input).expanduser().resolve()
        out = (
            Path(args.out).expanduser().resolve()
            if args.out
            else _default_out_path(inp, args.rv_window)
        )

        log.info("Event-time metrics start: input=%s rv_window=%d out=%s", inp, args.rv_window, out)

        df = _read_any(inp)
        log.info("Loaded input: rows=%d, cols=%d", len(df), len(df.columns))

        df = compute_event_time_metrics(df, rv_window=args.rv_window)

        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

        log.info("Wrote event-time metrics → %s (rows=%d, cols=%d)", out, len(df), len(df.columns))
        # a friendly stdout line for the user
        print(f"✅ Wrote event-time metrics → {out}")

    except Exception:
        log.exception("Event-time metrics failed")
        raise


if __name__ == "__main__":
    main()
