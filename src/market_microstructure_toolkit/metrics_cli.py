# Created by LP
# Date: 2025-08-17
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

from __future__ import annotations

import sys
from pathlib import Path

from .metrics import compute_row_metrics, ofi_l1, rolling_realized_variance


def _prefer_parquet(p: Path) -> bool:
    return p.suffix.lower() == ".parquet"


def _load_rows(path: Path):
    if _prefer_parquet(path):
        try:
            import pandas as pd

            df = pd.read_parquet(path)
            for _, r in df.iterrows():
                yield {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
            return
        except Exception:
            pass
    import csv

    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            yield r


def _write_csv(rows, header, out_path: Path):
    import csv

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    import argparse

    from .setup_log import setup_logging

    parser = argparse.ArgumentParser(description="Add metrics to a recorded CSV/Parquet.")
    parser.add_argument("input_file", help="Input CSV/Parquet from record.py")
    parser.add_argument("depth", type=int, help="Top-K depth to use for imbalance/notional")
    parser.add_argument(
        "--rv-window", type=int, default=20, help="Rolling RV window length (in rows)"
    )
    args = parser.parse_args()

    log = setup_logging(name="metrics_cli")
    in_path = Path(args.input_file)
    assert in_path.exists(), f"Input not found: {in_path}"

    out_path = in_path.with_name(in_path.stem + "_metrics.csv")

    # Stream all rows → keep a list to compute series metrics
    rows = list(_load_rows(in_path))
    if not rows:
        log.warning("No rows found in input.")
        sys.exit(0)

    # 1) per-row metrics
    enriched = []
    for r in rows:
        m = compute_row_metrics(r, depth=args.depth)
        # string-format floats for stability; keep None as "" in CSV
        formatted = {
            k: ("" if v is None else f"{v:.10f}" if isinstance(v, float) else v)
            for k, v in m.items()
        }
        enriched.append({**r, **formatted})

    # 2) series metrics: rolling RV based on 'mid'
    mids = []
    for r in enriched:
        try:
            mids.append(float(r["mid"]) if r["mid"] != "" else None)
        except Exception:
            mids.append(None)
    rv = rolling_realized_variance(mids, window=args.rv_window)

    # 3) series metric: OFI (contiguous rows; None for first)
    ofis = [""]
    for i in range(1, len(rows)):
        val = ofi_l1(rows[i - 1], rows[i])
        ofis.append("" if val is None else f"{val:.10f}")

    # graft series metrics
    for i, r in enumerate(enriched):
        r["rv_window"] = args.rv_window
        r["rv"] = "" if rv[i] is None else f"{rv[i]:.10f}"
        r["ofi_l1"] = ofis[i]

    # Header: original cols + new ones
    base_cols = list(rows[0].keys())
    add_cols = [
        "spread",
        "mid",
        "relative_spread_bps",
        "microprice",
        "microprice_imbalance_bps",
        "imbalance_l1",
        "imbalance_k",
        "notional_bid_k",
        "notional_ask_k",
        "rv_window",
        "rv",
        "ofi_l1",
    ]
    header = base_cols + add_cols
    _write_csv(enriched, header, out_path)

    log.info("Wrote metrics → %s", out_path)


if __name__ == "__main__":
    main()
