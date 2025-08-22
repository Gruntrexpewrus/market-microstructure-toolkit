# Created by LP
# Date: 2025-08-17
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

from __future__ import annotations

import numbers
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from .metrics import compute_row_metrics, ofi_l1, rolling_realized_variance
from .setup_log import setup_logging


def _prefer_parquet(p: Path) -> bool:
    return p.suffix.lower() == ".parquet"


def _load_parquet_rows(path: Path) -> Iterator[Dict[str, Any]]:
    """Strict parquet reader: fails with a clear message if engine missing."""
    try:
        import pandas as pd  # local import to keep CLI light

        df = pd.read_parquet(path)
    except ImportError as e:
        raise RuntimeError(
            "Reading Parquet requires 'pyarrow' or 'fastparquet'. "
            "Install one of them (e.g., 'pip install pyarrow')."
        ) from e
    for _, r in df.iterrows():
        d = r.to_dict()
        # normalize NaN->None
        for k, v in list(d.items()):
            # pandas present; safe to call isna
            if pd.isna(v):
                d[k] = None
        yield d


def _load_csv_rows(path: Path) -> Iterator[Dict[str, Any]]:
    import csv

    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            yield r


def _load_rows(path: Path) -> Iterator[Dict[str, Any]]:
    if _prefer_parquet(path):
        yield from _load_parquet_rows(path)
    else:
        yield from _load_csv_rows(path)


def _format_float(val: Any) -> str:
    """Format any real number to 10dp; keep '' for None."""
    if val is None:
        return ""
    if isinstance(val, numbers.Real):
        return f"{float(val):.10f}"
    return str(val)


def _write_csv(rows: Iterable[Dict[str, Any]], header: List[str], out_path: Path) -> None:
    import csv

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _as_float_or_none(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def main() -> None:
    import argparse

    # --- logging early
    log = setup_logging(name="metrics_cli")

    parser = argparse.ArgumentParser(description="Add metrics to a recorded CSV/Parquet.")
    parser.add_argument("input_file", help="Input CSV/Parquet from record.py or ws_record.py")
    parser.add_argument("depth", type=int, help="Top-K depth to use for imbalance/notional")
    parser.add_argument(
        "--rv-window", type=int, default=20, help="Rolling RV window length (in rows)"
    )
    args = parser.parse_args()

    in_path = Path(args.input_file).expanduser().resolve()
    if not in_path.exists():
        log.error("Input not found: %s", in_path)
        sys.exit(2)

    out_path = in_path.with_name(in_path.stem + "_metrics.csv")
    log.info("Reading %s", in_path)

    rows = list(_load_rows(in_path))
    if not rows:
        log.warning("No rows found in input.")
        sys.exit(0)
    log.info("Loaded %d rows", len(rows))

    # 1) per-row metrics
    enriched: List[Dict[str, Any]] = []
    for r in rows:
        m = compute_row_metrics(r, depth=args.depth)  # returns numeric metrics
        # format floats for CSV stability; keep None as ""
        formatted = {k: _format_float(v) for k, v in m.items()}
        enriched.append({**r, **formatted})

    # 2) series metrics: rolling RV based on 'mid'
    mids = [_as_float_or_none(r.get("mid")) for r in enriched]
    rv = rolling_realized_variance(mids, window=args.rv_window)

    # 3) series metric: OFI (contiguous rows; None for first)
    # Use original rows; ofi_l1(prev_row, curr_row) per your API
    ofis = [""]
    for i in range(1, len(rows)):
        val = ofi_l1(rows[i - 1], rows[i])
        ofis.append("" if val is None else f"{float(val):.10f}")

    # graft series metrics
    for i, r in enumerate(enriched):
        r["rv_window"] = args.rv_window
        r["rv"] = "" if rv[i] is None else f"{float(rv[i]):.10f}"
        r["ofi_l1"] = ofis[i]

    # Header: original cols + new ones (keep your order)
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(enriched, header, out_path)
    log.info("Wrote metrics → %s", out_path)
    print(f"✅ Wrote metrics → {out_path}")


if __name__ == "__main__":
    main()

# example usage
# mmt-metrics data/ws_bybit_ETHUSDTUSDT_d5_5s_5hz.csv 5 --rv-window 20
