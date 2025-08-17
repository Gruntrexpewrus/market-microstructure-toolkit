# Created by LP
# Date: 2025-08-17
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

# Add Phase-1 metrics to a recorded CSV/Parquet

from __future__ import annotations
import sys
from pathlib import Path

from .metrics import compute_row_metrics  # same package dir as this file


def _prefer_parquet(p: Path) -> bool:
    return p.suffix.lower() == ".parquet"


def _load_rows(path: Path, depth: int):
    """
    Yields rows (dicts). If parquet available (pyarrow/fastparquet), use pandas.
    Otherwise, stream CSV.
    """
    if _prefer_parquet(path):
        try:
            import pandas as pd

            df = pd.read_parquet(path)
            for _, r in df.iterrows():
                yield {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
            return
        except Exception:
            pass

    # Fallback: CSV
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
    from .setup_log import setup_logging

    log = setup_logging(name="metrics_cli")
    if len(sys.argv) < 3:
        print("Usage: python -m src.metrics_cli <input_file> <depth>")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    depth = int(sys.argv[2])
    assert in_path.exists(), f"Input not found: {in_path}"

    # Decide output filename
    out_path = in_path.with_name(in_path.stem + "_metrics.csv")

    # Stream input rows, compute metrics, and write combined CSV
    first_row = None
    rows_out = []
    for row in _load_rows(in_path, depth):
        if first_row is None:
            first_row = row
        m = compute_row_metrics(row, depth)
        rows_out.append(
            {
                **row,
                **{
                    k: ("" if v is None else f"{v:.10f}" if isinstance(v, float) else v)
                    for k, v in m.items()
                },
            }
        )

    if first_row is None:
        log.warning("No rows found in input.")
        sys.exit(0)

    # Build header: all original columns + new metric columns at the end
    header = list(first_row.keys()) + ["spread", "mid", "imbalance_l1", "imbalance_k"]
    _write_csv(rows_out, header, out_path)
    log.info("Wrote metrics → %s", out_path)


if __name__ == "__main__":
    main()
