# src/record.py
# Created by LP
# Date: 2025-08-15
# T4: Order book recorder → CSV or Parquet (fixed-width rows)
# src/record.py
# Created by LP
# Date: 2025-08-15

# T4: Order book recorder → CSV or Parquet (fixed-width rows)

from __future__ import annotations

import csv
import os
import time
from typing import List, Dict, Any
from pathlib import Path
import logging

# --- robust import for snapshot (works in both module and script modes) ---
try:
    # Package import (preferred): python -m src.record
    from .snapshot import fetch_order_book_snapshot
except Exception:
    # Script import fallback: python src/record.py
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
    from src.snapshot import fetch_order_book_snapshot  # type: ignore

# Named logger; configure handlers only in __main__ or via tests
log = logging.getLogger("record")

# ------------- helpers: paths & schema -------------


def _ensure_parent(path: str) -> None:
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)


def _meta_header() -> List[str]:
    # Stable meta order; append new meta to the end (don’t break readers)
    return [
        "ts_ms",
        "iso",
        "exchange_id",
        "symbol",
        "book_level",
        "raw_nonce",
        "best_bid",
        "best_ask",
    ]


def _header_for_depth(depth: int) -> List[str]:
    cols = _meta_header()
    for i in range(1, depth + 1):
        cols += [f"bid{i}_price", f"bid{i}_size"]
    for i in range(1, depth + 1):
        cols += [f"ask{i}_price", f"ask{i}_size"]
    return cols


def _flatten_snapshot_to_row(snap: Dict[str, Any], depth: int) -> Dict[str, Any]:
    """
    Convert normalized snapshot → flat dict matching _header_for_depth(depth).
    Floats are formatted as 10-decimal strings for CSV stability.
    Parquet path converts them back to numeric types.
    """
    row: Dict[str, Any] = {}

    # meta
    row["ts_ms"] = str(snap["ts_ms"])
    row["iso"] = snap["iso"]
    row["exchange_id"] = snap.get("exchange_id") or ""
    row["symbol"] = snap.get("symbol") or ""
    row["book_level"] = snap.get("book_level") or ""
    row["raw_nonce"] = "" if snap.get("raw_nonce") is None else str(snap["raw_nonce"])
    row["best_bid"] = "" if snap["best_bid"] is None else f'{snap["best_bid"]:.10f}'
    row["best_ask"] = "" if snap["best_ask"] is None else f'{snap["best_ask"]:.10f}'

    # sides with padding
    bids = snap["bids"][:depth]
    asks = snap["asks"][:depth]
    bids += [("", "")] * (depth - len(bids))
    asks += [("", "")] * (depth - len(asks))

    for i, (px, sz) in enumerate(bids, 1):
        row[f"bid{i}_price"] = "" if px == "" else f"{px:.10f}"
        row[f"bid{i}_size"] = "" if sz == "" else f"{sz:.10f}"
    for i, (px, sz) in enumerate(asks, 1):
        row[f"ask{i}_price"] = "" if px == "" else f"{px:.10f}"
        row[f"ask{i}_size"] = "" if sz == "" else f"{sz:.10f}"

    return row


# ------------- writers: CSV & Parquet -------------


def _write_csv(rows: List[Dict[str, Any]], out_path: str, depth: int) -> None:
    header = _header_for_depth(depth)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_parquet(rows: List[Dict[str, Any]], out_path: str, depth: int) -> None:
    try:
        import pandas as pd
    except Exception as e:
        log.warning(
            "Parquet requested, but pandas not available (%s). Falling back to CSV.", e
        )
        return _write_csv(rows, out_path.replace(".parquet", ".csv"), depth)

    df = pd.DataFrame(rows, columns=_header_for_depth(depth))

    # Cast numeric columns from strings (CSV-friendly) to numeric (Parquet-friendly)
    numeric_cols = ["best_bid", "best_ask"]
    for i in range(1, depth + 1):
        numeric_cols += [
            f"bid{i}_price",
            f"bid{i}_size",
            f"ask{i}_price",
            f"ask{i}_size",
        ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # blanks → NaN

    # Optional: keep ts_ms as nullable Int64
    df["ts_ms"] = pd.to_numeric(df["ts_ms"], errors="coerce").astype("Int64")

    # Try pyarrow, then fastparquet, else warn + CSV fallback
    try:
        df.to_parquet(out_path, engine="pyarrow", index=False)
    except Exception as e1:
        try:
            df.to_parquet(out_path, engine="fastparquet", index=False)
        except Exception as e2:
            log.warning(
                "Could not write parquet with pyarrow (%s) nor fastparquet (%s). Falling back to CSV.",
                e1,
                e2,
            )
            return _write_csv(rows, out_path.replace(".parquet", ".csv"), depth)


# ------------- public API -------------


def record_snapshots(
    ex,
    symbol: str,
    depth: int,
    seconds: int,
    hz: float,
    out_path: str,
    book_level: str = "L2",
    out_format: str = "csv",  # "csv" | "parquet"
) -> None:
    """
    Record `seconds` of snapshots at `hz` Hz to CSV or Parquet.

    Extra columns: exchange_id, symbol, book_level, raw_nonce
    """
    _ensure_parent(out_path)

    total_iters = int(seconds * hz)
    period = 1.0 / float(hz)
    rows: List[Dict[str, Any]] = []

    log.info(
        "Recording %ss at %s Hz (depth=%s, level=%s, format=%s) → %s",
        seconds,
        hz,
        depth,
        book_level,
        out_format,
        out_path,
    )

    next_t = time.time()
    written = 0

    for k in range(total_iters):
        now = time.time()
        if now < next_t:
            time.sleep(next_t - now)

        try:
            snap = fetch_order_book_snapshot(
                ex, symbol, depth=depth, book_level=book_level
            )
            rows.append(_flatten_snapshot_to_row(snap, depth))
            written += 1
        except Exception as e:
            log.warning("snapshot failed (%s/%s): %s", k + 1, total_iters, e)

        next_t += period

    # Emit once at the end
    if out_format.lower() == "parquet" or out_path.lower().endswith(".parquet"):
        if not out_path.lower().endswith(".parquet"):
            out_path = out_path + ".parquet"
        _write_parquet(rows, out_path, depth)
    else:
        if not out_path.lower().endswith(".csv"):
            out_path = out_path + ".csv"
        _write_csv(rows, out_path, depth)

    log.info("Done. Rows captured: %s → %s", written, out_path)


# ------------- demo / CLI -------------

if __name__ == "__main__":
    """
    Demo:
        - CSV:     python -m src.record
        - Parquet: python -m src.record parquet
        - Script:  python src/record.py
    """
    import sys

    # Configure logger only when running the demo:
    try:
        from .setup_log import setup_logging  # package mode
        from .exchange import make_exchange
    except Exception:
        # script mode fallback
        from src.setup_log import setup_logging  # type: ignore
        from src.exchange import make_exchange  # type: ignore

    setup_logging(name="record")

    seconds = 10
    fmt = "csv"
    if len(sys.argv) > 1:
        fmt = sys.argv[1].lower()

    Path("data").mkdir(parents=True, exist_ok=True)
    ex = make_exchange("bybit", default_type="swap", timeout=10000)
    symbol = "BTC/USDT:USDT"

    out = (
        f"data/BTCUSDT_swap_demo_{seconds}s.{ 'parquet' if fmt=='parquet' else 'csv' }"
    )
    record_snapshots(
        ex=ex,
        symbol=symbol,
        depth=100,
        seconds=seconds,
        hz=1.0,
        out_path=out,
        book_level="L2",
        out_format=fmt,
    )

    # optional second run in parquet for the demo
    if fmt != "parquet":
        record_snapshots(
            ex=ex,
            symbol=symbol,
            depth=100,
            seconds=seconds,
            hz=1.0,
            out_path=f"data/BTCUSDT_swap_demo_{seconds}s.parquet",
            book_level="L2",
            out_format="parquet",
        )
