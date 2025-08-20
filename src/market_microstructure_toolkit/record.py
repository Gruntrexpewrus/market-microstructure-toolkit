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
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

# --- robust import for snapshot (works in both module and script modes) ---
from .snapshot import fetch_order_book_snapshot

# ------------- helpers: paths & schema -------------
log = logging.getLogger(__name__)  # no handlers here


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
        log.warning("Parquet requested, but pandas not available (%s). Falling back to CSV.", e)
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
            snap = fetch_order_book_snapshot(ex, symbol, depth=depth, book_level=book_level)
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
    CLI usage examples:

      # 10s @ 1 Hz, Bybit USDT swap, depth=50, L2, CSV
      python -m src.record \
        --exchange bybit \
        --market-type swap \
        --symbol BTC/USDT:USDT \
        --seconds 10 \
        --hz 1 \
        --depth 50 \
        --book-level L2 \
        --format csv \
        --out data/BTCUSDT_swap_10s.csv

      # Same but Parquet:
      python -m src.record \
        --exchange bybit --market-type swap \
        --symbol BTC/USDT:USDT --seconds 10 --hz 1 \
        --depth 50 --book-level L2 --format parquet \
        --out data/BTCUSDT_swap_10s.parquet
    """
    import argparse
    import json

    from .exchange import make_exchange
    from .setup_log import setup_logging

    parser = argparse.ArgumentParser(description="Record order-book snapshots to CSV/Parquet.")
    parser.add_argument(
        "--exchange", required=True, help="CCXT exchange id, e.g., bybit, binance, okx"
    )
    parser.add_argument(
        "--market-type",
        choices=["spot", "swap", "future"],
        default="spot",
        help="CCXT defaultType for the exchange (default: spot)",
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Unified symbol, e.g., BTC/USDT or BTC/USDT:USDT for swaps",
    )
    parser.add_argument("--seconds", type=int, default=60, help="Duration to record (seconds)")
    parser.add_argument("--hz", type=float, default=1.0, help="Snapshots per second (Hz)")
    parser.add_argument("--depth", type=int, default=50, help="Top-N levels to save per side")
    parser.add_argument(
        "--book-level",
        choices=["L1", "L2", "L3"],
        default="L2",
        help="Order book detail level",
    )
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv", help="Output format")
    parser.add_argument(
        "--out",
        default="",
        help="Output path (.csv or .parquet). If empty, an auto name is used.",
    )
    parser.add_argument("--timeout", type=int, default=10000, help="Exchange HTTP timeout (ms)")
    parser.add_argument(
        "--params",
        default="",
        help="Optional JSON for exchange-specific params (rarely needed)",
    )

    args = parser.parse_args()

    # Configure the named logger used in this module
    log = setup_logging(name="record")

    # Build exchange
    ex = make_exchange(args.exchange, default_type=args.market_type, timeout=args.timeout)

    # Resolve output path
    if args.out:
        out_path = args.out
    else:
        Path("data").mkdir(parents=True, exist_ok=True)
        # eg: data/bybit_BTCUSDTUSDT_swap_L2_60s.parquet
        sym_sanitized = args.symbol.replace("/", "").replace(":", "")
        out_path = f"data/{args.exchange}_{sym_sanitized}_{args.market_type}_{args.book_level}_{args.seconds}s.{args.format}"
    # Optional venue params (JSON string)
    try:
        extra_params = json.loads(args.params) if args.params else {}
    except Exception as e:
        log.warning("Could not parse --params JSON (%s). Proceeding without extra params.", e)
        extra_params = {}

    # Run
    record_snapshots(
        ex=ex,
        symbol=args.symbol,
        depth=args.depth,
        seconds=args.seconds,
        hz=args.hz,
        out_path=out_path,
        book_level=args.book_level,
        out_format=args.format,
    )

    log.info("Saved: %s", out_path)
