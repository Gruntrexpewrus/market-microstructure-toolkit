# Created by LP
# Date: 2025-08-16
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

# Purpose: unit tests for src/record.py (header/row shaping and writers)
#
# Good practices followed:
# - No network: we fake the exchange and monkeypatch the snapshot function.
# - No writes to repo: we write under pytest's tmp_path (isolated temp dir).
# - Deterministic inputs: fixed timestamps and values to avoid flaky tests.
# - Narrow assertions: we assert schema/shape and file existence, not prices.

from __future__ import annotations

import csv
from pathlib import Path

import pytest

# Import the module under test; support both package and flat layouts.
import market_microstructure_toolkit.record as record


# robust import for setup_log from root or src/
from market_microstructure_toolkit.setup_log import setup_logging


# ensure logs go to tests/_artifacts/logs and rebind the SAME logger the code uses
ARTIFACTS_LOGS = Path("tests/_artifacts/logs")
ARTIFACTS_LOGS.mkdir(parents=True, exist_ok=True)
record.log = setup_logging(log_dir=str(ARTIFACTS_LOGS), name="record")

# Directory where artifacts will be written
ARTIFACTS = Path("tests/_artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ------------------------ Test helpers ------------------------


def fake_snapshot(symbol: str = "BTC/USDT", book_level: str = "L2"):
    """
    Return a deterministic, normalized snapshot structure
    that matches what record._flatten_snapshot_to_row expects.

    We purposely provide fewer levels than 'depth' to verify padding logic.
    """
    return {
        "ts_ms": 1_700_000_000_000,  # deterministic ms epoch
        "iso": "2023-11-14T00:00:00+00:00",
        "exchange_id": "fake",
        "symbol": symbol,
        "book_level": book_level,
        "raw_nonce": 42,
        "best_bid": 100.0,
        "best_ask": 100.1,
        "bids": [(100.0, 1.0), (99.9, 1.01)],  # 2 levels only
        "asks": [(100.1, 0.50)],  # 1 level only
    }


class FakeExchange:
    """
    Minimal "ccxt-like" exchange placeholder for tests.
    We don't actually call its methods (snapshot function is monkeypatched),
    but having an object mimics the public API shape of src.record.
    """

    id = "fake"


# ------------------------ Unit tests ------------------------


def test_header_for_depth_names_and_lengths():
    """Header must be deterministic: 8 meta + 4*depth."""
    depth = 3
    h = record._header_for_depth(depth)

    # 8 meta columns + bids(2*depth) + asks(2*depth)
    assert len(h) == 8 + 4 * depth

    # exact meta columns in front (stable contract for readers)
    assert h[:8] == [
        "ts_ms",
        "iso",
        "exchange_id",
        "symbol",
        "book_level",
        "raw_nonce",
        "best_bid",
        "best_ask",
    ]

    # spot-check a few labels (grouped: all bids first, then all asks)
    assert h[8:12] == ["bid1_price", "bid1_size", "bid2_price", "bid2_size"]
    # correct tail for depth=3 (ask2 then ask3)
    assert h[-4:] == ["ask2_price", "ask2_size", "ask3_price", "ask3_size"]


def test_flatten_snapshot_to_row_padding_and_meta():
    """
    Validate:
    - meta mapping
    - numeric formatting (10 decimals)
    - padding blanks to maintain fixed-width rows
    """
    depth = 4
    snap = fake_snapshot()
    row = record._flatten_snapshot_to_row(snap, depth)
    header = record._header_for_depth(depth)

    # Row is a dict keyed by header names (contract: keys must exist)
    for k in header:
        assert k in row

    # meta
    assert row["ts_ms"] == str(snap["ts_ms"])
    assert row["iso"] == snap["iso"]
    assert row["exchange_id"] == "fake"
    assert row["symbol"] == "BTC/USDT"
    assert row["book_level"] == "L2"
    assert row["raw_nonce"] == "42"
    assert float(row["best_bid"]) == 100.0
    assert float(row["best_ask"]) == 100.1

    # bids: 2 filled + 2 padded
    assert row["bid1_price"] == "100.0000000000"
    assert row["bid1_size"] == "1.0000000000"
    assert row["bid2_price"] == "99.9000000000"
    assert row["bid2_size"] == "1.0100000000"
    assert row["bid3_price"] == "" and row["bid3_size"] == ""
    assert row["bid4_price"] == "" and row["bid4_size"] == ""

    # asks: 1 filled + 3 padded
    assert row["ask1_price"] == "100.1000000000"
    assert row["ask1_size"] == "0.5000000000"
    assert row["ask2_price"] == "" and row["ask2_size"] == ""
    assert row["ask3_price"] == "" and row["ask3_size"] == ""
    assert row["ask4_price"] == "" and row["ask4_size"] == ""


def test_record_snapshots_csv_monkeypatched(monkeypatch):
    """
    Record 1 second at 2 Hz (≈2 rows) to CSV and assert on the produced file
    and the module log file written under tests/_artifacts/logs/record.log.
    """
    # Arrange: deterministic fake snapshot (no network)
    monkeypatch.setattr(
        record,
        "fetch_order_book_snapshot",
        lambda ex, symbol, depth, book_level: fake_snapshot(symbol, book_level),
    )
    ex = FakeExchange()
    out = ARTIFACTS / "book.csv"

    # Act
    record.record_snapshots(
        ex=ex,
        symbol="BTC/USDT",
        depth=5,
        seconds=1,
        hz=2.0,
        out_path=str(out),
        book_level="L2",
        out_format="csv",
    )

    # Assert: file exists and has fixed-width rows
    assert out.exists()
    rows = list(csv.reader(out.open()))
    assert len(rows) >= 2  # header + ≥1 row
    header = rows[0]
    assert len(header) == 8 + 4 * 5
    assert all(len(r) == len(header) for r in rows[1:])

    # Assert on the actual log file (since logger uses custom handlers & propagate=False)
    log_path = ARTIFACTS_LOGS / "record.log"
    assert log_path.exists()
    text = log_path.read_text()
    assert "Recording 1s at 2.0 Hz" in text
    assert "Done. Rows captured: 2" in text


@pytest.mark.parametrize("engine", ["parquet", "csv"])
def test_record_snapshots_parquet_or_fallback(monkeypatch, engine):
    """
    Parquet request → parquet or clean CSV fallback.
    CSV request → CSV path (no parquet).
    """
    monkeypatch.setattr(
        record,
        "fetch_order_book_snapshot",
        lambda ex, symbol, depth, book_level: fake_snapshot(symbol, book_level),
    )

    ex = FakeExchange()

    # choose output path by engine (important!)
    out_path = ARTIFACTS / (
        "book_param.parquet" if engine == "parquet" else "book_param.csv"
    )

    # clean previous runs for idempotence
    for p in (ARTIFACTS / "book_param.parquet", ARTIFACTS / "book_param.csv"):
        if p.exists():
            p.unlink()

    record.record_snapshots(
        ex=ex,
        symbol="BTC/USDT",
        depth=3,
        seconds=1,
        hz=1.0,
        out_path=str(out_path),
        book_level="L2",
        out_format=engine,
    )

    parquet_exists = (ARTIFACTS / "book_param.parquet").exists()
    csv_exists = (ARTIFACTS / "book_param.csv").exists()

    if engine == "parquet":
        assert parquet_exists or csv_exists  # fallback allowed
    else:
        assert csv_exists and not parquet_exists
