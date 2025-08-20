# Created by LP
# Date: 2025-08-19
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

from __future__ import annotations

import csv
import time

import pytest

# pytestmark = pytest.mark.asyncio
# import the internal coroutines directly
from market_microstructure_toolkit.ws_record import (
    _header_for_depth,
    _stream_rest_fallback,
)


class FakeExchange:
    id = "fake"


# Deterministic fake snapshot
def fake_snapshot(symbol: str = "ETH/USDT:USDT", depth: int = 3):
    ts = int(time.time() * 1000)
    return {
        "ts_ms": ts,
        "iso": "2024-01-01T00:00:00+00:00",
        "best_bid": 100.0,
        "best_ask": 100.1,
        "bids": [(100.0, 1.0), (99.9, 1.01)],
        "asks": [(100.1, 0.50)],
        "symbol": symbol,
        "exchange_id": "fake",
        "raw_nonce": 7,
        "book_level": "L2",
    }


@pytest.mark.asyncio
async def test_stream_rest_fallback_writes_csv(tmp_path, monkeypatch):
    import market_microstructure_toolkit.ws_record as ws

    # (1) Return deterministic snapshots (no network)
    monkeypatch.setattr(
        ws,
        "fetch_order_book_snapshot",
        lambda ex, s, depth, book_level: fake_snapshot(s, depth),
    )

    # (2) Avoid ccxt: make a fake exchange object
    monkeypatch.setattr(ws, "make_exchange", lambda name, **kw: FakeExchange())

    out = tmp_path / "out.csv"
    await _stream_rest_fallback(
        exchange_id="fake",
        symbol="ETH/USDT:USDT",
        depth=3,
        out=out,
        seconds=2,
        hz=5.0,
    )

    assert out.exists()
    rows = list(csv.reader(out.open()))
    assert len(rows) >= 2
    header = rows[0]
    assert header == _header_for_depth(3)
    assert all(len(r) == len(header) for r in rows[1:])
