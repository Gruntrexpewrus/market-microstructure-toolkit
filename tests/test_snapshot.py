import pytest

import market_microstructure_toolkit.snapshot as snapshot


class DummyExchange:
    def __init__(self, id="dummy"):
        self.id = id

    def fetch_order_book(self, symbol, limit=10, params=None):
        return {
            "timestamp": 1730000000000,
            "bids": [[100.0, 1.5], ["bad", "data"]],
            "asks": [[101.0, 2.0]],
            "symbol": symbol,
            "nonce": 123,
        }


def test_norm_valid_and_invalid():
    data = [[100, 1], ["bad", "data"], [101.5, 2.5]]
    result = snapshot._norm(data, 3)
    assert result == [(100.0, 1.0), (101.5, 2.5)]


def test_fetch_order_book_snapshot_L1(monkeypatch):
    ex = DummyExchange()
    # Patch L1 call
    snapshot_result = snapshot.fetch_order_book_snapshot(ex, "BTC/USDT", depth=1, book_level="L1")
    assert snapshot_result["best_bid"] == 100.0
    assert snapshot_result["best_ask"] == 101.0
    assert snapshot_result["symbol"] == "BTC/USDT"
    assert snapshot_result["book_level"] == "L1"
    assert "iso" in snapshot_result


def test_fetch_order_book_snapshot_invalid_level():
    ex = DummyExchange()
    with pytest.raises(ValueError):
        snapshot.fetch_order_book_snapshot(ex, "BTC/USDT", book_level="BAD")
