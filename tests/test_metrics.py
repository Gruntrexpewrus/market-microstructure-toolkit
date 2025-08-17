# Created by LP
# Date: 2025-08-17
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3


from __future__ import annotations
from market_microstructure_toolkit.metrics import compute_row_metrics


def _fake_row(depth: int = 3):
    # minimal row that mirrors recorder output schema (strings allowed)
    row = {
        "best_bid": "100.0000000000",
        "best_ask": "100.1000000000",
        "ts_ms": "1700000000000",
        "iso": "2023-11-14T00:00:00+00:00",
        "exchange_id": "fake",
        "symbol": "BTC/USDT",
        "book_level": "L2",
        "raw_nonce": "42",
    }
    # Level data (pad with blanks to depth)
    row.update(
        {
            "bid1_price": "100.0000000000",
            "bid1_size": "1.0000000000",
            "bid2_price": "99.9000000000",
            "bid2_size": "1.0100000000",
            "bid3_price": "",
            "bid3_size": "",
            "ask1_price": "100.1000000000",
            "ask1_size": "0.5000000000",
            "ask2_price": "",
            "ask2_size": "",
            "ask3_price": "",
            "ask3_size": "",
        }
    )
    return row


def test_compute_row_metrics_depth3():
    row = _fake_row(3)
    m = compute_row_metrics(row, depth=3)
    # spread = 0.1, mid = 100.05, l1 imbalance = 1.0 / (1.0 + 0.5) = 0.666..., depth-K = (1 + 1.01)/(1+1.01+0.5)
    assert abs(m["spread"] - 0.1) < 1e-12
    assert abs(m["mid"] - 100.05) < 1e-12
    assert abs(m["imbalance_l1"] - (1.0 / 1.5)) < 1e-12
    assert abs(m["imbalance_k"] - ((1.0 + 1.01) / (1.0 + 1.01 + 0.5))) < 1e-12
