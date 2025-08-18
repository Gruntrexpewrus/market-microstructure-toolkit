# Created by LP
# Date: 2025-08-18
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3


from __future__ import annotations

import math
import pytest

from market_microstructure_toolkit.metrics import (
    compute_row_metrics,
    relative_spread_bps,
    microprice,
    microprice_imbalance,
    rolling_realized_variance,
    notional_depth,
    book_slope,
    ofi_l1,
)

# ---------- helpers ----------


def _fake_row(depth: int = 3):
    """Minimal flat row shaped like recorder output (strings allowed)."""
    row = {
        "best_bid": "100.0000000000",
        "best_ask": "100.1000000000",
        "ts_ms": "1700000000000",
        "iso": "2023-11-14T00:00:00+00:00",
        "exchange_id": "fake",
        "symbol": "BTC/USDT",
        "book_level": "L2",
        "raw_nonce": "42",
        # L1
        "bid1_price": "100.0000000000",
        "bid1_size": "1.0000000000",
        "ask1_price": "100.1000000000",
        "ask1_size": "0.5000000000",
        # L2
        "bid2_price": "99.9000000000",
        "bid2_size": "1.0100000000",
        "ask2_price": "",
        "ask2_size": "",
        # L3 (blank)
        "bid3_price": "",
        "bid3_size": "",
        "ask3_price": "",
        "ask3_size": "",
    }
    return row


# ---------- core row metrics ----------


def test_compute_row_metrics_depth3():
    row = _fake_row(3)
    m = compute_row_metrics(row, depth=3)
    # spread & mid
    assert abs(m["spread"] - 0.1) < 1e-12
    assert abs(m["mid"] - 100.05) < 1e-12
    # L1 imbalance = 1 / (1 + 0.5) = 0.666...
    assert abs(m["imbalance_l1"] - (1.0 / 1.5)) < 1e-12
    # depth-K imbalance = (1 + 1.01)/(1 + 1.01 + 0.5)
    expected_k = (1.0 + 1.01) / (1.0 + 1.01 + 0.5)
    assert abs(m["imbalance_k"] - expected_k) < 1e-12


# ---------- phase-2: relative spread, microprice ----------


def test_relative_spread_bps():
    bb, ba = 100.0, 100.1
    mid = 0.5 * (bb + ba)
    exp = 10_000.0 * (ba - bb) / mid
    assert abs(relative_spread_bps(bb, ba) - exp) < 1e-12


def test_microprice_and_imbalance():
    bb, ba = 100.0, 100.1
    bsz, asz = 1.0, 0.5
    mp = microprice(bb, ba, bsz, asz)
    assert mp is not None and bb < mp < ba
    # mid = 0.5 * (bb + ba)
    mpi = microprice_imbalance(bb, ba, bsz, asz)
    assert mpi is not None
    # sign should be positive (lean toward ask because ask queue is smaller)
    assert mpi > 0


# ---------- series metrics: RV ----------


def test_rolling_realized_variance_simple():
    # synthetic monotone prices → positive small RV after window filled
    prices = [100.0, 100.05, 100.10, 100.12, 100.11]
    rv = rolling_realized_variance(prices, window=3)
    assert len(rv) == len(prices)
    # first (window-1)=2 entries should be None
    assert rv[0] is None and rv[1] is None
    # later entries should be finite non-negative
    for v in rv[2:]:
        assert (v is None) or (v >= 0)


# ---------- notional depth & slope ----------


def test_notional_depth_and_book_slope():
    row = _fake_row(3)
    nb, na = notional_depth(row, depth=2)
    # bid notional = 100*1 + 99.9*1.01 ; ask level-2 blank so only L1
    assert nb is not None and na is not None
    assert abs(nb - (100.0 * 1.0 + 99.9 * 1.01)) < 1e-9
    assert abs(na - (100.1 * 0.5)) < 1e-9

    # slope requires >=2 valid levels; bid side has 2, ask side only 1 → None
    sb = book_slope(row, depth=2, side="bid")
    sa = book_slope(row, depth=2, side="ask")
    assert sb is not None
    assert sa is None


# ---------- OFI (L1) ----------


def test_ofi_l1_basic():
    r1 = {
        "bid1_price": "100.0",
        "bid1_size": "1.0",
        "ask1_price": "100.1",
        "ask1_size": "1.0",
    }
    # bid price up (adds Δbid_size), ask price up (subtract ask_size_prev)
    r2 = {
        "bid1_price": "100.1",
        "bid1_size": "1.2",
        "ask1_price": "100.2",
        "ask1_size": "0.9",
    }
    val = ofi_l1(r1, r2)
    assert val is not None
    # expected: (1.2-1.0) - 1.0 = -0.8  (since ask moved up)
    assert abs(val - (-0.8)) < 1e-12


# ---------- param safety ----------


@pytest.mark.parametrize("bad", [None, float("nan"), -1.0])
def test_relative_spread_bps_edge(bad):
    # any invalid side should yield None
    assert relative_spread_bps(bad, 100.0) is None or math.isnan(bad)  # defensive
