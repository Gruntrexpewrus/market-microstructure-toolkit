# Created by LP
# Date: 2025-08-22
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

import sys

import pandas as pd

from market_microstructure_toolkit.event_metrics_cli import main as evt_cli_main


def test_event_metrics_cli_csv(tmp_path, monkeypatch):
    # Prepare a small CSV that matches our flat-schema expectations
    input_path = tmp_path / "book.csv"
    df = pd.DataFrame(
        {
            "ts_ms": [1, 2, 3, 4],
            "iso": ["t1", "t2", "t3", "t4"],
            "exchange_id": ["x"] * 4,
            "symbol": ["S"] * 4,
            "book_level": ["L2"] * 4,
            "raw_nonce": [None] * 4,
            "best_bid": [100.0, 100.1, 100.2, 100.1],
            "best_ask": [100.2, 100.3, 100.3, 100.2],
            "bid1_price": [100.0] * 4,
            "bid1_size": [5.0, 6.0, 5.5, 5.2],
            "ask1_price": [100.2] * 4,
            "ask1_size": [4.5, 4.8, 5.0, 4.7],
        }
    )
    df.to_csv(input_path, index=False)

    out_path = tmp_path / "out.csv"

    # Run CLI: mmt-event-metrics input --rv-window 3 --out out.csv
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")  # keep test output clean
    monkeypatch.setattr(
        sys,
        "argv",
        ["mmt-event-metrics", str(input_path), "--rv-window", "3", "--out", str(out_path)],
    )
    evt_cli_main()

    assert out_path.exists(), "CLI should write an output CSV"

    out = pd.read_csv(out_path)
    # basic sanity: new columns present
    for col in ["mid", "spread_bps", "microprice", "ofi_l1", "ret_mid", "rv_event_3"]:
        assert col in out.columns
    # shape preserved
    assert len(out) == len(df)
