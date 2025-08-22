# Created by LP
# Date: 2025-08-22
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

import numpy as np
import pandas as pd

from market_microstructure_toolkit.event_metrics import compute_event_time_metrics


def test_event_time_metrics_basic():
    # synthetic 5 updates with level-1 data
    df = pd.DataFrame(
        {
            "best_bid": [100.0, 100.5, 100.5, 100.4, 100.6],
            "best_ask": [100.2, 100.6, 100.6, 100.5, 100.7],
            "bid1_size": [10.0, 12.0, 8.0, 9.0, 11.0],
            "ask1_size": [9.0, 7.0, 7.5, 8.0, 7.0],
        }
    )

    out = compute_event_time_metrics(df.copy(), rv_window=3)

    # columns exist
    for col in ["mid", "spread_bps", "microprice", "ofi_l1", "ret_mid", "rv_event_3"]:
        assert col in out.columns

    # mid = (bb+ba)/2
    mid_expected = (df["best_bid"] + df["best_ask"]) / 2
    assert np.allclose(out["mid"], mid_expected, equal_nan=True)

    # spread_bps = (ba-bb)/mid * 1e4
    spread_expected = (df["best_ask"] - df["best_bid"]) / mid_expected * 1e4
    assert np.allclose(out["spread_bps"], spread_expected, equal_nan=True)

    # microprice = (ba*qb + bb*qa) / (qb+qa)
    denom = (df["bid1_size"] + df["ask1_size"]).replace(0, np.nan)
    micro_expected = (df["best_ask"] * df["bid1_size"] + df["best_bid"] * df["ask1_size"]) / denom
    micro_expected = micro_expected.fillna(mid_expected)
    assert np.allclose(out["microprice"], micro_expected, equal_nan=True)

    # ofi_l1 first row is 0 by convention
    assert out.loc[0, "ofi_l1"] == 0.0

    # ret_mid is log diff
    ret_expected = np.log(out["mid"]).diff()
    assert np.allclose(out["ret_mid"], ret_expected, equal_nan=True)

    # rv_event_3 is rolling sum of squared returns, window=3
    rv_expected = (ret_expected**2).rolling(3).sum()
    assert np.allclose(out["rv_event_3"], rv_expected, equal_nan=True)


def test_event_time_metrics_handles_missing_l1_sizes():
    # no bid1_size/ask1_size: should be created as 0 and microprice falls back to mid
    df = pd.DataFrame(
        {
            "best_bid": [10.0, 10.1],
            "best_ask": [10.2, 10.3],
        }
    )
    out = compute_event_time_metrics(df.copy(), rv_window=2)

    # created columns
    assert "bid1_size" in out.columns
    assert "ask1_size" in out.columns

    # microprice == mid because sizes are zero
    mid = (df["best_bid"] + df["best_ask"]) / 2
    assert np.allclose(out["microprice"], mid, equal_nan=True)

    # OFI well-defined
    assert "ofi_l1" in out.columns
