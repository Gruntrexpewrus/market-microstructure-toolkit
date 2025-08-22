# Created by LP
# Date: 2025-08-22
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

# Computes per-update features such as mid/microprice, spread (bps),
# Kyle-OFI at level-1, and rolling realized variance in *event time*
# (i.e., over a fixed number of updates, not wall-clock windows).

from __future__ import annotations

import numpy as np
import pandas as pd


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise a helpful error if any required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")


def compute_event_time_metrics(
    df: pd.DataFrame,
    *,
    rv_window: int = 20,
) -> pd.DataFrame:
    """
    Compute event-time features row-by-row from an order-book snapshot table.

    Expected schema (produced by `record.py` / `ws_record.py`):
      - best_bid, best_ask (floats)
      - bid1_size, ask1_size (level-1 sizes; floats)
      - Optional metadata (ts_ms, iso, exchange_id, ...)

    Adds columns:
      - mid
      - spread_bps
      - microprice
      - ofi_l1           (Kyle-style OFI at L1)
      - ret_mid          (log-return of mid)
      - rv_event_<W>     (rolling realized variance over W events)

    Parameters
    ----------
    df : pd.DataFrame
        Input table with one row per snapshot/event.
    rv_window : int, default 20
        Rolling window (in number of updates) for realized variance.

    Returns
    -------
    pd.DataFrame
        Original df with added feature columns (no copy).
    """
    # Validate necessary columns
    _require_cols(df, ["best_bid", "best_ask"])
    # L1 sizes are needed for microprice & OFI; if missing, fill as 0
    for col in ("bid1_size", "ask1_size"):
        if col not in df.columns:
            df[col] = 0.0

    # ---- Basic quotes ----
    bb = pd.to_numeric(df["best_bid"], errors="coerce")
    ba = pd.to_numeric(df["best_ask"], errors="coerce")

    mid = (bb + ba) / 2.0
    df["mid"] = mid

    # robust relative spread in bps (skip rows with invalid quotes)
    spread = (ba - bb).where((bb > 0) & (ba > 0))
    df["spread_bps"] = (spread / mid * 1e4).replace([np.inf, -np.inf], np.nan)

    # ---- Microprice (liquidity weighted mid at L1) ----
    q_bid = pd.to_numeric(df["bid1_size"], errors="coerce").fillna(0.0)
    q_ask = pd.to_numeric(df["ask1_size"], errors="coerce").fillna(0.0)
    denom = (q_bid + q_ask).replace(0.0, np.nan)
    micro = (ba * q_bid + bb * q_ask) / denom
    # fall back to mid if sizes are missing/zero
    df["microprice"] = micro.fillna(mid)

    # ---- Kyle-style OFI at L1 (event-time) ----
    # Standard definition:
    #   ΔB = 1{bid_p_t >= bid_p_{t-1}} * bid_q_t - 1{bid_p_t <= bid_p_{t-1}} * bid_q_{t-1}
    #   ΔA = 1{ask_p_t <= ask_p_{t-1}} * ask_q_t - 1{ask_p_t >= ask_p_{t-1}} * ask_q_{t-1}
    #   OFI = ΔB - ΔA
    bb_prev = bb.shift(1)
    ba_prev = ba.shift(1)
    qb_prev = q_bid.shift(1).fillna(0.0)
    qa_prev = q_ask.shift(1).fillna(0.0)

    # Bid contribution
    bid_up_or_same = (bb >= bb_prev) | bb_prev.isna()
    bid_down_or_same = (bb <= bb_prev) | bb_prev.isna()
    dB = bid_up_or_same.astype(float) * q_bid - bid_down_or_same.astype(float) * qb_prev

    # Ask contribution (note reversed inequalities)
    ask_down_or_same = (ba <= ba_prev) | ba_prev.isna()
    ask_up_or_same = (ba >= ba_prev) | ba_prev.isna()
    dA = ask_down_or_same.astype(float) * q_ask - ask_up_or_same.astype(float) * qa_prev

    df["ofi_l1"] = dB - dA
    df.loc[df.index[0], "ofi_l1"] = 0.0  # first row has no previous state

    # ---- Event-time realized variance of mid ----
    # (log returns squared, rolling sum)
    ret_mid = np.log(df["mid"]).diff()
    df["ret_mid"] = ret_mid
    df[f"rv_event_{rv_window}"] = (ret_mid**2).rolling(rv_window).sum()

    return df
