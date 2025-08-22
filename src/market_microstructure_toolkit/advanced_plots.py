# Created by LP
# Date: 2025-08-22
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Try to use your project's metrics.ofi_l1 if available
try:
    from .metrics import ofi_l1 as _ofi_l1_project
except Exception:  # fallback is defined below
    _ofi_l1_project = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


@dataclass
class SaveOpts:
    """Options controlling whether to save plots and where.

    Attributes
    ----------
    save : bool
        If True, figures are written to PNG files.
    outdir : Optional[Path]
        Directory where PNGs are saved. If None and `save=True`, caller should
        decide a default (e.g., plots/<stem>/).
    """

    save: bool = True
    outdir: Optional[Path] = None


# ------------------------------- utilities --------------------------------- #


def _save_or_show(fig: plt.Figure, name: str, opts: SaveOpts):
    """Save a figure as <outdir>/<name>.png or show it interactively."""
    if opts.save:
        assert opts.outdir is not None, "Save requested but outdir=None"
        opts.outdir.mkdir(parents=True, exist_ok=True)
        outfile = opts.outdir / f"{name}.png"
        fig.savefig(outfile, dpi=130, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved: %s", outfile)
    else:
        fig.tight_layout()
        fig.show()


def _fallback_ofi_l1(bid1_size: pd.Series, ask1_size: pd.Series) -> pd.Series:
    """Very simple OFI proxy from L1 sizes (placeholder if project metric not present)."""
    # Note: This is a placeholder signal: positive if bid size > ask size.
    return (bid1_size.fillna(0.0) - ask1_size.fillna(0.0)).astype(float)


def _ensure_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist: mid, spread_bps, microprice, ofi_l1, ret_mid.

    Notes
    -----
    - Uses L1 sizes for microprice/OFI.
    - Degrades gracefully if some size columns are missing (fills zeros).
    """
    out = df.copy()

    # mid
    if "mid" not in out:
        out["mid"] = (out["best_bid"] + out["best_ask"]) / 2.0

    # spread (bps)
    if "spread_bps" not in out:
        denom = out["mid"].replace(0, np.nan)
        out["spread_bps"] = ((out["best_ask"] - out["best_bid"]) / denom * 1e4).fillna(0.0)

    # l1 sizes (ensure)
    if "bid1_size" not in out:
        out["bid1_size"] = 0.0
    if "ask1_size" not in out:
        out["ask1_size"] = 0.0

    # microprice (liquidity-weighted mid at L1)
    if "microprice" not in out:
        denom = (out["bid1_size"] + out["ask1_size"]).replace(0, np.nan)
        mp = (out["best_ask"] * out["bid1_size"] + out["best_bid"] * out["ask1_size"]) / denom
        out["microprice"] = mp.fillna(out["mid"])

    # ofi_l1 (use project function if available; it should take two Series)
    if "ofi_l1" not in out:
        if _ofi_l1_project is not None:
            out["ofi_l1"] = _ofi_l1_project(
                out["bid1_size"].astype(float),
                out["ask1_size"].astype(float),
            )
        else:
            out["ofi_l1"] = _fallback_ofi_l1(
                out["bid1_size"].astype(float),
                out["ask1_size"].astype(float),
            )

    # ret_mid (log returns of mid)
    if "ret_mid" not in out:
        out["ret_mid"] = np.log(out["mid"]).diff().fillna(0.0)

    return out


# ------------------------------- plots ------------------------------------- #


def plot_spread_hist(df: pd.DataFrame, opts: SaveOpts, bins: int = 60):
    """Histogram of relative spread (bps)."""
    fig = plt.figure()
    ax = fig.gca()
    ax.hist(df["spread_bps"].astype(float).values, bins=bins)
    ax.set_title("Relative Spread (bps)")
    ax.set_xlabel("bps")
    ax.set_ylabel("count")
    _save_or_show(fig, "spread_bps_hist", opts)


def plot_microprice_premium(df: pd.DataFrame, opts: SaveOpts):
    """Time series of microprice premium: (microprice - mid)."""
    prem = (df["microprice"] - df["mid"]).astype(float)
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(prem.values)
    ax.set_title("Microprice Premium (microprice - mid)")
    ax.set_xlabel("index")
    ax.set_ylabel("price units")
    _save_or_show(fig, "microprice_premium", opts)


def plot_ofi_vs_returns(df: pd.DataFrame, window_corr: int, save: SaveOpts):
    """Plot OFI vs mid returns and their rolling correlation."""
    ofi = df["ofi_l1"].astype(float)
    ret = df["ret_mid"].astype(float)

    # 2 panels: time series and rolling corr
    # (Keep separate figures per style guide)
    fig1 = plt.figure()
    ax1 = fig1.gca()
    ax1.plot(ofi.values)
    ax1.set_title("OFI (L1)")
    ax1.set_xlabel("index")
    ax1.set_ylabel("OFI")
    _save_or_show(fig1, "ofi_l1", save)

    fig2 = plt.figure()
    ax2 = fig2.gca()
    ax2.plot(ret.values)
    ax2.set_title("Mid Log-Returns")
    ax2.set_xlabel("index")
    ax2.set_ylabel("ret_mid")
    _save_or_show(fig2, "ret_mid", save)

    if window_corr > 1:
        corr = pd.Series(ofi).rolling(window_corr).corr(pd.Series(ret)).fillna(0.0)
        fig3 = plt.figure()
        ax3 = fig3.gca()
        ax3.plot(corr.values)
        ax3.set_title(f"Rolling Corr(OFI, ret_mid) — window={window_corr}")
        ax3.set_xlabel("index")
        ax3.set_ylabel("corr")
        _save_or_show(fig3, "ofi_vs_returns_rolling_corr", save)


def plot_rv_event(df: pd.DataFrame, window: int, save: SaveOpts):
    """Event-time realized variance of mid returns over a rolling window."""
    ret2 = df["ret_mid"].astype(float) ** 2
    rv = ret2.rolling(window).sum().fillna(0.0)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(rv.values)
    ax.set_title(f"Event-time Realized Variance (window={window})")
    ax.set_xlabel("index")
    ax.set_ylabel("RV")
    _save_or_show(fig, "rv_event", save)


def _extract_depth(df: pd.DataFrame, side: str, K: int) -> pd.DataFrame:
    """Extract top-K depth as a tidy DataFrame with columns ['level', 'price', 'size'] for one side."""
    levels = []
    for k in range(1, K + 1):
        px_col = f"{side}{k}_price"
        sz_col = f"{side}{k}_size"
        if px_col in df.columns and sz_col in df.columns:
            levels.append(
                pd.DataFrame(
                    {
                        "level": [k] * len(df),
                        "price": df[px_col].astype(float).values,
                        "size": df[sz_col].astype(float).values,
                    }
                )
            )
    if not levels:
        return pd.DataFrame(columns=["level", "price", "size"])
    stacked = pd.concat(levels, axis=0, ignore_index=True)
    return stacked


def plot_depth_curve(df: pd.DataFrame, K: int, save: SaveOpts):
    """Plot average depth curve (price vs avg size) for bids/asks separately."""
    bids = _extract_depth(df, "bid", K)
    asks = _extract_depth(df, "ask", K)

    fig1 = plt.figure()
    ax1 = fig1.gca()
    if not bids.empty:
        avg_bids = bids.groupby("level")["size"].mean()
        ax1.plot(avg_bids.index.values, avg_bids.values)
    ax1.set_title(f"Average Bid Depth vs Level (K={K})")
    ax1.set_xlabel("level")
    ax1.set_ylabel("avg size")
    _save_or_show(fig1, "avg_bid_depth_curve", save)

    fig2 = plt.figure()
    ax2 = fig2.gca()
    if not asks.empty:
        avg_asks = asks.groupby("level")["size"].mean()
        ax2.plot(avg_asks.index.values, avg_asks.values)
    ax2.set_title(f"Average Ask Depth vs Level (K={K})")
    ax2.set_xlabel("level")
    ax2.set_ylabel("avg size")
    _save_or_show(fig2, "avg_ask_depth_curve", save)


def plot_depth_slope(df: pd.DataFrame, K: int, save: SaveOpts):
    """Plot an approximate slope: Δsize across levels (bid/ask), averaged over time."""
    bids = _extract_depth(df, "bid", K)
    asks = _extract_depth(df, "ask", K)

    def slope(series: pd.Series) -> pd.Series:
        # finite difference along the level axis
        s = series.sort_index()
        return s.diff().fillna(0.0)

    fig1 = plt.figure()
    ax1 = fig1.gca()
    if not bids.empty:
        avg_bids = bids.groupby("level")["size"].mean().sort_index()
        ax1.plot(avg_bids.index.values, slope(avg_bids).values)
    ax1.set_title(f"Bid Depth Slope (Δsize per level, K={K})")
    ax1.set_xlabel("level")
    ax1.set_ylabel("Δsize")
    _save_or_show(fig1, "bid_depth_slope", save)

    fig2 = plt.figure()
    ax2 = fig2.gca()
    if not asks.empty:
        avg_asks = asks.groupby("level")["size"].mean().sort_index()
        ax2.plot(avg_asks.index.values, slope(avg_asks).values)
    ax2.set_title(f"Ask Depth Slope (Δsize per level, K={K})")
    ax2.set_xlabel("level")
    ax2.set_ylabel("Δsize")
    _save_or_show(fig2, "ask_depth_slope", save)


# ------------------------------- orchestration ----------------------------- #


def run_advanced_plots(
    path: Path,
    outdir: Optional[Path] = None,
    save: bool = True,
    depth_k: int = 10,
    rv_window: int = 100,
    corr_window: int = 100,
    tail: Optional[int] = None,
):
    """Load data, ensure metrics, and render advanced analytics plots.

    Parameters
    ----------
    path : Path
        Input CSV or Parquet file with the flat snapshot schema.
    outdir : Optional[Path]
        Output directory for PNGs; default is plots/<input-stem>/ when saving.
    save : bool
        If True, save PNGs; otherwise show interactively.
    depth_k : int
        Top-K levels for depth plots.
    rv_window : int
        Rolling window for event-time realized variance.
    corr_window : int
        Rolling window for OFI vs returns correlation.
    tail : Optional[int]
        If provided, only the last N rows are used (speed).
    """
    path = Path(path)
    log.info("Loading: %s", path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if tail:
        df = df.tail(int(tail)).reset_index(drop=True)
        log.info("Using tail=%d (rows now %d)", int(tail), len(df))
    else:
        log.info("Rows loaded: %d", len(df))

    df = _ensure_metrics(df)
    log.info("Ensured metrics: mid, spread_bps, microprice, ofi_l1, ret_mid")

    if outdir is None and save:
        outdir = Path("plots") / path.stem

    opts = SaveOpts(save=save, outdir=outdir)
    log.info("Plotting → save=%s, outdir=%s", save, str(outdir) if outdir else None)

    plot_spread_hist(df, opts)
    plot_microprice_premium(df, opts)
    plot_ofi_vs_returns(df, window_corr=corr_window, save=opts)
    plot_rv_event(df, window=rv_window, save=opts)
    plot_depth_curve(df, K=depth_k, save=opts)
    plot_depth_slope(df, K=depth_k, save=opts)

    if not save:
        log.info("Showing plots interactively…")
        plt.show()
    else:
        log.info("Saved all plots to %s", outdir)
