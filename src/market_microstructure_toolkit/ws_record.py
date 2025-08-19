# Created by LP
# Date: 2025-08-19
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

from __future__ import annotations
import asyncio
import csv
import time
import logging
from pathlib import Path
from typing import Dict, Any

from .setup_log import setup_logging
from .snapshot import fetch_order_book_snapshot
from .exchange import make_exchange

log = logging.getLogger(__name__)  # handlers set in main()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _maybe_init_csv(out: Path, header: list[str]) -> None:
    _ensure_parent(out)
    if not out.exists():
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)


def _flatten_to_row(snap: Dict[str, Any], depth: int) -> Dict[str, Any]:
    # same formatting style as record.py (10 decimals, blanks for padding)
    row: Dict[str, Any] = {
        "ts_ms": str(snap["ts_ms"]),
        "iso": snap["iso"],
        "exchange_id": snap.get("exchange_id") or "",
        "symbol": snap.get("symbol") or "",
        "book_level": snap.get("book_level") or "L2",
        "raw_nonce": "" if snap.get("raw_nonce") is None else str(snap["raw_nonce"]),
        "best_bid": "" if snap["best_bid"] is None else f'{snap["best_bid"]:.10f}',
        "best_ask": "" if snap["best_ask"] is None else f'{snap["best_ask"]:.10f}',
    }
    bids = (snap.get("bids") or [])[:depth]
    asks = (snap.get("asks") or [])[:depth]
    bids += [("", "")] * (depth - len(bids))
    asks += [("", "")] * (depth - len(asks))
    for i, (px, sz) in enumerate(bids, 1):
        row[f"bid{i}_price"] = "" if px == "" else f"{px:.10f}"
        row[f"bid{i}_size"] = "" if sz == "" else f"{sz:.10f}"
    for i, (px, sz) in enumerate(asks, 1):
        row[f"ask{i}_price"] = "" if px == "" else f"{px:.10f}"
        row[f"ask{i}_size"] = "" if sz == "" else f"{sz:.10f}"
    return row


def _header_for_depth(depth: int) -> list[str]:
    h = [
        "ts_ms",
        "iso",
        "exchange_id",
        "symbol",
        "book_level",
        "raw_nonce",
        "best_bid",
        "best_ask",
    ]
    for i in range(1, depth + 1):
        h += [f"bid{i}_price", f"bid{i}_size"]
    for i in range(1, depth + 1):
        h += [f"ask{i}_price", f"ask{i}_size"]
    return h


async def _stream_ccxtpro(
    exchange_id: str, symbol: str, depth: int, out: Path, seconds: int
):
    import ccxtpro  # requires user to install ccxt.pro

    ex = getattr(ccxtpro, exchange_id)({"enableRateLimit": True})
    # ccxt.pro uses .watch_order_book
    header = _header_for_depth(depth)
    _maybe_init_csv(out, header)
    start = time.time()
    written = 0
    try:
        while time.time() - start < seconds:
            ob = await ex.watch_order_book(symbol, limit=depth)
            # Build normalized snapshot compatible with record.py
            snap = {
                "ts_ms": ob.get("timestamp") or int(time.time() * 1000),
                "iso": time.strftime(
                    "%Y-%m-%dT%H:%M:%S+00:00",
                    time.gmtime(
                        (ob.get("timestamp") or int(time.time() * 1000)) / 1000
                    ),
                ),
                "best_bid": (ob.get("bids") or [[None, None]])[0][0],
                "best_ask": (ob.get("asks") or [[None, None]])[0][0],
                "bids": [
                    (float(px), float(sz)) for px, sz in (ob.get("bids") or [])[:depth]
                ],
                "asks": [
                    (float(px), float(sz)) for px, sz in (ob.get("asks") or [])[:depth]
                ],
                "symbol": symbol,
                "exchange_id": ex.id,
                "raw_nonce": ob.get("nonce"),
                "book_level": "L2",
            }
            row = _flatten_to_row(snap, depth)
            with out.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writerow(row)
            written += 1
    finally:
        await ex.close()
    log.info("ccxt.pro stream complete. Rows: %s → %s", written, out)


async def _stream_rest_fallback(
    exchange_id: str, symbol: str, depth: int, out: Path, seconds: int, hz: float
):
    # Use existing fetch_order_book_snapshot in an asyncio loop
    ex = make_exchange(exchange_id, default_type=None, timeout=10000)
    period = 1.0 / float(hz)
    header = _header_for_depth(depth)
    _maybe_init_csv(out, header)
    start = time.time()
    next_t = start
    written = 0
    while time.time() - start < seconds:
        now = time.time()
        if now < next_t:
            await asyncio.sleep(next_t - now)
        snap = fetch_order_book_snapshot(ex, symbol, depth=depth, book_level="L2")
        row = _flatten_to_row(snap, depth)
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writerow(row)
        written += 1
        next_t += period
    log.info("REST fallback stream complete. Rows: %s → %s", written, out)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Websocket (or REST-fallback) order-book recorder → CSV"
    )
    parser.add_argument(
        "--exchange", required=True, help="ccxt id (e.g., bybit, okx, binance)"
    )
    parser.add_argument(
        "--symbol", required=True, help="Unified symbol (e.g., ETH/USDT:USDT)"
    )
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument(
        "--hz", type=float, default=5.0, help="Only used for REST fallback"
    )
    parser.add_argument(
        "--out",
        default="",
        help="CSV path; default: data/ws_<exchange>_<sym>_<depth>_<secs>.csv",
    )
    args = parser.parse_args()

    log = setup_logging(name="ws_record")

    sym_sanitized = args.symbol.replace("/", "").replace(":", "")
    out = Path(
        args.out
        or f"data/ws_{args.exchange}_{sym_sanitized}_d{args.depth}_{args.seconds}s.csv"
    )

    log.info(
        "Starting stream: ex=%s sym=%s depth=%d secs=%d → %s",
        args.exchange,
        args.symbol,
        args.depth,
        args.seconds,
        out,
    )

    try:
        import ccxtpro  # noqa

        has_pro = True
    except Exception:
        has_pro = False
        log.warning("ccxt.pro not available → using REST fallback at %.2f Hz", args.hz)

    if has_pro:
        asyncio.run(
            _stream_ccxtpro(args.exchange, args.symbol, args.depth, out, args.seconds)
        )
    else:
        asyncio.run(
            _stream_rest_fallback(
                args.exchange, args.symbol, args.depth, out, args.seconds, args.hz
            )
        )


if __name__ == "__main__":
    main()
