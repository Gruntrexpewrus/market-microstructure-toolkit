# Created by LP
# Date: 2025-08-20
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

"""
Order-book recorder with websocket (ccxt.pro) or REST fallback.

Design:
- A single producer task collects snapshots (WS or REST).
- A single consumer task writes rows to CSV and keeps the file handle open
  for efficiency and ordering. A sentinel object cleanly shuts the writer down.
- REST producer aims for ~hz periodic sampling with backpressure (no drops).
- Websocket producer enqueues on every book update pushed by the exchange.

CLI entry point: `mmt-ws-record` (see pyproject’s [project.scripts]).
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

from .exchange import make_exchange
from .setup_log import setup_logging
from .snapshot import fetch_order_book_snapshot

log = logging.getLogger(__name__)  # handlers set in main()

# Sentinel placed on the queue to tell the writer to flush/close and exit.
_SENTINEL = object()


def _ensure_parent(path: Path) -> None:
    """Ensure the parent directory of `path` exists (mkdir -p)."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _header_for_depth(depth: int) -> list[str]:
    """Build a CSV header for best bid/ask plus depth-level ladders.

    Columns:
      ts_ms, iso, exchange_id, symbol, book_level, raw_nonce, best_bid, best_ask,
      bid1_price, bid1_size, ..., bid{depth}_price, bid{depth}_size,
      ask1_price, ask1_size, ..., ask{depth}_price, ask{depth}_size
    """
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


def _flatten_to_row(snap: Dict[str, Any], depth: int) -> Dict[str, Any]:
    """Normalize a snapshot dict into a flat CSV row with fixed-width ladders.

    - Formats numeric prices/sizes with 10 decimals.
    - Pads missing ladder levels with blanks so each row has identical length.
    """
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


def _open_csv_writer(out: Path, header: list[str]):
    """Open `out` for appending and write header if file is empty."""
    _ensure_parent(out)
    f = out.open("a", newline="")
    w = csv.DictWriter(f, fieldnames=header)
    try:
        empty = os.stat(out).st_size == 0
    except FileNotFoundError:
        empty = True
    if empty:
        w.writeheader()
    return f, w


async def _writer_consumer(
    queue: "asyncio.Queue[dict | object]", out: Path, header: list[str]
) -> int:
    """Drain rows from `queue` and append to CSV in arrival order.

    Keeps a single file handle open for efficiency. Returns total rows written.
    Stops when the `_SENTINEL` object is received.
    """
    f, w = _open_csv_writer(out, header)
    written = 0
    try:
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                queue.task_done()
                break
            w.writerow(item)  # type: ignore[arg-type]
            written += 1
            queue.task_done()
        f.flush()
        return written
    finally:
        f.close()


async def _producer_rest(
    queue: "asyncio.Queue[dict | object]",
    exchange_id: str,
    symbol: str,
    depth: int,
    seconds: int,
    hz: float,
) -> int:
    """Poll REST at ~`hz` for `seconds` and enqueue flattened rows.

    Uses `make_exchange` + `fetch_order_book_snapshot`. Backpressure is provided
    by the queue: if disk is slow, `queue.put()` awaits, so we don't drop data.
    Returns the number of produced rows (enqueued).
    """
    ex = make_exchange(exchange_id, default_type=None, timeout=10000)
    period = 1.0 / float(hz)
    start = time.time()
    next_t = start
    produced = 0
    try:
        while time.time() - start < seconds:
            now = time.time()
            if now < next_t:
                await asyncio.sleep(next_t - now)
            iter_start = time.time()
            snap = fetch_order_book_snapshot(ex, symbol, depth=depth, book_level="L2")
            row = _flatten_to_row(snap, depth)
            await queue.put(row)  # backpressure: don't drop
            produced += 1
            next_t += period
            # optional: drift diagnostics
            drift = time.time() - next_t
            if drift > 0.05:
                log.debug(
                    "REST loop late by %.0f ms (iter %.0f ms)",
                    drift * 1000,
                    (time.time() - iter_start) * 1000,
                )
    finally:
        # signal completion
        await queue.put(_SENTINEL)
    return produced


async def _producer_ws(
    queue: "asyncio.Queue[dict | object]",
    exchange_id: str,
    symbol: str,
    depth: int,
    seconds: int,
) -> int:
    """Consume websocket order-book updates via ccxt.pro and enqueue rows.

    Each pushed update from the exchange is normalized and queued.
    Returns the number of produced rows (enqueued).
    """
    import ccxtpro  # requires user to install ccxt.pro

    ex = getattr(ccxtpro, exchange_id)({"enableRateLimit": True})
    start = time.time()
    produced = 0
    try:
        while time.time() - start < seconds:
            ob = await ex.watch_order_book(symbol, limit=depth)
            ts_ms = ob.get("timestamp") or int(time.time() * 1000)
            snap = {
                "ts_ms": ts_ms,
                "iso": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(ts_ms / 1000)),
                "best_bid": (ob.get("bids") or [[None, None]])[0][0],
                "best_ask": (ob.get("asks") or [[None, None]])[0][0],
                "bids": [(float(px), float(sz)) for px, sz in (ob.get("bids") or [])[:depth]],
                "asks": [(float(px), float(sz)) for px, sz in (ob.get("asks") or [])[:depth]],
                "symbol": symbol,
                "exchange_id": ex.id,
                "raw_nonce": ob.get("nonce"),
                "book_level": "L2",
            }
            row = _flatten_to_row(snap, depth)
            await queue.put(row)
            produced += 1
    finally:
        await ex.close()
        await queue.put(_SENTINEL)
    return produced


async def _stream_ccxtpro(
    exchange_id: str, symbol: str, depth: int, out: Path, seconds: int
) -> None:
    """High-throughput WS capture with backpressure writer (no drops)."""
    header = _header_for_depth(depth)
    queue: asyncio.Queue[dict | object] = asyncio.Queue()

    consumer_task = asyncio.create_task(_writer_consumer(queue, out, header))
    try:
        produced = await _producer_ws(queue, exchange_id, symbol, depth, seconds)
        await consumer_task  # wait until writer drains and closes
        written = consumer_task.result()
        log.info(
            "ccxt.pro stream complete: produced=%s, written=%s → %s",
            produced,
            written,
            out,
        )
    except Exception:
        # try to shut down writer gracefully
        with contextlib.suppress(Exception):
            await queue.put(_SENTINEL)
            await consumer_task
        raise


async def _stream_rest_fallback(
    exchange_id: str, symbol: str, depth: int, out: Path, seconds: int, hz: float
) -> None:
    """Periodic REST polling with backpressure writer (no drops).

    Target row count is approximately `seconds * hz`. We log the achieved % and
    warn if we fall notably short (e.g., network or rate-limit issues).
    """
    header = _header_for_depth(depth)
    queue: asyncio.Queue[dict | object] = asyncio.Queue()

    consumer_task = asyncio.create_task(_writer_consumer(queue, out, header))
    try:
        produced = await _producer_rest(queue, exchange_id, symbol, depth, seconds, hz)
        await consumer_task
        written = consumer_task.result()
        target = int(seconds * hz)
        pct = 100.0 * written / max(1, target)
        log.info(
            "REST stream complete: produced=%s, written=%s (target=%s, %.1f%%) → %s",
            produced,
            written,
            target,
            pct,
            out,
        )
        if written < 0.9 * target:
            log.warning("Under target rate; consider reducing --hz or increasing timeout")
    except Exception:
        with contextlib.suppress(Exception):
            await queue.put(_SENTINEL)
            await consumer_task
        raise


def main() -> None:
    """CLI entry point.

    Examples:
        python -m market_microstructure_toolkit.ws_record \\
          --exchange bybit \\
          --symbol ETH/USDT:USDT \\
          --depth 5 \\
          --seconds 5 \\
          --hz 2
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Websocket (or REST-fallback) order-book recorder → CSV"
    )
    parser.add_argument("--exchange", required=True, help="ccxt id (e.g., bybit, okx, binance)")
    parser.add_argument("--symbol", required=True, help="Unified symbol (e.g., ETH/USDT:USDT)")
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--hz", type=float, default=5.0, help="Only used for REST fallback")
    parser.add_argument(
        "--out",
        default="",
        help="CSV path; default: data/ws_<exchange>_<sym>_d<depth>_<secs>_<hz>hz.csv",
    )
    args = parser.parse_args()

    setup_logging(name="ws_record")

    sym_sanitized = args.symbol.replace("/", "").replace(":", "")
    out = Path(
        args.out
        or f"data/ws_{args.exchange}_{sym_sanitized}_d{args.depth}_{args.seconds}s_{int(args.hz)}hz.csv"
    )

    log.info(
        "Starting stream: ex=%s sym=%s depth=%d secs=%d → %s",
        args.exchange,
        args.symbol,
        args.depth,
        args.seconds,
        out,
    )

    # Prefer ccxt.pro if present; otherwise REST fallback.
    try:
        import ccxtpro  # noqa: F401

        has_pro = True
    except Exception:
        has_pro = False
        log.warning("ccxt.pro not available → using REST fallback at %.2f Hz", args.hz)

    if has_pro:
        asyncio.run(_stream_ccxtpro(args.exchange, args.symbol, args.depth, out, args.seconds))
    else:
        asyncio.run(
            _stream_rest_fallback(
                args.exchange, args.symbol, args.depth, out, args.seconds, args.hz
            )
        )


if __name__ == "__main__":
    main()
