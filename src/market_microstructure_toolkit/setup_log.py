from __future__ import annotations

# src/market_microstructure_toolkit/setup_log.py
import inspect
import logging
from pathlib import Path


def setup_logging(log_dir: str = "logs", name: str | None = None) -> logging.Logger:
    """
    Configure root logging (file + console) and return a named child logger.

    - Root logger gets handlers so any logger (logging.getLogger(__name__)) propagates here.
    - Returns a child logger named `name` (or the caller's module filename).
    - Log file at logs/<name>.log
    """
    if name is None:
        frame = inspect.stack()[1]
        name = Path(frame.filename).stem

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logfile = Path(log_dir) / f"{name}.log"

    # Formatter
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Root logger gets (fresh) handlers
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Clear existing handlers (so re-running a CLI doesn’t duplicate outputs)
    for h in list(root.handlers):
        root.removeHandler(h)

    fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(ch)

    # Return a child logger; let it propagate to root
    logger = logging.getLogger(name)
    logger.propagate = True  # ensure it bubbles up to root handlers
    return logger
