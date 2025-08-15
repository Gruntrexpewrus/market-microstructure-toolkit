from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: str = "logs", name: str | None = None):
    """
    Set up logging with:
    - Automatic logger name based on the calling script's filename (unless 'name' is provided)
    - Log file saved in `log_dir` (default: 'logs/')
    - Logs written both to console and file
    - Default log level: INFO

    Parameters
    ----------
    log_dir : str
        Directory where the log file will be stored. Defaults to 'logs'.
    name : str | None
        Optional logger name. If None, it will automatically use the calling script's filename.

    Returns
    -------
    logging.Logger
        Configured logger object ready to use.
    """

    # If the user did not pass a 'name', try to detect the name of the calling script
    if name is None:
        # __file__ is the file where THIS function is defined (usually the logging module)
        name = Path(__file__).stem

        # If this function is IMPORTED into another file, __file__ will not be the caller
        # So we inspect the call stack and grab the filename of the caller instead
        import inspect

        frame = inspect.stack()[1]  # 0 = current frame, 1 = caller frame
        name = Path(frame.filename).stem  # Extract filename without extension

    # Ensure the logging directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Path for the log file, e.g., logs/scriptname.log
    logfile = Path(log_dir) / f"{name}.log"

    # Create (or get) the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Default logging level
    logger.propagate = False  # Prevent double logging if root logger also has handlers

    # Remove old handlers if this function is called multiple times in the same run
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Define log format: timestamp, log level, logger name, and message
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # File handler: writes logs to a file (overwrite mode)
    fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)

    # Console handler: prints logs to stdout
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    # Attach both handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
