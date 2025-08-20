import pytest

from market_microstructure_toolkit.setup_log import setup_logging


def test_setup_logging_creates_logfile(tmp_path):
    """
    Test that setup_logging:
    1. Detects the calling script name correctly
    2. Creates a log file in the specified directory
    3. Writes log messages to the file
    """

    # Temporary log directory (pytest gives us tmp_path)
    log_dir = tmp_path / "logs"

    # Create logger (name=None so it auto-detects test script name)
    logger = setup_logging(log_dir=str(log_dir), name=None)

    # Log some messages
    logger.info("This is an info message")
    logger.error("This is an error message")

    # Expected logfile path
    log_file = log_dir / "test_setup_log.log"

    # === Assertions ===
    assert log_file.exists(), f"Log file {log_file} was not created."

    content = log_file.read_text()
    assert "This is an info message" in content
    assert "This is an error message" in content
    assert "INFO" in content
    assert "ERROR" in content


if __name__ == "__main__":
    # Run the test manually (optional)
    pytest.main([__file__, "-v", "-s"])
