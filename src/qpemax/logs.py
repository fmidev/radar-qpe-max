import logging


def streamlogger_setup(logger: logging.Logger, loglevel: int = logging.INFO) -> None:
    """Setup logger with StreamHandler."""
    logger.setLevel(loglevel)
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(loglevel)
        logger.addHandler(ch)