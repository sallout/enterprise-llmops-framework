"""Application-owned logging; configuration and secrets must never be logged."""
import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    logger = logging.getLogger("policy_dissertation")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
