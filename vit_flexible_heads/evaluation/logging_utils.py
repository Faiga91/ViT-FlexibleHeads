"""
Logging utilities for the evaluation module.
"""

import logging


def configure_logging(level=logging.DEBUG):
    """Configure logging for the evaluation module."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return logging.getLogger(__name__)
