"""Logger setup for the ingestion module.

Creates a namespaced logger that applications can integrate with their own
logging configuration. No handlers are added here so that host applications
(repo backend, Kaggle notebook, scripts) remain in control of output.
"""

from __future__ import annotations

import logging

LOGGER_NAME = "sentinelgraph.preprocessing.ingestion"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return the ingestion logger, creating it if necessary.

    The logger emits at WARNING by default and has no handlers attached, so
    it is safe to import in library contexts without producing side effects.
    Host applications should configure handlers/levels as needed.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.WARNING)
    return logger
