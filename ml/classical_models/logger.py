"""Training logger for classical ML experiments.

Provides a namespaced logger that writes structured log messages. Host
applications (backend, Kaggle notebook, scripts) remain in control of handler
configuration; this module only attaches a NullHandler by default.
"""

from __future__ import annotations

import logging

LOGGER_NAME = "sentinelgraph.ml.classical_models"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return the classical models logger, creating it if necessary.

    The logger emits at WARNING by default and has no handlers attached, so
    it is safe to import in library contexts without producing side effects.
    Host applications should configure handlers/levels as needed.
    """
    import logging as _logging

    logger = _logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_logging.NullHandler())
    logger.setLevel(_logging.WARNING)
    return logger
