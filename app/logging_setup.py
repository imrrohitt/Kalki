from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging() -> None:
    """Send app INFO logs to the console. Uvicorn only configures its own loggers."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)
    app_logger.addHandler(handler)
    app_logger.propagate = False

    for name in ("httpx", "httpcore", "openai", "LiteLLM", "litellm"):
        logging.getLogger(name).setLevel(logging.WARNING)
