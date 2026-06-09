# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

"""Configure colored logging for the project."""

import logging
import colorlog


def setup_logger(name: str = None) -> logging.Logger:
    """Set up a colored logger.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance with color configuration
    """
    # create a logger with the specified name
    logger = colorlog.getLogger(name or __name__)

    # avoid reconfiguring if the logger already has handlers
    if logger.handlers:
        return logger

    # configure the logger to use colorlog
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s: %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
            style="%",
        )
    )

    # add the handler to the logger and set the logging level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # avoid propagating the log messages to the parent logger
    logger.propagate = False

    return logger
