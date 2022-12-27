import functools
import logging
import sys
from os import PathLike
from pathlib import Path
from typing import Optional, Union


def configure_logger(
    logger: logging.Logger,
    level: int = logging.WARNING,
    file: Optional[Union[str, PathLike]] = None,
    formatter: Optional[Union[str, logging.Formatter]] = None,
) -> logging.Logger:
    """Configures the logging levels and output for a pre-existing logger instance."""
    # configure the format of logging messages
    if formatter is None:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    elif isinstance(formatter, str):
        formatter = logging.Formatter(formatter, datefmt="%Y-%m-%d %H:%M:%S")
    assert isinstance(formatter, logging.Formatter), "formatter is invalid."

    # setup console logging
    console_log = logging.StreamHandler()  # console logger
    console_log.setLevel(level)
    console_log.setFormatter(formatter)
    logger.addHandler(console_log)

    # setup file logging, if a file path is provided
    if file is not None:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        try:
            file_log = logging.FileHandler(str(file))
        except IsADirectoryError as err:
            raise IsADirectoryError(f"Invalid path '{str(file)}' for file log") from err
        file_log.setLevel(logging.DEBUG)
        file_log.setFormatter(formatter)
        logger.addHandler(file_log)
        logger.debug(f"Initialising file log output to {str(file)}.")

    return logger


def setup_logger(
    name: Optional[str] = None,
    level: int = logging.WARNING,
    file: Optional[Union[str, PathLike]] = None,
    formatter: Optional[Union[str, logging.Formatter]] = None,
) -> logging.Logger:
    """Creates and/or gets a new logger instance 'name' and configures it for use."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return configure_logger(logger, level, file, formatter)
