from typing import Optional
import logging

from .data import ensure_path

# The default logging level for the entire app.
DEFAULT_LEVEL = logging.INFO


def init_logger(
    name: str,
    filename: str,
    level: Optional[int] = None,
    filemode: str = "w",
    encoding: str = "utf-8",
    format_: str = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
    datefmt: Optional[str] = "%y-%m-%d %H:%M",
    to_console: bool = False,
) -> logging.Logger:
    """Convenience method for initializing a Logger. Parameters are the same as basicConfig()."""
    logger = logging.getLogger(name)
    if level is None:
        logger.setLevel(DEFAULT_LEVEL)
    else:
        logger.setLevel(level)
    ensure_path(filename)
    handler = logging.FileHandler(filename=filename, mode=filemode, encoding=encoding)
    handler.setFormatter(logging.Formatter(fmt=format_, datefmt=datefmt))
    logger.addHandler(handler)
    if to_console:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(fmt=format_, datefmt=datefmt))
        logger.addHandler(console)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Returns the Logger with the given name."""
    return logging.getLogger(name)


class ConditionalPrinter:
    def __init__(self, condition: bool):
        """
        Prints only if 'condition' is True.
        Usage example:

            cprint = ConditionalPrinter(config.verbose)
            ...
            cprint(*args, **kwargs)  # Same args and kwargs as built-in print().
        """
        self.condition = condition

    def __call__(self, *args, **kwargs) -> None:
        if self.condition:
            print(*args, **kwargs)
