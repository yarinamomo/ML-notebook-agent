import logging
import os
from pathlib import Path

from rich.logging import RichHandler

from src.utils.ui import console


def _setup_root_logger() -> None:
    logger = logging.getLogger("mlnotebookagent")
    logger.setLevel(os.getenv("NOTEBOOK_AGENT_LOG_LEVEL", "ERROR").upper())
    _handler = RichHandler(
        show_path=False,
        show_time=False,
        show_level=False,
        markup=True,
        console=console,
    )
    _formatter = logging.Formatter(
        "[%(threadName)s]%(name)s: %(levelname)s: %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)


def add_file_handler(
    path: Path | str, level: int = logging.DEBUG, *, print_path: bool = True
) -> None:
    logger = logging.getLogger("mlnotebookagent")
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(threadName)s]%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if print_path:
        print(f"Logging to '{path}'")


_setup_root_logger()
logger = logging.getLogger("mlnotebookagent")


__all__ = ["logger"]
