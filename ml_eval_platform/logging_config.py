import logging
import sys

from pythonjsonlogger.json import JsonFormatter


def configure_logging(level: str) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(run_id)s %(external_id)s %(stage)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
