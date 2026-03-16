import logging
import sys

_FORMAT = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(funcName)s:%(lineno)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LEVEL_COLORS = {"DEBUG": "36", "INFO": "32", "WARNING": "33", "ERROR": "31", "CRITICAL": "1;31"}
_DIM = "\033[2m"
_RESET = "\033[0m"


class ColoredNewLineFormatter(logging.Formatter):
    """Colored formatter that also aligns multi-line messages."""

    def format(self, record):
        c = _LEVEL_COLORS.get(record.levelname)
        if c:
            saved = record.levelname, record.filename, record.funcName, record.lineno
            bold = "1;" if record.levelname == "CRITICAL" else ""
            record.levelname = f"\033[{bold}{c}m{record.levelname}{_RESET}"
            record.filename = f"{_DIM}{record.filename}"
            record.funcName = f"{record.funcName}"
            record.lineno = f"{saved[3]}{_RESET}"
            msg = super().format(record)
            record.levelname, record.filename, record.funcName, record.lineno = saved
        else:
            msg = super().format(record)

        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = logging.getLogger("kdflow")
_default_handler = None


def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)
    global _default_handler
    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(logging.INFO)
        _root_logger.addHandler(_default_handler)
    fmt = ColoredNewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    _default_handler.setFormatter(fmt)
    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


_setup_logger()


def init_logger(name: str):
    # Use the same settings as above for root logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(_default_handler)
    logger.propagate = False
    return logger
