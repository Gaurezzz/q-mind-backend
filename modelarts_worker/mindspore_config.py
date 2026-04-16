import mindspore as ms
import os
import sys
import logging
from logging.handlers import RotatingFileHandler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

_LOG_DIR = os.path.join(CURRENT_DIR, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

def _setup_logging():
    root = logging.getLogger("miniqds")
    if root.handlers:
        return 

    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-45s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = RotatingFileHandler(
        os.path.join(_LOG_DIR, "miniqds.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    root.addHandler(ch)
    root.addHandler(fh)

_setup_logging()

def get_logger(name):
    return logging.getLogger(f"miniqds.{name}")

_device_logger = get_logger("device")

def configure_device():
    targets = ["Ascend", "GPU", "CPU"]
    for target in targets:
        try:
            if target == "CPU":
                ms.set_context(device_target=target, mode=ms.GRAPH_MODE)
            else:
                ms.set_context(device_target=target, device_id=0, mode=ms.GRAPH_MODE)
            
            _device_logger.info("MindSpore %s active on: %s", ms.__version__, target)
            return
        except Exception:
            continue
    
    ms.set_context(device_target="CPU")

configure_device()