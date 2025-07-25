import logging
import os
from pathlib import Path

log_level = logging.DEBUG
logger = logging.getLogger("PEKA")
# check if logger has been initialized
if not logger.hasHandlers():
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# get current location
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = str(Path(REPO_DIR).parent)

# set environment variables
import sys
sys.path.append("External_models/HEST/src/")
sys.path.append("External_models/scFoundation/")
