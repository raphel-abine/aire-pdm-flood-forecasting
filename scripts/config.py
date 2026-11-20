"""
config.py - shared paths and station constants for all scripts
===============================================================
Set two environment variables, or edit the defaults below:
  CAMELS_GB_DIR : folder that contains the CAMELS-GB v2 download
                  (Catchment_Attributes, Catchment_Timeseries, ...)
                  default: <repository>/data
  PDM_OUTPUT_DIR: folder where outputs (tables, figures, caches) are written
                  default: <repository>/outputs
Figures are saved with the non-interactive Agg backend, so the scripts run
from a terminal without opening windows.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("CAMELS_GB_DIR", REPO / "data"))
WORK = Path(os.environ.get("PDM_OUTPUT_DIR", REPO / "outputs"))
FIG = WORK / "figures"
WORK.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# pdm.py sits next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))

GAUGE = "27035"          # Aire at Kildwick Bridge (EA station F1503)
AREA_KM2 = 283.39        # catchment area from CAMELS-GB v2
