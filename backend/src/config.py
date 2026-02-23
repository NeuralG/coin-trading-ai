import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE_PATH = os.path.join(BASE_DIR, "data", "coin-data-hourly.parquet")

TICKER = "BTC-USD"
