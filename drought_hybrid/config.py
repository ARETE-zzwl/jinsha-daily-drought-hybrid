from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "station_daily"
CMIP_DIR = PROJECT_ROOT / "data" / "external" / "cmip_station_daily_extract"
DEM_SUMMARY_FILE = PROJECT_ROOT / "data" / "processed" / "station_metadata.csv"
OUT_DIR = PROJECT_ROOT / "results" / "runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_FEATURES = ["pr", "tmean", "tmax", "tmin", "et0", "wind", "rad_net", "rad_down"]
MODE_NAMES = ["low", "mid", "high"]
