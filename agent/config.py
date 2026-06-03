import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(ENV_PATH)

ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"
DATASET_LOG_PATH = DATA_DIR / "dataset_log.csv"
API_AUDIT_LOG_PATH = DATA_DIR / "api_audit_log.csv"
TRAINING_DATA_PATH = DATA_PROCESSED_DIR / "training_data.csv"
FETCH_STATE_PATH = DATA_PROCESSED_DIR / "fetch_state.json"
SAMPLE_DATA_PATH = ROOT_DIR / "sample_data.csv"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
KAGGLE_CONFIG_DIR = DATA_RAW_DIR / ".kaggle"

MAX_DOWNLOAD_SIZE_MB = 25
MIN_SEQUENCE_LENGTH = 30
FETCH_PAGE_WINDOW = 4
FETCH_PAGE_CYCLE_LIMIT = 20
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def ensure_directories() -> None:
    """Create runtime directories without touching ignored secret files."""
    for path in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)
    KAGGLE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def env_status() -> dict:
    """Return safe environment status without exposing secret values."""
    return {
        "ENTREZ_EMAIL": bool(ENTREZ_EMAIL),
        "KAGGLE_USERNAME": bool(KAGGLE_USERNAME),
        "KAGGLE_KEY": bool(KAGGLE_KEY),
    }


ensure_directories()

os.environ.setdefault("KAGGLE_CONFIG_DIR", str(KAGGLE_CONFIG_DIR))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
if KAGGLE_USERNAME:
    os.environ.setdefault("KAGGLE_USERNAME", KAGGLE_USERNAME)
if KAGGLE_KEY:
    os.environ.setdefault("KAGGLE_KEY", KAGGLE_KEY)
