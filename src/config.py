"""
Configuration management for the Blood Chain System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "database"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
DTM_DIR = DATA_DIR / "2024年版全臺灣20公尺網格數值地形模型DTM資料"

# Ensure directories exist
DB_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # Default general model

# Per-feature models (override via .env when needed)
# - SQL_QA_MODEL: for generating SQL from natural language
# - ANSWER_LLM_MODEL: for natural-language answer from SQL results
# - CHART_LLM_MODEL: for chart type inference
SQL_QA_MODEL = os.getenv("SQL_QA_MODEL", "gpt-4o")
ANSWER_LLM_MODEL = os.getenv("ANSWER_LLM_MODEL", "gpt-5")
CHART_LLM_MODEL = os.getenv("CHART_LLM_MODEL", "gpt-5-mini")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_DIR}/bloodchain.db")

# Application Settings
APP_TITLE = "熱血飛騰：血品供應韌性系統"
APP_SUBTITLE = "Emergency Blood Chain System"
APP_LOGIN_USERNAME = os.getenv("APP_LOGIN_USERNAME", "").strip()
APP_LOGIN_PASSWORD = os.getenv("APP_LOGIN_PASSWORD", "")

# Map Configuration
DEFAULT_MAP_CENTER = [25.0330, 121.5654]  # Taipei coordinates
DEFAULT_MAP_ZOOM = 11

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
GOOGLE_GEOCODE_ENDPOINT = (
    os.getenv(
        "GOOGLE_GEOCODE_ENDPOINT",
        "https://maps.googleapis.com/maps/api/geocode/json",
    )
    .strip()
    .rstrip("/")
)

# OpenRouteService Configuration
OPENROUTESERVICE_API_KEY = os.getenv("OPENROUTESERVICE_API_KEY", "").strip()
OPENROUTESERVICE_BASE_URL = (
    os.getenv("OPENROUTESERVICE_BASE_URL", "https://api.openrouteservice.org")
    .strip()
    .rstrip("/")
)

# Elevation / DTM Configuration
DEFAULT_DTM_FILENAME = "不分幅_台灣20MDEM(2024).tif"
DTM_TIF_PATH = Path(
    os.getenv("DTM_TIF_PATH", str(DTM_DIR / DEFAULT_DTM_FILENAME))
).expanduser()
DTM_SAMPLE_INTERVAL_M = float(os.getenv("DTM_SAMPLE_INTERVAL_M", "50"))

# Geocoder Configuration
GEOCODER_PROVIDER = os.getenv("GEOCODER_PROVIDER", "nominatim").strip().lower()
GEOCODER_USER_AGENT = os.getenv(
    "GEOCODER_USER_AGENT", "aero-bloodchain-uav-profile"
).strip()
GEOCODER_TIMEOUT = float(os.getenv("GEOCODER_TIMEOUT", "5"))
GEOCODER_RATE_LIMIT_SECONDS = float(os.getenv("GEOCODER_RATE_LIMIT_SECONDS", "1"))

# Embedding Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-large"

# Emergency manual (RAG) paths
# New structure: each manual page is a markdown file under this folder
# A complete PDF can also be present for download/preview
EMERGENCY_MANUAL_DIR = Path(
    os.getenv(
        "EMERGENCY_MANUAL_DIR",
        str(DATA_DIR / "emergency-patient-rescue-process"),
    )
)
EMERGENCY_MANUAL_PDF = EMERGENCY_MANUAL_DIR / "emergency-patient-rescue-process.pdf"

# Backward-compatibility: previous single-markdown path (unused in new flow)
EMERGENCY_MANUAL_PATH = DATA_DIR / "emergency-patient-rescue-process.md"
