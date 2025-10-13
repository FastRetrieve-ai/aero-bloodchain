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

# Ensure directories exist
DB_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")  # Will use GPT-5 when available

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_DIR}/bloodchain.db")

# Application Settings
APP_TITLE = "熱血飛騰：血品供應韌性系統"
APP_SUBTITLE = "Emergency Blood Chain System"

# Map Configuration
DEFAULT_MAP_CENTER = [25.0330, 121.5654]  # Taipei coordinates
DEFAULT_MAP_ZOOM = 11

# Embedding Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-large"

# Manual file path
EMERGENCY_MANUAL_PATH = DATA_DIR / "emergency-patient-rescue-process.md"

