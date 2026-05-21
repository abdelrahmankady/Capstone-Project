from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Resolve project root — either this file lives inside chatbot/ (standalone
# mode) or we are imported from the unified API where the root .env was
# already loaded by api/main.py.  We try the root .env first, then fall
# back to chatbot/.env for backward compatibility.
# ---------------------------------------------------------------------------
CHATBOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CHATBOT_DIR.parent   # llm_api/

# Prefer the root-level .env so the unified project has a single config.
ROOT_ENV = PROJECT_ROOT / ".env"
LOCAL_ENV = CHATBOT_DIR / ".env"

if ROOT_ENV.exists():
    load_dotenv(ROOT_ENV, override=True)
elif LOCAL_ENV.exists():
    load_dotenv(LOCAL_ENV, override=True)

# ---------------------------------------------------------------------------
# Google Gemini API settings
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GOOGLE_LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.0-flash").strip()
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-2").strip()

# ---------------------------------------------------------------------------
# Storage paths — all resolved relative to PROJECT_ROOT so the unified
# project uses a single data directory, not per-subpackage dirs.
# ---------------------------------------------------------------------------
CHROMA_DB_PATH = str(
    (PROJECT_ROOT / os.getenv("CHROMA_DB_PATH", "./chroma_db")).resolve()
)
EEG_SCANS_PATH = PROJECT_ROOT / os.getenv("EEG_SCANS_PATH", "./data/eeg_scans")

# ---------------------------------------------------------------------------
# RAG retrieval parameters
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))

# ---------------------------------------------------------------------------
# Flask port (used by api/main.py; surfaced here for convenience)
# ---------------------------------------------------------------------------
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))

# ---------------------------------------------------------------------------
# Internal constants (not env-configurable)
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
VERIFY_THRESHOLD = 0.60

