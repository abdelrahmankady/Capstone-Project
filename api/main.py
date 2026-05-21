from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Load environment from the project root .env BEFORE any other imports
#    so that every module (config.py, vectorize.py, etc.) picks up the
#    correct values.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH, override=True)

# ---------------------------------------------------------------------------
# 2. Ensure the chatbot/ directory is on sys.path so its internal imports
#    (``from config import ...``, ``from agents.retrieve import ...``, etc.)
#    resolve correctly. This must happen before Flask or the blueprints
#    try to import any chatbot code.
# ---------------------------------------------------------------------------
CHATBOT_DIR = str(PROJECT_ROOT / "chatbot")
if CHATBOT_DIR not in sys.path:
    sys.path.insert(0, CHATBOT_DIR)

# Also add the project root itself so ``from ml.epiwave_predict_api import ...``
# and ``from api.shared.eeg_schema import ...`` resolve.
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

# ---------------------------------------------------------------------------
# 3. Flask app factory
# ---------------------------------------------------------------------------
from flask import Flask, send_from_directory
from flask_cors import CORS


def create_app() -> Flask:
    """Create and configure the Flask application."""

    # The frontend/ directory is served as static files.
    frontend_dir = PROJECT_ROOT / "frontend"

    app = Flask(
        __name__,
        static_folder=str(frontend_dir / "static"),
        static_url_path="/static",
    )

    # --- CORS: allow the frontend (served from same origin or file://) ---
    CORS(app)

    # --- Register blueprints ---
    from api.predict_routes import predict_bp
    from api.chatbot_routes import chat_bp
    from api.eeg_routes import eeg_bp

    app.register_blueprint(predict_bp)    # /api/predict
    app.register_blueprint(chat_bp)       # /api/chat
    app.register_blueprint(eeg_bp)        # /api/eeg/analyze

    # --- Serve frontend HTML pages ---
    @app.route("/")
    def index():
        """Serve the landing page."""
        return send_from_directory(str(frontend_dir), "index.html")

    @app.route("/upload")
    @app.route("/upload.html")
    def upload_page():
        """Serve the upload page without caching."""
        response = send_from_directory(str(frontend_dir), "upload.html")
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.route("/brain")
    @app.route("/brain.html")
    def brain_page():
        """Serve the 3D brain visualization page."""
        return send_from_directory(str(frontend_dir), "brain.html")

    # --- Catch-all for any other frontend files (e.g. .json, .xml, .obj) ---
    @app.route("/<path:filename>")
    def serve_frontend_file(filename):
        """Serve any file from the frontend directory tree."""
        # First try the frontend root
        file_path = frontend_dir / filename
        if file_path.exists() and file_path.is_file():
            return send_from_directory(str(frontend_dir), filename)
        # Then try the static subdirectory
        static_path = frontend_dir / "static" / filename
        if static_path.exists() and static_path.is_file():
            return send_from_directory(str(frontend_dir / "static"), filename)
        return {"error": "Not found"}, 404

    # --- Health check endpoint ---
    @app.route("/api/health", methods=["GET"])
    def health():
        """Simple health check."""
        return {"status": "ok", "project": "EpiWave"}

    return app


# ---------------------------------------------------------------------------
# 4. Entry point — run the dev server
# ---------------------------------------------------------------------------
app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", "5000"))

    print("=" * 60)
    print("  EpiWave — Unified API Server")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Frontend     : {PROJECT_ROOT / 'frontend'}")
    print(f"  Port         : {port}")
    print("=" * 60)

    app.run(host="0.0.0.0", port=port, debug=True)
