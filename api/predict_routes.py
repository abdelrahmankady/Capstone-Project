from __future__ import annotations

import os
from pathlib import Path

from flask import Blueprint, jsonify, request

# ---------------------------------------------------------------------------
# Blueprint definition
# ---------------------------------------------------------------------------
predict_bp = Blueprint("predict", __name__, url_prefix="/api")


@predict_bp.route("/predict", methods=["POST"])
def predict():
    """Handle EEG file upload and return ML prediction results.

    Expects:
        multipart/form-data with a field named 'file' containing an .edf file.

    Returns:
        JSON dict conforming to EEG_ANALYSIS_SCHEMA on success, or a JSON
        error object with an appropriate HTTP status code on failure.
    """
    # --- Validate that a file was sent ---
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send a multipart form with a 'file' field."}), 400

    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        return jsonify({"error": "Empty filename."}), 400

    # --- Validate file extension ---
    filename = file.filename
    if not filename.lower().endswith(".edf"):
        return jsonify({"error": f"Unsupported file type '{Path(filename).suffix}'. Only .edf files are accepted."}), 400

    # --- Save the uploaded file to EEG_SCANS_PATH ---
    eeg_scans_path = os.getenv("EEG_SCANS_PATH", "./data/eeg_scans")
    save_dir = Path(eeg_scans_path).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / filename
    file.save(str(save_path))

    try:
        # --- Import and run the ML prediction pipeline ---
        from ml.epiwave_predict_api import run_prediction

        result = run_prediction(str(save_path))

        # --- Validate the result against the schema before returning ---
        from api.shared.eeg_schema import validate_eeg_output

        validate_eeg_output(result)

        # --- Automatically ingest into the RAG knowledge base ---
        # This ensures the chatbot can always answer questions about the
        # latest scan, regardless of whether the frontend's second call
        # (/api/eeg/analyze) succeeds or not.
        try:
            from rag.ingest import run_ingestion_from_analysis
            run_ingestion_from_analysis(result)
            print(f"[predict] Auto-ingested '{result['filename']}' into ChromaDB")
        except Exception as ingest_exc:
            # Log but don't fail the prediction response
            print(f"[predict] Auto-ingestion warning: {ingest_exc}")

        return jsonify(result), 200

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500
