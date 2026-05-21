from __future__ import annotations

from flask import Blueprint, jsonify, request

# ---------------------------------------------------------------------------
# Blueprint definition
# ---------------------------------------------------------------------------
eeg_bp = Blueprint("eeg", __name__, url_prefix="/api/eeg")


@eeg_bp.route("/analyze", methods=["POST"])
def analyze():
    """Validate prediction output and ingest it into the RAG vector store.

    Expects:
        JSON body matching EEG_ANALYSIS_SCHEMA (the dict returned by
        POST /api/predict).

    Returns:
        JSON object with ingestion status on success, or a JSON error
        with the appropriate HTTP status code on failure.
    """
    # --- Parse JSON body ---
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # --- Validate against the shared schema ---
    from api.shared.eeg_schema import validate_eeg_output

    try:
        validate_eeg_output(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

    try:
        # --- Ingest into the RAG pipeline ---
        # The chatbot's rag/ingest.py expects a dict similar to the output
        # of eeg.analyzer.analyze_eeg(). We reshape our schema-conformant
        # dict to match the fields that ingest.py reads.
        from rag.ingest import run_ingestion_from_analysis

        # Build an analysis-like dict that ingest.py expects.
        # Our schema is a superset — ingest.py only reads specific keys.
        analysis_dict = {
            "filename": data["filename"],
            "duration_seconds": data["duration_seconds"],
            "num_channels": data["num_channels"],
            "channel_names": data["channel_names"],
            "sampling_frequency": data["sampling_frequency"],
            "seizure_events": data["seizure_events"],
            "spike_count": data["spike_count"],
            "wave_patterns": data["wave_patterns"],
            "channel_stats": data.get("channel_stats", []),
            "prediction_label": data["prediction_label"],
            "confidence_score": data["confidence_score"],
            "summary_text": data["summary_text"],
        }

        doc_count, chunk_count = run_ingestion_from_analysis(analysis_dict)

        return jsonify({
            "status": "success",
            "documents_ingested": doc_count,
            "chunks_stored": chunk_count,
            "filename": data["filename"],
        }), 200

    except Exception as exc:
        return jsonify({"error": f"Ingestion failed: {exc}"}), 500

