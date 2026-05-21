from __future__ import annotations

from flask import Blueprint, Response, jsonify, request, stream_with_context

# ---------------------------------------------------------------------------
# Blueprint definition
# ---------------------------------------------------------------------------
chat_bp = Blueprint("chat", __name__, url_prefix="/api")


@chat_bp.route("/chat", methods=["POST"])
def chat():
    """Run the retrieve → verify → respond pipeline and stream the response.

    Expects:
        JSON body:
        {
            "message": "user's question",
            "history": [
                ["previous user msg", "previous assistant msg"],
                ...
            ]
        }

    Returns:
        Streamed text/plain response. Each chunk is a token from the LLM.
        On error, returns a JSON error object with the appropriate status.
    """
    # --- Parse JSON body ---
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "'message' field is required and cannot be empty."}), 400

    # History is a list of [user_msg, assistant_msg] pairs.
    raw_history = data.get("history", [])
    history: list[tuple[str, str]] = []
    for entry in raw_history:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            history.append((str(entry[0]), str(entry[1])))

    try:
        # --- Import agents pipeline ---
        from agents.retrieve import retrieve_chunks
        from agents.verify import verify_chunks
        from agents.respond import stream_response

        # --- Step 1: Retrieve relevant chunks from ChromaDB ---
        filename = data.get("filename")
        retrieved_chunks = retrieve_chunks(message, filename=filename)

        # --- Step 2: Verify retrieved chunks (filter low-relevance) ---
        verified_chunks = verify_chunks(retrieved_chunks)

        # --- Step 3: Check if ChromaDB has any scan data at all ---
        # This tells the LLM whether scans have been analyzed, even if the
        # current query didn't match any chunks closely.
        from rag.vectorize import _get_chroma_collection

        try:
            has_scans = _get_chroma_collection().count() > 0
        except Exception:
            # If ChromaDB is empty or unreachable, treat as no scans.
            has_scans = False

        # --- Step 4: Stream the LLM response ---
        def generate():
            """Generator that yields tokens from the LLM response stream."""
            try:
                for token in stream_response(
                    verified_chunks,
                    history,
                    message,
                    has_scans=has_scans,
                ):
                    yield token
            except Exception as exc:
                # If the stream fails mid-way, yield an error marker.
                yield f"\n\n[Error: {exc}]"

        return Response(
            stream_with_context(generate()),
            mimetype="text/plain",
            headers={
                # Prevent buffering so tokens arrive in real-time.
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )

    except Exception as exc:
        return jsonify({"error": f"Chat pipeline failed: {exc}"}), 500
