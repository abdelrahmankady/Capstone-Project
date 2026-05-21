from __future__ import annotations

from typing import Any, TypedDict


# ---------------------------------------------------------------------------
# Canonical EEG analysis output schema
# ---------------------------------------------------------------------------

class EEG_ANALYSIS_SCHEMA(TypedDict):
    """TypedDict representing the required shape of every EEG analysis result.

    Every field listed here MUST be present in the dict returned by
    ``ml.epiwave_predict_api.run_prediction()`` and accepted by
    ``api.eeg_routes`` for downstream RAG ingestion.
    """
    filename: str                    # Original .edf filename (basename)
    scan_date: str                   # ISO-8601 timestamp of when the scan was analyzed
    duration_seconds: float          # Total recording length in seconds
    num_channels: int                # Number of EEG channels in the recording
    channel_names: list[str]         # List of channel name strings
    sampling_frequency: float        # Sampling rate in Hz
    seizure_events: list[dict]       # List of detected seizure event dicts
    spike_count: int                 # Total spikes detected across all channels
    wave_patterns: list[dict]        # Detected frequency-band patterns per channel
    prediction_label: str            # Overall prediction: 'normal', 'preictal', or 'seizure'
    confidence_score: float          # Confidence score (0.0 – 1.0)
    summary_text: str                # Human-readable summary paragraph


# ---------------------------------------------------------------------------
# Required field definitions with expected types for runtime validation
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS: dict[str, type] = {
    "filename":             str,
    "scan_date":            str,
    "duration_seconds":     (int, float),   # accept both numeric types
    "num_channels":         int,
    "channel_names":        list,
    "sampling_frequency":   (int, float),
    "seizure_events":       list,
    "spike_count":          int,
    "wave_patterns":        list,
    "prediction_label":     str,
    "confidence_score":     (int, float),
    "summary_text":         str,
}


def validate_eeg_output(data: dict[str, Any]) -> dict[str, Any]:
    """Validate that *data* conforms to the EEG_ANALYSIS_SCHEMA.

    Parameters
    ----------
    data : dict
        The prediction output dict to validate.

    Returns
    -------
    dict
        The same *data* dict, unchanged, if validation passes.

    Raises
    ------
    ValueError
        If any required field is missing or has an incorrect type.
        The error message names the offending field for easy debugging.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a dict for EEG output validation, got {type(data).__name__}."
        )

    for field_name, expected_type in _REQUIRED_FIELDS.items():
        # --- Check presence ---
        if field_name not in data:
            raise ValueError(
                f"Missing required field '{field_name}' in EEG analysis output."
            )

        # --- Check type ---
        value = data[field_name]
        if not isinstance(value, expected_type):
            raise ValueError(
                f"Field '{field_name}' has incorrect type: "
                f"expected {expected_type}, got {type(value).__name__}."
            )

    return data
