from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np
import pywt
from scipy.ndimage import zoom as ndimage_zoom

# ---------------------------------------------------------------------------
# Lazy-load tensorflow to keep import time low when the module is only
# imported for type-checking or schema reference.
# ---------------------------------------------------------------------------
_model = None  # Will be loaded on first call to run_prediction()


# =============================
# CONFIGURATION (matches training)
# =============================
# Model path is read from env or falls back to the default location.
MODEL_DIR = Path(os.getenv(
    "EPIWAVE_MODEL_DIR",
    str(Path(__file__).resolve().parent.parent / "EpiWave_Model" / "models"),
))
MODEL_FILENAME = "epiwave_multiclass_mobilenet_best.keras"

# Signal processing constants — must match training exactly.
SELECTED_CHANNELS = ["FP1-F7", "F7-T7", "T7-P7"]
WINDOW_SECONDS = 4
OVERLAP_SECONDS = 2
LOW_FREQ = 0.5
HIGH_FREQ = 40
NOTCH_FREQ = 60
IMAGE_SIZE = (224, 224)

# Class label mapping — index order matches the trained model.
CLASS_NAMES = ["normal", "preictal", "seizure"]


# =============================
# MODEL LOADING
# =============================

def _load_model():
    """Load the trained Keras model into module-level cache.

    Called lazily on the first prediction request so the import itself
    stays lightweight.
    """
    global _model
    if _model is not None:
        return _model

    import tensorflow as tf
    from tensorflow.keras.models import load_model

    model_path = MODEL_DIR / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Make sure you have run the training script first."
        )

    print(f"[ml] Loading model from {model_path} ...")
    _model = load_model(str(model_path))
    print("[ml] Model loaded successfully.")
    return _model


# =============================
# SIGNAL PROCESSING
# =============================

def _load_and_preprocess_edf(file_path: str | Path):
    """Load an EDF file and return the averaged filtered signal + sfreq.

    Returns
    -------
    signal : np.ndarray
        1-D averaged signal across selected channels.
    sfreq : int
        Sampling frequency in Hz.
    raw : mne.io.Raw
        The MNE Raw object (for metadata extraction).
    """
    raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)

    # Pick the training-spec channels if available, fallback otherwise.
    available = [ch for ch in SELECTED_CHANNELS if ch in raw.ch_names]
    if not available:
        eeg_channels = [
            ch for ch in raw.ch_names
            if "EEG" in ch or "FP" in ch or "C" in ch
        ]
        available = eeg_channels[:3]

    raw.pick_channels(available)
    raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)
    raw.notch_filter(NOTCH_FREQ, verbose=False)

    data = raw.get_data()
    sfreq = int(raw.info["sfreq"])
    signal = np.mean(data, axis=0)  # Average across channels

    return signal, sfreq, raw


def _extract_segments(signal: np.ndarray, sfreq: int) -> list[dict]:
    """Extract 4-second windows with 50 % overlap."""
    window_samples = WINDOW_SECONDS * sfreq
    step_samples = int(window_samples * (1 - OVERLAP_SECONDS / WINDOW_SECONDS))

    segments: list[dict] = []
    for start in range(0, len(signal) - window_samples, step_samples):
        segments.append({
            "segment": signal[start : start + window_samples],
            "start_sec": start / sfreq,
            "end_sec": (start + window_samples) / sfreq,
        })
    return segments


def _generate_cwt_array(segment: np.ndarray, sfreq: int) -> np.ndarray:
    """Generate a CWT scalogram as a normalised numpy array."""
    frequencies = np.linspace(LOW_FREQ, HIGH_FREQ, 64)
    wavelet = "morl"
    scales = pywt.central_frequency(wavelet) * sfreq / frequencies

    coefficients, _ = pywt.cwt(segment, scales, wavelet, sampling_period=1 / sfreq)

    power = np.abs(coefficients)
    power = np.log1p(power)

    # Normalise to [0, 1]
    p_min, p_max = power.min(), power.max()
    if p_max > p_min:
        power = (power - p_min) / (p_max - p_min)
    else:
        power = np.zeros_like(power)

    # Resize both frequency and time axes to 224 to match the model input shape.
    zoom_factor_y = IMAGE_SIZE[0] / power.shape[0]
    zoom_factor_x = IMAGE_SIZE[1] / power.shape[1]
    power = ndimage_zoom(power, (zoom_factor_y, zoom_factor_x), order=1)

    return power


def _predict_segment(segment: np.ndarray, sfreq: int) -> dict:
    """Run the loaded model on a single EEG window."""
    model = _load_model()
    scalogram = _generate_cwt_array(segment, sfreq)

    # Stack to 3-channel (224, 224, 3) to match MobileNetV2 input shape.
    scalogram_3ch = np.stack([scalogram, scalogram, scalogram], axis=-1)
    input_array = np.expand_dims(scalogram_3ch, axis=0)

    probabilities = model.predict(input_array, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities))

    return {
        "label": CLASS_NAMES[predicted_idx],
        "probabilities": {
            "normal": float(probabilities[0]),
            "preictal": float(probabilities[1]),
            "seizure": float(probabilities[2]),
        },
    }


# =============================
# PUBLIC ENTRY POINT
# =============================

def run_prediction(edf_path: str) -> dict:
    """Run the full prediction pipeline on an EDF file.

    Parameters
    ----------
    edf_path : str
        Absolute or relative path to the .edf file.

    Returns
    -------
    dict
        Conforms to ``api.shared.eeg_schema.EEG_ANALYSIS_SCHEMA``.
        All required fields are populated, including ``summary_text``.
    """
    path = Path(edf_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"EDF file not found: {path}")

    # --- Load & preprocess ---
    signal, sfreq, raw = _load_and_preprocess_edf(path)
    segments = _extract_segments(signal, sfreq)

    if not segments:
        raise ValueError("No valid segments could be extracted from the EDF file.")

    # --- Per-segment predictions (Optimised with chunked batching) ---
    seizure_count = 0
    preictal_count = 0
    normal_count = 0
    max_seizure_prob = 0.0
    max_preictal_prob = 0.0
    seizure_events: list[dict] = []

    model = _load_model()
    BATCH_SIZE = 64
    
    for i in range(0, len(segments), BATCH_SIZE):
        batch_segments = segments[i : i + BATCH_SIZE]
        
        # Prepare inputs for this batch
        input_arrays = []
        for seg in batch_segments:
            scalogram = _generate_cwt_array(seg["segment"], sfreq)
            scalogram_3ch = np.stack([scalogram, scalogram, scalogram], axis=-1)
            # Use float32 to save memory
            input_arrays.append(scalogram_3ch.astype(np.float32))
            
        batch_input = np.array(input_arrays)
        
        # Predict the batch all at once
        batch_probs = model.predict(batch_input, batch_size=BATCH_SIZE, verbose=0)
        
        # Process results
        for j, seg in enumerate(batch_segments):
            probs = batch_probs[j]
            predicted_idx = int(np.argmax(probs))
            label = CLASS_NAMES[predicted_idx]
            
            seizure_prob = float(probs[2])
            preictal_prob = float(probs[1])
            
            if label == "seizure":
                seizure_count += 1
                max_seizure_prob = max(max_seizure_prob, seizure_prob)
                seizure_events.append({
                    "start_sec": round(seg["start_sec"], 2),
                    "end_sec": round(seg["end_sec"], 2),
                    "channel": ", ".join(raw.ch_names),
                    "confidence": round(seizure_prob, 3),
                    "energy_ratio": 0.0,
                })
            elif label == "preictal":
                preictal_count += 1
                max_preictal_prob = max(max_preictal_prob, preictal_prob)
            else:
                normal_count += 1

    # --- Overall prediction ---
    if seizure_count > 0:
        prediction_label = "seizure"
        confidence_score = max_seizure_prob
    elif preictal_count > 0:
        prediction_label = "preictal"
        confidence_score = max_preictal_prob
    else:
        prediction_label = "normal"
        # Average normal probability across all segments
        confidence_score = (
            sum(1.0 for _ in range(normal_count)) / len(segments)
            if segments else 0.0
        )
        # Re-calculate as average of actual normal probs (re-run is expensive,
        # so use the ratio of normal segments as a proxy).
        confidence_score = normal_count / len(segments) if segments else 0.0

    # --- Wave patterns (simplified — dominant bands per available channel) ---
    wave_patterns: list[dict] = []
    for ch_name in raw.ch_names:
        wave_patterns.append({
            "channel": ch_name,
            "dominant_band": "unknown",
            "band_power": {},
        })

    # --- Build summary text ---
    summary_lines = [
        f"EEG Scan Analysis Summary for '{path.name}'",
        f"Recording duration: {raw.times[-1]:.1f} seconds across {len(raw.ch_names)} channel(s).",
        f"Segments analyzed: {len(segments)} ({seizure_count} seizure, "
        f"{preictal_count} preictal, {normal_count} normal).",
    ]

    if seizure_events:
        summary_lines.append(f"Candidate seizure events: {len(seizure_events)}.")
        for idx, evt in enumerate(seizure_events, start=1):
            summary_lines.append(
                f"  Event {idx}: {evt['start_sec']}s – {evt['end_sec']}s "
                f"(confidence={evt['confidence']})"
            )
    else:
        summary_lines.append("No seizure events detected.")

    summary_lines.append(f"Overall prediction: {prediction_label} "
                         f"(confidence: {confidence_score:.1%}).")
    summary_lines.append(
        "\nDISCLAIMER: This analysis is for educational and informational purposes only. "
        "It does not constitute a medical diagnosis. Consult a qualified neurologist."
    )
    summary_text = "\n".join(summary_lines)

    # --- Assemble output dict matching EEG_ANALYSIS_SCHEMA ---
    return {
        "filename": path.name,
        "scan_date": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(float(raw.times[-1]), 2),
        "num_channels": len(raw.ch_names),
        "channel_names": list(raw.ch_names),
        "sampling_frequency": float(sfreq),
        "seizure_events": seizure_events,
        "spike_count": seizure_count,  # Using seizure segment count as proxy
        "wave_patterns": wave_patterns,
        "prediction_label": prediction_label,
        "confidence_score": round(confidence_score, 4),
        "summary_text": summary_text,
    }
