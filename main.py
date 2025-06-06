from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import librosa
import numpy as np
import pyloudnorm as pyln
import io

app = FastAPI(title="Audio Analyzer API")

class AudioAnalysisResult(BaseModel):
    duration_seconds: float
    loudness_integrated: float  # LUFS
    loudness_true_peak: float
    rms: float
    dominant_freq_hz: float
    key_note: str
    tempo_bpm: float
    zero_crossing_rate_mean: float
    spectral_centroid_mean: float
    spectral_bandwidth_mean: float
    spectral_rolloff_mean: float
    chroma_dominant: str
    peak_amplitude: float

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def freq_to_note(freq):
    if freq <= 0 or np.isnan(freq):
        return "N/A"
    try:
        midi = int(round(69 + 12 * np.log2(freq / 440.0)))
        note = NOTE_NAMES[midi % 12]
        octave = midi // 12 - 1
        return f"{note}{octave}"
    except:
        return "N/A"

def analyze_audio(data: bytes):
    y, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)

    if len(y) < 4096:
        raise ValueError("Audio too short for reliable analysis.")

    duration = librosa.get_duration(y=y, sr=sr)

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    # true_peak = meter.true_peak_level(y)  ❌ НЕ СУЩЕСТВУЕТ
    true_peak = np.max(np.abs(y))  # ✅ заменили

    rms = float(np.sqrt(np.mean(y ** 2)))
    peak_amplitude = float(np.max(np.abs(y)))

    D = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    avg_spectrum = np.mean(D, axis=1)
    dominant_freq = freqs[np.argmax(avg_spectrum)]

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    key_note = freq_to_note(dominant_freq)

    zcr = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = float(np.mean(zcr))

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    spectral_centroid_mean = float(np.mean(spectral_centroid))
    spectral_bandwidth_mean = float(np.mean(spectral_bandwidth))
    spectral_rolloff_mean = float(np.mean(spectral_rolloff))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    dominant_chroma_index = int(np.argmax(chroma_mean))
    dominant_chroma = NOTE_NAMES[dominant_chroma_index]

    return {
        "duration_seconds": duration,
        "loudness_integrated": loudness,
        "loudness_true_peak": true_peak,
        "rms": rms,
        "dominant_freq_hz": dominant_freq,
        "key_note": key_note,
        "tempo_bpm": tempo,
        "zero_crossing_rate_mean": zero_crossing_rate_mean,
        "spectral_centroid_mean": spectral_centroid_mean,
        "spectral_bandwidth_mean": spectral_bandwidth_mean,
        "spectral_rolloff_mean": spectral_rolloff_mean,
        "chroma_dominant": dominant_chroma,
        "peak_amplitude": peak_amplitude,
    }

@app.post("/analyze", response_model=AudioAnalysisResult)
async def analyze(file: UploadFile = File(...)):
    try:
        data = await file.read()
        results = analyze_audio(data)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze audio: {str(e)}")
