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
    if freq <= 0:
        return "N/A"
    # Преобразуем частоту в MIDI-ноту
    midi = int(round(69 + 12 * np.log2(freq / 440.0)))
    note = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{note}{octave}"

def analyze_audio(data: bytes):
    y, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)

    duration = librosa.get_duration(y=y, sr=sr)

    meter = pyln.Meter(sr)  # для LUFS
    loudness = meter.integrated_loudness(y)
    true_peak = meter.true_peak_level(y)

    rms = float(np.sqrt(np.mean(y ** 2)))
    peak_amplitude = float(np.max(np.abs(y)))

    # Доминирующая частота — через спектрограмму
    D = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    avg_spectrum = np.mean(D, axis=1)
    dominant_freq = freqs[np.argmax(avg_spectrum)]

    # Темп (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Нота ключа (тональность) — по доминирующей частоте
    key_note = freq_to_note(dominant_freq)

    # Zero Crossing Rate — занятость
    zcr = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = float(np.mean(zcr))

    # Спектральные характеристики
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    spectral_centroid_mean = float(np.mean(spectral_centroid))
    spectral_bandwidth_mean = float(np.mean(spectral_bandwidth))
    spectral_rolloff_mean = float(np.mean(spectral_rolloff))

    # Хрома — определяем доминирующую хрома-бин (тональный класс)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    dominant_chroma_index = np.argmax(chroma_mean)
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
    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    data = await file.read()
    results = analyze_audio(data)
    return results
