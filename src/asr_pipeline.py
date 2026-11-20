"""Pipeline de prétraitement audio et transcription ASR (Étape 2)."""
import io
import logging
from pathlib import Path
from typing import Union, Tuple, Dict, Optional
import numpy as np
import scipy.signal
import soundfile as sf

logger = logging.getLogger(__name__)

# Modules ASR optionnels
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_REC_AVAILABLE = True
except ImportError:
    SPEECH_REC_AVAILABLE = False

_whisper_model = None


def get_whisper_model(model_name: str = "tiny"):
    """Charge ou retourne l'instance en cache du modèle Whisper."""
    global _whisper_model
    if not WHISPER_AVAILABLE:
        return None
    if _whisper_model is None:
        try:
            logger.info("Chargement du modèle Whisper (%s)...", model_name)
            _whisper_model = whisper.load_model(model_name)
        except Exception as e:
            logger.warning("Impossible de charger le modèle Whisper : %s", e)
            _whisper_model = None
    return _whisper_model


def _load_audio_data(audio_input: Union[str, Path, bytes, io.BytesIO]) -> Tuple[np.ndarray, int]:
    """Charge les données audio brutes en tableau numpy et retourne (data, sample_rate)."""
    if isinstance(audio_input, (str, Path)):
        data, samplerate = sf.read(str(audio_input))
    elif isinstance(audio_input, (bytes, io.BytesIO)):
        buf = io.BytesIO(audio_input) if isinstance(audio_input, bytes) else audio_input
        buf.seek(0)
        data, samplerate = sf.read(buf)
    else:
        raise ValueError(f"Type d'entrée audio non supporté : {type(audio_input)}")

    # Conversion en mono si stéréo / multi-canal
    if data.ndim > 1:
        data = data.mean(axis=1)

    return data, samplerate


def transcribe_with_whisper(data: np.ndarray, samplerate: int, model_name: str = "tiny") -> str:
    """Transcription locale via Whisper (sans dépendance à l'exécutable système ffmpeg)."""
    model = get_whisper_model(model_name)
    if model is None:
        raise RuntimeError("Modèle Whisper indisponible.")

    # Rééchantillonnage à 16 000 Hz si nécessaire
    target_sr = 16000
    if samplerate != target_sr:
        num_samples = int(len(data) * target_sr / samplerate)
        resampled_data = scipy.signal.resample(data, num_samples)
    else:
        resampled_data = data

    audio_tensor = resampled_data.astype(np.float32)
    # Détection et transcription en français
    result = model.transcribe(audio_tensor, language="fr", fp16=False)
    return result.get("text", "").strip()


def transcribe_with_speech_recognition(data: np.ndarray, samplerate: int) -> str:
    """Transcription légère via SpeechRecognition (Google Speech API)."""
    if not SPEECH_REC_AVAILABLE:
        raise RuntimeError("SpeechRecognition n'est pas disponible.")

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, data, samplerate, format="WAV")
    wav_buffer.seek(0)

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_buffer) as source:
        audio_record = recognizer.record(source)

    text = recognizer.recognize_google(audio_record, language="fr-FR")
    return text.strip()


def transcribe_audio(
    audio_input: Union[str, Path, bytes, io.BytesIO],
    engine: str = "auto"
) -> str:
    """
    Transcrit un fichier audio (.wav, .mp3, etc.) en texte français.
    - engine: 'auto', 'whisper', ou 'speech_recognition'
    """
    data, samplerate = _load_audio_data(audio_input)
    text = ""

    if engine in ("auto", "speech_recognition") and SPEECH_REC_AVAILABLE:
        try:
            text = transcribe_with_speech_recognition(data, samplerate)
            if text:
                return text
        except Exception as e:
            logger.info("SpeechRecognition a échoué (%s), tentative avec Whisper...", e)

    if engine in ("auto", "whisper") and WHISPER_AVAILABLE:
        try:
            text = transcribe_with_whisper(data, samplerate)
            if text:
                return text
        except Exception as e:
            logger.error("Erreur Whisper : %s", e)

    return text.strip()


def process_audio_sentiment(
    audio_input: Union[str, Path, bytes, io.BytesIO],
    classifier=None,
    engine: str = "auto"
) -> Tuple[str, str, float, Dict[str, float]]:
    """
    Pipeline multimodal complet :
    Audio (.wav, .mp3) -> Transcription ASR -> Inférence du Sentiment via le modèle texte unique.
    Retourne : (texte_extrait, sentiment_predit, confiance, probabilites_classes)
    """
    from src.classifier import get_classifier
    clf = classifier or get_classifier()

    texte_extrait = transcribe_audio(audio_input, engine=engine)
    sentiment, confiance, details = clf.predict(texte_extrait)

    return texte_extrait, sentiment, confiance, details
