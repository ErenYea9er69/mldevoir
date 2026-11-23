"""Configuration centralisée pour le classifieur de sentiment multi-source."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset_sentiment_ecommerce"
STATIC_DIR = BASE_DIR / "static"

MODEL_PATH = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

TEXT_DATASET_PATH = DATASET_DIR / "textes.csv"
IMAGE_DATASET_PATH = DATASET_DIR / "images.csv"
AUDIO_DATASET_PATH = DATASET_DIR / "audios.csv"

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
ALLOWED_IMAGE_MIMES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "application/octet-stream",  # Souvent envoyé par certains clients HTTP
}

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
ALLOWED_AUDIO_MIMES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/flac",
    "audio/x-m4a",
    "audio/mp4",
    "application/octet-stream",
}

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 
DEFAULT_CONFIDENCE_ROUND = 4
