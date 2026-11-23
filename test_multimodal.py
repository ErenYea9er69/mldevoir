"""Test CLI universel pour l'Étape 2 : évalue n'importe quel fichier image ou audio externe."""
import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ocr_pipeline import process_image_sentiment
from src.asr_pipeline import process_audio_sentiment
from src.config import ALLOWED_IMAGE_EXTENSIONS, ALLOWED_AUDIO_EXTENSIONS


def evaluate_file(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"❌ Erreur : Le fichier '{file_path}' est introuvable.")
        sys.exit(1)

    ext = path.suffix.lower()
    print(f"\n=======================================================")
    print(f"📂 Fichier analysé : {path.name} ({path.resolve()})")
    print(f"=======================================================")

    if ext in ALLOWED_IMAGE_EXTENSIONS:
        print("🔍 Type détecté : IMAGE (Exécution du Pipeline OCR)...")
        texte, sentiment, confiance, details = process_image_sentiment(path)
        print(f"\n📝 Texte extrait par OCR :")
        print(f"   \"{texte}\"")
        print(f"\n🎯 Diagnostic du modèle :")
        print(f"   Sentiment  : {sentiment.upper()}")
        print(f"   Confiance  : {confiance * 100:.2f}%")
        print(f"   Détails    : {details}")

    elif ext in ALLOWED_AUDIO_EXTENSIONS:
        print("🎙️ Type détecté : AUDIO (Exécution du Pipeline ASR)...")
        texte, sentiment, confiance, details = process_audio_sentiment(path)
        print(f"\n🗣️ Transcription vocale ASR :")
        print(f"   \"{texte}\"")
        print(f"\n🎯 Diagnostic du modèle :")
        print(f"   Sentiment  : {sentiment.upper()}")
        print(f"   Confiance  : {confiance * 100:.2f}%")
        print(f"   Détails    : {details}")

    else:
        print(f"⚠️ Extension '{ext}' non reconnue.")
        print(f"   Images autorisées : {sorted(list(ALLOWED_IMAGE_EXTENSIONS))}")
        print(f"   Audios autorisés  : {sorted(list(ALLOWED_AUDIO_EXTENSIONS))}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Tester n'importe quelle image ou fichier audio avec le classifieur multi-source."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="dataset_sentiment_ecommerce/images/I01.png",
        help="Chemin vers le fichier image (.png, .jpg...) ou audio (.mp3, .wav...)",
    )
    parser.add_argument("--image", "-i", help="Chemin explicite vers une image")
    parser.add_argument("--audio", "-a", help="Chemin explicite vers un audio")

    args = parser.parse_args()

    if args.image:
        evaluate_file(args.image)
    elif args.audio:
        evaluate_file(args.audio)
    else:
        evaluate_file(args.file)


if __name__ == "__main__":
    main()
