"""Pipeline de prétraitement d'images et extraction OCR Tesseract"""
import io
import os
import shutil
import logging
from pathlib import Path
from typing import Union, Tuple, Dict
from PIL import Image

logger = logging.getLogger(__name__)

# Détection et configuration de pytesseract
try:
    import pytesseract
    # Recherche automatique de l'exécutable tesseract standard
    _possible_tesseract_paths = [
        os.getenv("TESSERACT_CMD"),
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\hp\AppData\Local\Tesseract-OCR\tesseract.exe",
    ]
    for p in _possible_tesseract_paths:
        if p and os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

# Moteur de repli natif Windows (WinOCR)
try:
    import winocr
    WINOCR_AVAILABLE = True
except ImportError:
    WINOCR_AVAILABLE = False


def _load_pil_image(image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    """Charge une image PIL à partir de différents formats d'entrée."""
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input))
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    else:
        raise ValueError(f"Format d'image non supporté : {type(image_input)}")

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def extract_text_from_image(image_input: Union[str, Path, bytes, Image.Image]) -> str:
    """
    Extrait le texte brut d'une image :
    1. Tente pytesseract (conforme à l'énoncé du projet).
    2. Si l'exécutable tesseract n'est pas présent sur la machine hôte,
       utilise le moteur natif Windows OCR (WinOCR) comme repli transparent et précis.
    """
    img = _load_pil_image(image_input)
    text = ""

    if PYTESSERACT_AVAILABLE:
        try:
            # Tente avec la langue française puis par défaut
            try:
                text = pytesseract.image_to_string(img, lang="fra")
            except Exception:
                text = pytesseract.image_to_string(img)
            text = text.strip()
            if text:
                return text
        except Exception as e:
            logger.info("pytesseract non disponible ou binaire tesseract introuvable (%s). Utilisation du repli OCR.", e)

    # Repli Windows Media OCR
    if WINOCR_AVAILABLE and os.name == "nt":
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(winocr.recognize_pil_sync, img)
                result = future.result()
            text = result.get("text", "").strip()
            if text:
                return text
        except Exception as e:
            logger.error("Erreur WinOCR : %s", e)

    return text.strip()


def process_image_sentiment(
    image_input: Union[str, Path, bytes, Image.Image],
    classifier=None
) -> Tuple[str, str, float, Dict[str, float]]:
    """
    Pipeline multimodal complet :
    Image -> Extraction OCR du texte brut -> Inférence du Sentiment via le modèle texte unique.
    Retourne : (texte_extrait, sentiment_predit, confiance, probabilites_classes)
    """
    from src.classifier import get_classifier
    clf = classifier or get_classifier()

    texte_extrait = extract_text_from_image(image_input)
    sentiment, confiance, details = clf.predict(texte_extrait)

    return texte_extrait, sentiment, confiance, details


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "dataset_sentiment_ecommerce/images/I01.png"
    print(f"--- Test OCR sur : {target} ---")
    txt, sent, conf, details = process_image_sentiment(target)
    print(f"Texte Extrait : {txt}")
    print(f"Sentiment     : {sent.upper()} (Confiance: {conf*100:.1f}%)")
    print(f"Détails       : {details}")
