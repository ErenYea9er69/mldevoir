"""Suite de tests automatisés unitaires et d'intégration (Étape 5)."""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from main import app
from src.text_preprocessing import clean_text
from src.classifier import get_classifier

client = TestClient(app)

DATASET_DIR = Path("dataset_sentiment_ecommerce")
IMAGE_SAMPLE = DATASET_DIR / "images" / "I01.png"
AUDIO_SAMPLE = DATASET_DIR / "audios" / "A01.mp3"


def test_health_check():
    """Vérifie l'endpoint /health."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["model_loaded"] is True


def test_root_html():
    """Vérifie que la racine sert l'interface HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_predict_text_success():
    """Vérifie la prédiction textuelle avec les 3 sentiments."""
    cases = [
        ("Produit conforme à la description, livraison super rapide en 24h, je recommande vivement !", "positif"),
        ("Article reçu cassé et le service client ne répond pas depuis une semaine. Très déçu.", "negatif"),
        ("Le colis est arrivé dans les délais annoncés. L'emballage est standard.", "neutre"),
    ]

    for text, expected_sentiment in cases:
        response = client.post("/predict/text", json={"texte": text})
        assert response.status_code == 200, f"Erreur pour le texte : {text}"
        data = response.json()
        assert data["source"] == "texte"
        assert data["texte_extrait"] == text
        assert data["sentiment_predit"] in ("positif", "negatif", "neutre")
        assert 0.0 <= data["confiance"] <= 1.0
        assert "details" in data
        assert expected_sentiment in data["details"]


def test_predict_text_validation_errors():
    """Vérifie la gestion rigoureuse des erreurs Pydantic (422 Unprocessable Entity)."""
    # 1. Corps vide
    res1 = client.post("/predict/text", json={})
    assert res1.status_code == 422

    # 2. Chaîne vide
    res2 = client.post("/predict/text", json={"texte": ""})
    assert res2.status_code == 422

    # 3. Espaces uniquement
    res3 = client.post("/predict/text", json={"texte": "    "})
    assert res3.status_code == 422

    # 4. Mauvais type
    res4 = client.post("/predict/text", json={"texte": 12345})
    assert res4.status_code == 422


def test_predict_image_success():
    """Vérifie l'extraction OCR et la prédiction sur une image valide."""
    assert IMAGE_SAMPLE.exists(), f"Fichier de test manquant : {IMAGE_SAMPLE}"

    with open(IMAGE_SAMPLE, "rb") as f:
        response = client.post(
            "/predict/image",
            files={"file": ("I01.png", f, "image/png")}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "image"
    assert len(data["texte_extrait"]) > 0
    assert data["sentiment_predit"] in ("positif", "negatif", "neutre")
    assert 0.0 <= data["confiance"] <= 1.0


def test_predict_image_validation_errors():
    """Vérifie le rejet des formats images invalides et fichiers vides."""
    # Extension interdite
    response = client.post(
        "/predict/image",
        files={"file": ("malicieux.exe", b"fake binary", "application/x-msdownload")}
    )
    assert response.status_code == 422

    # Fichier vide (0 octet)
    empty_resp = client.post(
        "/predict/image",
        files={"file": ("empty.png", b"", "image/png")}
    )
    assert empty_resp.status_code == 400


def test_predict_audio_success():
    """Vérifie la transcription ASR et la prédiction sur un fichier audio valide."""
    assert AUDIO_SAMPLE.exists(), f"Fichier de test manquant : {AUDIO_SAMPLE}"

    with open(AUDIO_SAMPLE, "rb") as f:
        response = client.post(
            "/predict/audio",
            files={"file": ("A01.mp3", f, "audio/mpeg")}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "audio"
    assert len(data["texte_extrait"]) > 0
    assert data["sentiment_predit"] in ("positif", "negatif", "neutre")
    assert 0.0 <= data["confiance"] <= 1.0


def test_predict_audio_validation_errors():
    """Vérifie le rejet des formats audios non supportés et fichiers vides."""
    # Extension interdite
    response = client.post(
        "/predict/audio",
        files={"file": ("document.pdf", b"%PDF-1.4...", "application/pdf")}
    )
    assert response.status_code == 422

    # Fichier vide
    empty_resp = client.post(
        "/predict/audio",
        files={"file": ("empty.mp3", b"", "audio/mpeg")}
    )
    assert empty_resp.status_code == 400


def test_predict_auto_endpoint():
    """Vérifie l'auto-détection sur POST /predict."""
    # 1. Détection Texte JSON
    res_text = client.post("/predict", json={"texte": "Service après-vente remarquable et très aimable !"})
    assert res_text.status_code == 200
    assert res_text.json()["source"] == "texte"

    # 2. Détection Image multipart
    with open(IMAGE_SAMPLE, "rb") as f:
        res_img = client.post("/predict", files={"file": ("I01.png", f, "image/png")})
    assert res_img.status_code == 200
    assert res_img.json()["source"] == "image"

    # 3. Détection Audio multipart
    with open(AUDIO_SAMPLE, "rb") as f:
        res_aud = client.post("/predict", files={"file": ("A01.mp3", f, "audio/mpeg")})
    assert res_aud.status_code == 200
    assert res_aud.json()["source"] == "audio"


def test_text_preprocessing_negation_preservation():
    """Vérifie que le prétraitement conserve bien les mots de négation critiques."""
    phrase = "Ce produit n'est pas bon du tout, je ne recommande rien."
    cleaned = clean_text(phrase)
    # Les termes de négation 'pas', 'rien' doivent subsister
    assert "pas" in cleaned
    assert "rien" in cleaned
