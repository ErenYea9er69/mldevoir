"""Module d'inférence pour la classification de sentiment textuel."""
import joblib
from pathlib import Path
from typing import Dict, Tuple

from src.config import MODEL_PATH, VECTORIZER_PATH, DEFAULT_CONFIDENCE_ROUND
from src.text_preprocessing import clean_text


class SentimentClassifier:
    """Classe encapsulant le modèle et le vectoriseur pour l'inférence."""

    def __init__(self, model_path: Path = MODEL_PATH, vectorizer_path: Path = VECTORIZER_PATH):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self._load()

    def _load(self):
        """Charge les artefacts sérialisés depuis le disque."""
        if not self.model_path.exists() or not self.vectorizer_path.exists():
            raise FileNotFoundError(
                f"Modèle ({self.model_path}) ou vectoriseur ({self.vectorizer_path}) manquant. "
                "Veuillez exécuter train.py au préalable."
            )
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

    def predict(self, raw_text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Prédit le sentiment d'un texte brut.
        Retourne :
        - sentiment_predit (ex: 'positif', 'negatif', 'neutre')
        - confiance (probabilité de la classe prédite, float)
        - details (dictionnaire des probabilités de chaque classe)
        """
        if not raw_text or not raw_text.strip():
            return "neutre", 0.0, {"positif": 0.0, "negatif": 0.0, "neutre": 1.0}

        cleaned = clean_text(raw_text)
        if not cleaned:
            return "neutre", 0.0, {"positif": 0.0, "negatif": 0.0, "neutre": 1.0}

        X = self.vectorizer.transform([cleaned])
        pred_label = self.model.predict(X)[0]

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            classes = self.model.classes_
            class_probs = {
                cls_name: round(float(prob), DEFAULT_CONFIDENCE_ROUND)
                for cls_name, prob in zip(classes, probs)
            }
            confidence = round(float(probs.max()), DEFAULT_CONFIDENCE_ROUND)
        else:
            confidence = 1.0
            class_probs = {pred_label: 1.0}

        return str(pred_label), confidence, class_probs


# Singleton pour réutilisation rapide dans l'API
_classifier_instance = None


def get_classifier() -> SentimentClassifier:
    """Fournit l'instance partagée du classifieur."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = SentimentClassifier()
    return _classifier_instance
