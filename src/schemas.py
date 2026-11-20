"""Schémas Pydantic pour la validation des requêtes et réponses."""
from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class SentimentEnum(str, Enum):
    POSITIF = "positif"
    NEGATIF = "negatif"
    NEUTRE = "neutre"


class SourceEnum(str, Enum):
    TEXTE = "texte"
    IMAGE = "image"
    AUDIO = "audio"


class TextPredictionRequest(BaseModel):
    """Corps de la requête pour la prédiction textuelle."""
    texte: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Le texte de l'avis client à classifier",
        examples=["Produit conforme à la description, livraison très rapide !"]
    )

    @field_validator("texte")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Le texte fourni ne peut pas être vide ou composé uniquement d'espaces.")
        return v.strip()


class PredictionResponse(BaseModel):
    """Format JSON unifié de la réponse prédictive."""
    source: SourceEnum = Field(..., description="Modalité d'entrée (texte, image, audio)")
    texte_extrait: str = Field(..., description="Texte extrait ou fourni")
    sentiment_predit: SentimentEnum = Field(..., description="Sentiment prédit (positif, negatif, neutre)")
    confiance: float = Field(..., ge=0.0, le=1.0, description="Score de confiance du modèle entre 0.0 et 1.0")
    details: Optional[Dict[str, float]] = Field(
        default=None,
        description="Distribution des probabilités pour chaque classe"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source": "texte",
                "texte_extrait": "Produit conforme à la description, livraison très rapide !",
                "sentiment_predit": "positif",
                "confiance": 0.8924,
                "details": {
                    "positif": 0.8924,
                    "negatif": 0.0412,
                    "neutre": 0.0664
                }
            }
        }
    )


class HealthCheckResponse(BaseModel):
    """Statut de santé du service API."""
    status: str
    model_loaded: bool
    version: str
