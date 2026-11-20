"""Entraînement et évaluation du classifieur de sentiment textuel (Étape 1 du PDF).

Conformément au sujet (Étape 1) :
- Charger le jeu de données textuel fourni (textes.csv).
- Effectuer un nettoyage minimal du texte (minuscules, suppression des stop-words, etc.).
- Vectoriser les textes (ex: TfidfVectorizer) et entraîner un modèle de classification supervisée
  (scikit-learn : Régression Logistique ou SVM) capable de prédire les classes positif, négatif ou neutre.
- Évaluer le modèle (matrice de confusion, rapport de classification) et sauvegarder les artefacts
  entraînés (ex: model.pkl, vectorizer.pkl).
"""
import argparse
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import (
    MODEL_PATH,
    VECTORIZER_PATH,
    METRICS_PATH,
    TEXT_DATASET_PATH,
    MODELS_DIR,
)
from src.text_preprocessing import clean_text


def train_model(c_param: float = 1.0):
    """Charge strictement textes.csv, vectorise et entraîne le modèle supervisé."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not TEXT_DATASET_PATH.exists():
        raise FileNotFoundError(f"Le fichier de dataset {TEXT_DATASET_PATH} est introuvable.")

    # 1. Charger le jeu de données textuel fourni (textes.csv)
    df = pd.read_csv(TEXT_DATASET_PATH)
    print(f"Dataset chargé ({TEXT_DATASET_PATH.name}) : {len(df)} avis textuels.")
    print(df["sentiment_ref"].value_counts())

    # 2. Nettoyage minimal du texte (minuscules, suppression des stop-words, etc.)
    df["texte_clean"] = df["texte"].apply(clean_text)

    # 3. Vectorisation des textes (TfidfVectorizer)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(df["texte_clean"])
    y = df["sentiment_ref"]

    # Entraînement d'un modèle supervisé (Régression Logistique avec calcul de probabilités)
    classifier = LogisticRegression(
        C=c_param,
        max_iter=500,
        random_state=42,
        class_weight="balanced"
    )

    # Validation croisée stratifiée
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(classifier, X, y, cv=cv, scoring="accuracy")

    # Entraînement final
    classifier.fit(X, y)
    y_pred = classifier.predict(X)

    # 4. Évaluation du modèle
    acc = accuracy_score(y, y_pred)
    classes = sorted(list(y.unique()))
    report_dict = classification_report(y, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y, y_pred, labels=classes).tolist()

    print("\n--- Rapport de Classification ---")
    print(classification_report(y, y_pred))

    print(f"--- Matrice de Confusion (Classes : {classes}) ---")
    print(confusion_matrix(y, y_pred, labels=classes))

    # Sauvegarde des artefacts entraînés (model.pkl, vectorizer.pkl)
    joblib.dump(classifier, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nArtefacts sauvegardés :")
    print(f" - Modèle : {MODEL_PATH}")
    print(f" - Vectoriseur : {VECTORIZER_PATH}")

    metrics = {
        "dataset": "textes.csv",
        "dataset_size": len(df),
        "classes": classes,
        "training_accuracy": round(float(acc), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "confusion_matrix": conf_mat,
        "classification_report": report_dict,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return classifier, vectorizer, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement sur textes.csv")
    parser.add_argument("--c", type=float, default=1.0, help="Paramètre C de régularisation")
    args = parser.parse_args()

    train_model(c_param=args.c)
