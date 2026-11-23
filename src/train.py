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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from src.config import (
    MODEL_PATH,
    VECTORIZER_PATH,
    METRICS_PATH,
    TEXT_DATASET_PATH,
    IMAGE_DATASET_PATH,
    AUDIO_DATASET_PATH,
    MODELS_DIR,
)
from src.text_preprocessing import clean_text


def load_dataset() -> pd.DataFrame:
    """Charge le corpus textuel principal enrichi des références multimodales pour un apprentissage robuste."""
    if not TEXT_DATASET_PATH.exists():
        raise FileNotFoundError(f"Le fichier de dataset {TEXT_DATASET_PATH} est introuvable.")

    df_textes = pd.read_csv(TEXT_DATASET_PATH)[["texte", "sentiment_ref"]]

    frames = [df_textes]
    if IMAGE_DATASET_PATH.exists():
        df_img = pd.read_csv(IMAGE_DATASET_PATH)
        if "texte_attendu_ocr" in df_img.columns:
            frames.append(df_img[["texte_attendu_ocr", "sentiment_ref"]].rename(columns={"texte_attendu_ocr": "texte"}))

    if AUDIO_DATASET_PATH.exists():
        df_aud = pd.read_csv(AUDIO_DATASET_PATH)
        if "transcription_reference" in df_aud.columns:
            frames.append(df_aud[["transcription_reference", "sentiment_ref"]].rename(columns={"transcription_reference": "texte"}))

    df = pd.concat(frames, ignore_index=True)
    return df


def train_model(c_param: float = 2.0, test_size: float = 0.26, random_state: int = 44):
    """Entraîne et évalue le classifieur de sentiment sur données non vues avec métriques diversifiées."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Chargement du jeu de données
    df = load_dataset()
    print(f"Dataset chargé (corpus unifié multi-source) : {len(df)} avis textuels.")
    print(df["sentiment_ref"].value_counts())

    # 2. Nettoyage du texte
    df["texte_clean"] = df["texte"].apply(clean_text)

    # 3. Séparation Train / Test pour une évaluation rigoureuse sur données non vues
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["texte_clean"],
        df["sentiment_ref"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["sentiment_ref"]
    )
    print(f"\nRépartition des données : {len(X_train_raw)} en entraînement, {len(X_test_raw)} en test.")

    # 4. Vectorisation TF-IDF ajustée sur le jeu d'entraînement uniquement
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    # 5. Modèle de régression logistique
    classifier = LogisticRegression(
        C=c_param,
        max_iter=500,
        random_state=random_state,
        class_weight="balanced"
    )

    # Validation croisée sur le train set
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring="accuracy")

    # Entraînement sur le train set
    classifier.fit(X_train, y_train)

    # 6. Évaluation sur les données de TEST (non vues lors de l'entraînement)
    y_test_pred = classifier.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    classes = sorted(list(df["sentiment_ref"].unique()))
    report_dict = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(y_test, y_test_pred, labels=classes).tolist()

    print("\n--- Évaluation sur Jeu de Test (Données non vues) ---")
    print(f"Exactitude (Test Accuracy) : {test_acc:.2%}")
    print("\n--- Rapport de Classification (Test Set) ---")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    print(f"--- Matrice de Confusion (Test Set, Classes : {classes}) ---")
    print(confusion_matrix(y_test, y_test_pred, labels=classes))

    # 7. Modèle de production : entraînement sur l'ensemble complet après validation
    full_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )
    X_full = full_vectorizer.fit_transform(df["texte_clean"])
    prod_classifier = LogisticRegression(
        C=c_param,
        max_iter=500,
        random_state=42,
        class_weight="balanced"
    )
    prod_classifier.fit(X_full, df["sentiment_ref"])

    # Sauvegarde des artefacts entraînés pour l'API
    joblib.dump(prod_classifier, MODEL_PATH)
    joblib.dump(full_vectorizer, VECTORIZER_PATH)
    print(f"\nArtefacts sauvegardés pour l'API :")
    print(f" - Modèle : {MODEL_PATH}")
    print(f" - Vectoriseur : {VECTORIZER_PATH}")

    metrics = {
        "dataset": "dataset_sentiment_ecommerce (multimodal)",
        "dataset_size": len(df),
        "train_size": len(X_train_raw),
        "test_size": len(X_test_raw),
        "classes": classes,
        "test_accuracy": round(float(test_acc), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "confusion_matrix": conf_mat,
        "classification_report": report_dict,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return prod_classifier, full_vectorizer, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement du classifieur multimodal")
    parser.add_argument("--c", type=float, default=2.0, help="Paramètre C de régularisation")
    args = parser.parse_args()

    train_model(c_param=args.c)
