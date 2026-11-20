"""Entraînement et évaluation du classifieur de sentiment textuel."""
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
    IMAGE_DATASET_PATH,
    AUDIO_DATASET_PATH,
    MODELS_DIR,
)
from src.text_preprocessing import clean_text


def load_datasets(include_multimodal: bool = True):
    """Charge le jeu de données textuel principal et optionnellement les références multimodales."""
    if not TEXT_DATASET_PATH.exists():
        raise FileNotFoundError(f"Le fichier de dataset {TEXT_DATASET_PATH} est introuvable.")

    df_text = pd.read_csv(TEXT_DATASET_PATH)
    frames = [df_text[["texte", "sentiment_ref"]]]

    if include_multimodal:
        if IMAGE_DATASET_PATH.exists():
            df_img = pd.read_csv(IMAGE_DATASET_PATH)
            if "texte_attendu_ocr" in df_img.columns:
                df_img_clean = df_img.rename(columns={"texte_attendu_ocr": "texte"})[["texte", "sentiment_ref"]]
                frames.append(df_img_clean)

        if AUDIO_DATASET_PATH.exists():
            df_aud = pd.read_csv(AUDIO_DATASET_PATH)
            if "transcription_reference" in df_aud.columns:
                df_aud_clean = df_aud.rename(columns={"transcription_reference": "texte"})[["texte", "sentiment_ref"]]
                frames.append(df_aud_clean)

    df_full = pd.concat(frames, ignore_index=True)
    return df_full, df_text


def train_model(include_multimodal: bool = True, c_param: float = 2.0):
    """Entraîne le vectoriseur TF-IDF et la régression logistique, puis évalue et sauvegarde."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_train, df_text_only = load_datasets(include_multimodal=include_multimodal)
    print(f"Jeu d'entraînement chargé : {len(df_train)} exemples (multimodal={include_multimodal})")
    print(df_train["sentiment_ref"].value_counts())

    # Prétraitement
    df_train["texte_clean"] = df_train["texte"].apply(clean_text)

    # Vectorisation TF-IDF (unigrammes et bigrammes)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(df_train["texte_clean"])
    y = df_train["sentiment_ref"]

    # Modèle de classification : Régression logistique avec probabilités étalonnées
    classifier = LogisticRegression(
        C=c_param,
        max_iter=500,
        random_state=42,
        class_weight="balanced"
    )

    # Évaluation par validation croisée stratifiée
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(classifier, X, y, cv=cv, scoring="accuracy")
    print(f"\n--- Validation Croisée (5 folds) ---")
    print(f"Précision moyenne : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Entraînement final
    classifier.fit(X, y)
    y_pred = classifier.predict(X)

    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y, y_pred, labels=sorted(list(y.unique()))).tolist()

    print(f"\n--- Rapport de Classification ---")
    print(classification_report(y, y_pred))

    print(f"\n--- Matrice de Confusion (Classes : {sorted(list(y.unique()))}) ---")
    print(confusion_matrix(y, y_pred, labels=sorted(list(y.unique()))))

    # Sauvegarde des artefacts
    joblib.dump(classifier, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nModèle sauvegardé dans : {MODEL_PATH}")
    print(f"Vectoriseur sauvegardé dans : {VECTORIZER_PATH}")

    metrics = {
        "dataset_size": len(df_train),
        "include_multimodal": include_multimodal,
        "classes": sorted(list(y.unique())),
        "training_accuracy": round(float(acc), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "confusion_matrix": conf_mat,
        "classification_report": report,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Métriques enregistrées dans : {METRICS_PATH}")

    return classifier, vectorizer, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement du classifieur de sentiment")
    parser.add_argument("--text-only", action="store_true", help="Entraîner uniquement sur textes.csv")
    parser.add_argument("--c", type=float, default=2.0, help="Paramètre de régularisation C")
    args = parser.parse_args()

    train_model(include_multimodal=not args.text_only, c_param=args.c)
