# Classifieur de Sentiment Multi-source (E-commerce Customer Voice)
**Mini-Projet — Machine Learning Appliqué (2026)**

Système unifié et intelligent d'analyse de sentiment client (positif, négatif, neutre) acceptant trois modalités d'entrée hétérogènes : **texte direct**, **capture d'écran d'avis (OCR)**, et **message vocal dicté (ASR)**.

---

## 1. Contexte & Architecture en Cascade

Une plateforme e-commerce reçoit quotidiennement des avis sous des formats hétérogènes. Afin d'éviter la complexité et le coût de maintenance de trois modèles distincts, le système adopte une **architecture en cascade** :

```
                  ┌────────────────────────────────────────┐
   Texte ────────►│ POST /predict/text                     ├──┐
                  │ (Modèle de sentiment textuel)          │  │
                  └────────────────────────────────────────┘  │
                  ┌────────────────────────────────────────┐  │
   Image ────────►│ POST /predict/image                    ├──┼──► Réponse JSON unifiée :
                  │ OCR (pytesseract / winocr) ─► Modèle   │  │   {
                  └────────────────────────────────────────┘  │     "source": "...",
                  ┌────────────────────────────────────────┐  │     "texte_extrait": "...",
   Audio ────────►│ POST /predict/audio                    ├──┤     "sentiment_predit": "...",
                  │ Transcription (Whisper/ASR) ─► Modèle  │  │     "confiance": 0.89,
                  └────────────────────────────────────────┘  │     "details": {...}
                  ┌────────────────────────────────────────┐  │   }
   Auto ─────────►│ POST /predict (Détection automatique)  ├──┘
                  └────────────────────────────────────────┘
```

Toutes les modalités non textuelles (images et audios) sont converties en texte brut via leurs pipelines respectifs, puis transmises au **même classifieur textuel supervisé**.

---

## 2. Structure du Projet

```
mlproject/
├── dataset_sentiment_ecommerce/  # Jeu de données de référence
│   ├── textes.csv                # 24 avis textuels équilibrés (8 pos, 8 neg, 8 neutre)
│   ├── images.csv                # Références OCR et sentiments des 9 images
│   ├── audios.csv                # Transcriptions de référence et sentiments des 9 audios
│   ├── images/                   # Captures d'écran (I01.png à I09.png)
│   └── audios/                   # Enregistrements vocaux (A01.mp3 à A09.mp3)
├── models/                       # Modèles sérialisés et métriques
│   ├── model.pkl                 # Modèle de classification entraîné (Logistic Regression)
│   ├── vectorizer.pkl            # Vectoriseur TF-IDF (1-grammes et 2-grammes)
│   └── metrics.json              # Métriques d'évaluation et matrice de confusion
├── src/                          # Modules applicatifs modulaires
│   ├── __init__.py
│   ├── config.py                 # Configuration centrale (chemins, types MIME, tailles max)
│   ├── text_preprocessing.py     # Nettoyage, normalisation et filtrage des stop-words
│   ├── train.py                  # Script d'entraînement, validation croisée et persistance
│   ├── classifier.py             # Inférence du classifieur textuel et calcul de confiance
│   ├── ocr_pipeline.py           # Pipeline Image -> OCR (pytesseract + fallback winocr)
│   ├── asr_pipeline.py           # Pipeline Audio -> ASR (OpenAI Whisper + SpeechRecognition)
│   └── schemas.py                # Schémas Pydantic de validation et réponse
├── static/                       # Mini-application Web de test intégrée
│   └── index.html                # Interface moderne (Tailwind-like, responsive, jauges)
├── tests/                        # Tests automatisés avec pytest et httpx
│   ├── __init__.py
│   └── test_api.py               # Tests unitaires et d'intégration des endpoints et rejets
├── main.py                       # Application FastAPI (routes REST, validation, CORS, static)
├── train.py                      # Point d'entrée racine pour ré-entraîner les modèles
├── streamlit_app.py              # Mini-application Web de test dédiée Streamlit (Option B)
├── requirements.txt              # Liste des dépendances gelées
└── README.md                     # Documentation technique complète
```

---

## 3. Installation et Prérequis

### Prérequis
- **Python 3.10 à 3.14**
- Système d'exploitation : Windows, Linux ou macOS.

### Installation des Dépendances
Dans un terminal ouvert dans le dossier du projet :
```bash
pip install -r requirements.txt
```

> **Note OCR & ASR :** 
> - Pour l'OCR, le module utilise en priorité `pytesseract`. Sur Windows, en l'absence de l'exécutable système Tesseract, le module bascule automatiquement sur `winocr` (moteur natif Windows Media OCR) sans aucune configuration supplémentaire requise.
> - Pour l'ASR, le décodage audio est assuré en pur Python par `soundfile` et `scipy`, rendant l'exécution de Whisper locale et autonome sans dépendre d'un binaire `ffmpeg` externe.

---

## 4. Entraînement et Persistance des Modèles

Pour entraîner ou ré-entraîner le modèle à partir du dataset :
```bash
python train.py
```
Cette commande :
1. Charge strictement les avis du jeu de données textuel fourni (`textes.csv`).
2. Applique le prétraitement textuel (normalisation NFKD, mise en minuscules, suppression des stop-words non polarisés tout en préservant scrupuleusement les négations).
3. Vectorise le corpus avec `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)`.
4. Évalue par validation croisée stratifiée 5-folds (`StratifiedKFold`).
5. Sérialise `models/model.pkl`, `models/vectorizer.pkl` et enregistre les scores dans `models/metrics.json`.

---

## 5. Démarrage de l'API REST FastAPI

Lancez le serveur d'API avec rechargement automatique :
```bash
uvicorn main:app --reload
```
Le serveur démarre sur **`http://127.0.0.1:8000`**.

### Documentation Interactive
- **Swagger UI** : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc** : [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 6. Mini-Applications Web de Test

Deux interfaces de test sont fournies conformément à l'**Étape 4 (Option B)** :

### Option 1 : Interface Web Intégrée (HTML/JS Moderne)
L'application web est directement servie par FastAPI. Il suffit d'ouvrir dans un navigateur :
👉 **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

Fonctionnalités :
- Onglets dédiés : ✍️ Texte, 🖼️ Image (OCR), 🎙️ Audio (ASR), ⚡ Auto-Détection.
- Boutons d'exemples pré-remplis en 1 clic.
- Affichage visuel du sentiment (vert positif, rouge négatif, gris neutre).
- Jauge animée du score de confiance et répartition des probabilités par classe.
- Inspecteur du JSON brut renvoyé par l'API.

### Option 2 : Application Dédiée Streamlit
Pour lancer l'application Streamlit autonome :
```bash
streamlit run streamlit_app.py
```
Elle communique en arrière-plan avec l'API FastAPI et propose un tableau de bord complet avec métriques, matrice de confusion et graphiques.

---

## 7. Exécution des Tests Automatisés

Pour lancer l'ensemble de la suite de tests automatisés :
```bash
python -m pytest tests/test_api.py -v
```

Les tests couvrent :
- `test_health_check` : Vérification du statut de l'API et de la disponibilité du modèle.
- `test_root_html` : Vérification de la disponibilité de l'interface Web statique.
- `test_predict_text_success` : Validation du format de réponse JSON unifiée sur des avis positifs, négatifs et neutres.
- `test_predict_text_validation_errors` : Validation des rejets `422 Unprocessable Entity` (corps vide, chaîne blanche, types erronés).
- `test_predict_image_success` : Validation du pipeline OCR sur image PNG réelle.
- `test_predict_image_validation_errors` : Rejet des extensions interdites (`422`) et des fichiers vides (`400`).
- `test_predict_audio_success` : Validation du pipeline ASR sur enregistrement audio réel.
- `test_predict_audio_validation_errors` : Rejet des fichiers non supportés (`422`) et vides (`400`).
- `test_predict_auto_endpoint` : Validation de l'orientation dynamique sur `POST /predict`.
- `test_text_preprocessing_negation_preservation` : Validation de la préservation des négations ("ne ... pas", "rien", "jamais").

---

## 8. Spécification des Endpoints de l'API

### Schéma de Réponse JSON Unifié
Tous les endpoints de prédiction renvoient rigoureusement le même schéma :
```json
{
  "source": "texte",
  "texte_extrait": "Produit conforme à la description, livraison super rapide en 24h, je recommande vivement !",
  "sentiment_predit": "positif",
  "confiance": 0.8924,
  "details": {
    "positif": 0.8924,
    "neutre": 0.0664,
    "negatif": 0.0412
  }
}
```

### Exemples d'Appel cURL

#### 1. Prédiction Textuelle
```bash
curl -X POST "http://127.0.0.1:8000/predict/text" \
     -H "Content-Type: application/json" \
     -d '{"texte": "Produit conforme à la description, livraison super rapide en 24h, je recommande vivement !"}'
```

#### 2. Prédiction Image (OCR)
```bash
curl -X POST "http://127.0.0.1:8000/predict/image" \
     -F "file=@dataset_sentiment_ecommerce/images/I01.png;type=image/png"
```

#### 3. Prédiction Audio (ASR)
```bash
curl -X POST "http://127.0.0.1:8000/predict/audio" \
     -F "file=@dataset_sentiment_ecommerce/audios/A01.mp3;type=audio/mpeg"
```

#### 4. Prédiction Universelle (Auto-Détection)
```bash
# Avec du texte :
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"texte": "Article reçu cassé et le service client ne répond pas."}'

# Avec un fichier quelconque :
curl -X POST "http://127.0.0.1:8000/predict" \
     -F "file=@dataset_sentiment_ecommerce/images/I02.png"
```

---

## 9. Justification des Choix Techniques

1. **Architecture en Cascade :**
   Permet d'isoler la complexité multimodale aux étapes d'extraction (OCR et ASR) et de mutualiser toute la logique métier, la maintenance et les tests sur un unique classifieur textuel.
2. **Modélisation & Prétraitement Textuel :**
   - L'utilisation de TF-IDF avec unigrammes et bigrammes (`(1, 2)`) permet de capturer des expressions discriminantes telles que *"ne répond pas"*, *"service client"*, *"super rapide"*, *"sans hésiter"*.
   - Le filtrage des stop-words a été configuré sur mesure pour exclure les mots fonctionnels neutres (articles, prépositions) tout en **préservant scrupuleusement les mots porteurs de polarité et de négation** (*pas*, *plus*, *sans*, *jamais*, *aucun*, *rien*, *trop*, *très*).
   - La régression logistique étalonnée garantit des scores de confiance probabilistes fiables (`predict_proba`) et une inférence quasi-instantanée (< 5 ms).
3. **Pipeline OCR (Image) :**
   - Utilisation de `pytesseract` et `Pillow` comme préconisé dans le sujet.
   - Intégration d'un mécanisme de repli transparent sur `winocr` pour assurer une portabilité totale sans dépendance contraignante d'installation système.
4. **Pipeline ASR (Audio) :**
   - Utilisation du modèle `OpenAI Whisper` (`tiny`) combiné à `SpeechRecognition` pour une transcription robuste en français.
   - Chargement audio via `soundfile` et rééchantillonnage `scipy.signal` garantissant une exécution autonome sans besoin du binaire `ffmpeg`.
5. **FastAPI & Validation Pydantic :**
   - Contrôle strict des types MIME, des extensions autorisées et de la taille maximale des fichiers (10 Mo).
   - Documentation interactive OpenAPI/Swagger générée automatiquement.
