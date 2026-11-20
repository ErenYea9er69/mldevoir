"""Application principale FastAPI pour la classification de sentiment multi-source."""
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_IMAGE_MIMES,
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_AUDIO_MIMES,
    MAX_FILE_SIZE_BYTES,
    STATIC_DIR,
    DATASET_DIR,
    MODEL_PATH,
)
from src.schemas import (
    TextPredictionRequest,
    PredictionResponse,
    HealthCheckResponse,
    SourceEnum,
    SentimentEnum,
)
from src.classifier import get_classifier
from src.ocr_pipeline import process_image_sentiment
from src.asr_pipeline import process_audio_sentiment


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisation au démarrage et nettoyage à l'arrêt."""
    try:
        get_classifier()
    except Exception as e:
        print(f"Avertissement lors du chargement initial du modèle : {e}")
    yield


app = FastAPI(
    title="API de Classification de Sentiment Multi-source",
    description=(
        "Système unifié pour analyser le sentiment (positif, négatif, neutre) d'avis clients "
        "e-commerce issus de trois modalités hétérogènes : texte direct, capture d'écran (OCR), "
        "et message audio vocal (ASR), fondé sur une architecture en cascade."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montage des fichiers statiques
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if DATASET_DIR.exists():
    app.mount("/dataset_sentiment_ecommerce", StaticFiles(directory=str(DATASET_DIR)), name="dataset")


@app.get("/", response_class=HTMLResponse, tags=["Interface Web"])
async def root():
    """Page d'accueil servant l'interface Web interactive de test."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h1>API de Sentiment Multi-source</h1><p>Documentation : <a href='/docs'>/docs</a></p>"
    )


@app.get("/health", response_model=HealthCheckResponse, tags=["Santé"])
async def health_check():
    """Vérifie le bon fonctionnement du service et des artefacts prédictifs."""
    model_ok = MODEL_PATH.exists()
    return HealthCheckResponse(
        status="healthy" if model_ok else "degraded",
        model_loaded=model_ok,
        version="1.0.0",
    )


@app.post(
    "/predict/text",
    response_model=PredictionResponse,
    tags=["Prédiction"],
    summary="Prédire le sentiment d'un avis textuel",
)
async def predict_text(request: TextPredictionRequest):
    """
    Reçoit un texte JSON et prédit son sentiment.
    """
    classifier = get_classifier()
    sentiment, confiance, details = classifier.predict(request.texte)

    return PredictionResponse(
        source=SourceEnum.TEXTE,
        texte_extrait=request.texte,
        sentiment_predit=SentimentEnum(sentiment),
        confiance=confiance,
        details=details,
    )


@app.post(
    "/predict/image",
    response_model=PredictionResponse,
    tags=["Prédiction"],
    summary="Prédire le sentiment d'un avis extrait d'une image (OCR)",
)
async def predict_image(file: UploadFile = File(...)):
    """
    Reçoit une image (capture d'écran, ticket de réclamation), extrait le texte brut via OCR
    puis prédit le sentiment avec le modèle textuel en cascade.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Nom de fichier d'image manquant."
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Extension '{ext}' non supportée pour une image. Extensions autorisées : {sorted(list(ALLOWED_IMAGE_EXTENSIONS))}"
        )

    if file.content_type and file.content_type not in ALLOWED_IMAGE_MIMES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Type MIME '{file.content_type}' non supporté pour une image."
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le fichier image envoyé est vide (0 octet)."
        )

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Fichier trop volumineux. Taille maximale : {MAX_FILE_SIZE_BYTES // (1024*1024)} Mo."
        )

    try:
        texte_extrait, sentiment, confiance, details = process_image_sentiment(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors du traitement de l'image : {str(e)}"
        )

    return PredictionResponse(
        source=SourceEnum.IMAGE,
        texte_extrait=texte_extrait if texte_extrait else "[Aucun texte détecté dans l'image]",
        sentiment_predit=SentimentEnum(sentiment),
        confiance=confiance,
        details=details,
    )


@app.post(
    "/predict/audio",
    response_model=PredictionResponse,
    tags=["Prédiction"],
    summary="Prédire le sentiment d'un avis audio vocal (ASR)",
)
async def predict_audio(file: UploadFile = File(...)):
    """
    Reçoit un fichier audio vocal (.wav, .mp3, etc.), réalise la transcription textuelle ASR,
    puis transmet le texte extrait au classifieur.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Nom de fichier audio manquant."
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Extension '{ext}' non supportée pour l'audio. Extensions autorisées : {sorted(list(ALLOWED_AUDIO_EXTENSIONS))}"
        )

    if file.content_type and file.content_type not in ALLOWED_AUDIO_MIMES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Type MIME '{file.content_type}' non supporté pour l'audio."
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le fichier audio envoyé est vide (0 octet)."
        )

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Fichier trop volumineux. Taille maximale : {MAX_FILE_SIZE_BYTES // (1024*1024)} Mo."
        )

    try:
        texte_extrait, sentiment, confiance, details = process_audio_sentiment(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de la transcription du fichier audio : {str(e)}"
        )

    return PredictionResponse(
        source=SourceEnum.AUDIO,
        texte_extrait=texte_extrait if texte_extrait else "[Aucune voix intelligible détectée]",
        sentiment_predit=SentimentEnum(sentiment),
        confiance=confiance,
        details=details,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prédiction"],
    summary="Détection automatique de la modalité et prédiction unifiée",
)
async def predict_auto(
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    """
    Endpoint intelligent universel avec détection automatique du type d'entrée :
    - Si la requête est en 'application/json', traite comme du texte.
    - Si un fichier multipart est envoyé, détecte s'il s'agit d'une image ou d'un audio.
    """
    content_type = request.headers.get("content-type", "").lower()

    # 1. Requête JSON pure (Texte)
    if "application/json" in content_type:
        try:
            body = await request.json()
            texte = body.get("texte") or body.get("text")
            if not texte or not str(texte).strip():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Champ 'texte' requis et non vide dans le corps JSON."
                )
            req = TextPredictionRequest(texte=texte)
            return await predict_text(req)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"JSON invalide : {str(e)}"
            )

    # 2. Requête Multipart / Form-Data avec fichier
    if file is not None and file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext in ALLOWED_IMAGE_EXTENSIONS:
            return await predict_image(file)
        elif ext in ALLOWED_AUDIO_EXTENSIONS:
            return await predict_audio(file)
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Type de fichier non reconnu pour '{file.filename}'. Extensions supportées : images {sorted(list(ALLOWED_IMAGE_EXTENSIONS))}, audios {sorted(list(ALLOWED_AUDIO_EXTENSIONS))}."
            )

    # 3. Requête multipart avec champ formulaire texte (sans fichier)
    try:
        form = await request.form()
        if "texte" in form and str(form["texte"]).strip():
            req = TextPredictionRequest(texte=str(form["texte"]).strip())
            return await predict_text(req)
    except Exception:
        pass

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Entrée non reconnue. Envoyez du JSON avec le champ 'texte', ou un fichier image/audio multipart."
    )
