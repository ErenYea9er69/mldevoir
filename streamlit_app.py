"""Application Streamlit de test pour l'analyse de sentiment multi-source."""
import io
import json
from pathlib import Path
import requests
import streamlit as st
import pandas as pd

API_DEFAULT_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Customer Voice — Multi-source Sentiment",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ E-commerce Customer Voice — Sentiment Classifier")
st.markdown(
    "Mini-projet ML Appliqué 2026 : Architecture en cascade pour classifier des avis "
    "clients provenant de **texte brut**, d'**images (OCR)** ou de **messages vocaux (ASR)**."
)

# Sidebar
st.sidebar.header("⚙️ Configuration API")
api_url = st.sidebar.text_input("URL Backend FastAPI", value=API_DEFAULT_URL)

try:
    health_resp = requests.get(f"{api_url}/health", timeout=2)
    if health_resp.status_code == 200:
        st.sidebar.success("🟢 API en ligne et opérationnelle")
    else:
        st.sidebar.warning(f"🟠 API en mode dégradé ({health_resp.status_code})")
except Exception:
    st.sidebar.error("🔴 Impossible de contacter l'API FastAPI")
    st.sidebar.info("Lancez l'API dans un terminal : `uvicorn main:app --reload`")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Liens utiles")
st.sidebar.markdown(f"- [Swagger UI Documentation]({api_url}/docs)")
st.sidebar.markdown(f"- [ReDoc Documentation]({api_url}/redoc)")

# Tabs
tab_text, tab_image, tab_audio, tab_metrics = st.tabs([
    "✍️ Texte",
    "🖼️ Image (OCR)",
    "🎙️ Audio (ASR)",
    "📊 Métriques du Modèle"
])


def display_prediction_result(data: dict):
    """Affiche de manière visuelle et soignée la réponse JSON unifiée."""
    st.markdown("### 🎯 Résultat de la Prédiction")

    sentiment = data.get("sentiment_predit", "inconnu")
    confiance = data.get("confiance", 0.0)
    source = data.get("source", "texte")
    texte_extrait = data.get("texte_extrait", "")
    details = data.get("details", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        color = "green" if sentiment == "positif" else "red" if sentiment == "negatif" else "gray"
        st.metric(label="Sentiment Prédit", value=sentiment.upper())
    with col2:
        st.metric(label="Score de Confiance", value=f"{confiance * 100:.1f} %")
    with col3:
        st.metric(label="Modalité Source", value=source.upper())

    st.progress(float(confiance))

    st.markdown("#### 📝 Texte extrait / traité")
    st.info(texte_extrait)

    if details:
        st.markdown("#### 📈 Distribution des Probabilités")
        df_probs = pd.DataFrame([details]).T.reset_index()
        df_probs.columns = ["Classe", "Probabilité"]
        st.bar_chart(df_probs.set_index("Classe"))

    with st.expander("🔍 Voir le JSON brut de réponse"):
        st.json(data)


# 1. TAB TEXTE
with tab_text:
    st.subheader("Analyse d'un avis textuel direct")
    sample_text = st.selectbox(
        "Sélectionner un exemple du dataset :",
        [
            "",
            "Produit conforme à la description, livraison super rapide en 24h, je recommande vivement !",
            "Article reçu cassé et le service client ne répond pas depuis une semaine. Très déçu.",
            "Le colis est arrivé dans les délais annoncés. L'emballage est standard."
        ]
    )
    user_text = st.text_area("Ou saisissez votre propre avis client :", value=sample_text, height=120)

    if st.button("🚀 Prédire le sentiment du texte", key="btn_text"):
        if not user_text.strip():
            st.error("Veuillez saisir un texte non vide.")
        else:
            with st.spinner("Analyse textuelle en cours..."):
                try:
                    res = requests.post(f"{api_url}/predict/text", json={"texte": user_text.strip()})
                    if res.status_code == 200:
                        display_prediction_result(res.json())
                    else:
                        st.error(f"Erreur API ({res.status_code}) : {res.text}")
                except Exception as e:
                    st.error(f"Échec de la connexion à l'API : {e}")

# 2. TAB IMAGE (OCR)
with tab_image:
    st.subheader("Analyse d'une capture d'écran / ticket de caisse (OCR)")
    uploaded_image = st.file_uploader("Choisissez une image...", type=["png", "jpg", "jpeg", "webp"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Image importée", width=400)

        if st.button("🔍 Extraire le texte (OCR) & Prédire", key="btn_img"):
            with st.spinner("Extraction OCR et classification du sentiment..."):
                try:
                    files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                    res = requests.post(f"{api_url}/predict/image", files=files)
                    if res.status_code == 200:
                        display_prediction_result(res.json())
                    else:
                        st.error(f"Erreur API ({res.status_code}) : {res.text}")
                except Exception as e:
                    st.error(f"Échec de la connexion à l'API : {e}")

# 3. TAB AUDIO (ASR)
with tab_audio:
    st.subheader("Analyse d'un avis vocal ou dictée audio (ASR)")
    uploaded_audio = st.file_uploader("Choisissez un fichier audio...", type=["wav", "mp3", "ogg", "flac"])

    if uploaded_audio is not None:
        st.audio(uploaded_audio)

        if st.button("🎙️ Transcrire (ASR) & Prédire", key="btn_aud"):
            with st.spinner("Transcription ASR et classification du sentiment..."):
                try:
                    files = {"file": (uploaded_audio.name, uploaded_audio.getvalue(), uploaded_audio.type)}
                    res = requests.post(f"{api_url}/predict/audio", files=files)
                    if res.status_code == 200:
                        display_prediction_result(res.json())
                    else:
                        st.error(f"Erreur API ({res.status_code}) : {res.text}")
                except Exception as e:
                    st.error(f"Échec de la connexion à l'API : {e}")

# 4. TAB METRICS
with tab_metrics:
    st.subheader("Évaluation et métriques du modèle entraîné")
    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Exemples d'entraînement", metrics.get("dataset_size", "N/A"))
        m_col2.metric("Précision CV (5-folds)", f"{metrics.get('cv_accuracy_mean', 0)*100:.1f} %")
        m_col3.metric("Précision Entraînement", f"{metrics.get('training_accuracy', 0)*100:.1f} %")

        st.markdown("#### Matrice de Confusion")
        classes = metrics.get("classes", ["negatif", "neutre", "positif"])
        conf_mat = metrics.get("confusion_matrix", [])
        if conf_mat:
            df_cm = pd.DataFrame(conf_mat, index=classes, columns=classes)
            st.dataframe(df_cm)

        st.markdown("#### Rapport de Classification Détaillé")
        rep = metrics.get("classification_report", {})
        if rep:
            st.json(rep)
    else:
        st.warning("Aucun fichier de métriques trouvé dans `models/metrics.json`. Lancez `python train.py`.")
