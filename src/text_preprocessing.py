"""Module de prétraitement et de nettoyage du texte français."""
import re
import unicodedata
from typing import List, Set

# Mots vides en français ne portant pas de charge émotionnelle directe.
# IMPORTANT : On conserve explicitement les marqueurs de négation et d'intensité
# ('ne', 'pas', 'plus', 'aucun', 'sans', 'jamais', 'rien', 'trop', 'tres', etc.)
# car ils inversent ou amplifient radicalement le sentiment.
FRENCH_STOP_WORDS_NEUTRAL: Set[str] = {
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l",
    "et", "ou", "mais", "a", "au", "aux", "avec", "en", "dans",
    "par", "pour", "sur", "sous", "vers", "chez",
    "ce", "cet", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "notre", "nos", "votre", "vos", "leur", "leurs",
    "qui", "que", "quoi", "dont", "ou",
    "il", "elle", "on", "ils", "elles", "je", "tu", "nous", "vous",
    "est", "sont", "ete", "etre", "avoir", "ai", "as", "a", "avons", "avez", "ont",
    "ceci", "cela", "ca"
}

def normalize_accents(text: str) -> str:
    """Supprime les accents pour uniformiser les variantes orthographiques."""
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Nettoie et normalise une chaîne de texte français :
    - Mise en minuscules
    - Suppression des URLs et mentions
    - Normalisation des accents
    - Remplacement de la ponctuation superflue par des espaces
    - Filtrage optionnel des mots vides neutres
    - Réduction des espaces multiples
    """
    if not isinstance(text, str):
        return ""
    
    # Mise en minuscules
    cleaned = text.lower()
    
    # Suppression des URLs (http/https/www)
    cleaned = re.sub(r'https?://\S+|www\.\S+', ' ', cleaned)
    
    # Suppression des mentions réseaux sociaux (@user) et hashtags (#tag)
    cleaned = re.sub(r'[@#]\w+', ' ', cleaned)
    
    # Normalisation des accents
    cleaned = normalize_accents(cleaned)
    
    # Conservation des lettres alphabétiques et espaces
    cleaned = re.sub(r'[^a-z\s]', ' ', cleaned)
    
    # Découpage en tokens
    tokens: List[str] = cleaned.split()
    
    if remove_stopwords:
        tokens = [token for token in tokens if token not in FRENCH_STOP_WORDS_NEUTRAL]
        
    return " ".join(tokens)
