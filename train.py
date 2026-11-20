"""Point d'entrée racine pour l'entraînement du modèle sur textes.csv."""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.train import train_model

if __name__ == "__main__":
    train_model()
