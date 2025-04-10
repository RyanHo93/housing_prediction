# tests/test_api.py
from pathlib import Path
import sys

# Ajouter le dossier parent au path pour permettre l'import de main.py
sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import app  # maintenant ça devrait marcher
from fastapi.testclient import TestClient

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_prediction():
    data = {"features": [3.5, 34.0, -118.0]}
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
