from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Charger le modèle
model = joblib.load("model/model_california.joblib")

# Créer l'app FastAPI
app = FastAPI(title="API de prédiction des prix d'appartements (Californie)")

# Définir le format d'entrée attendu
class InputData(BaseModel):
    features: list[float]

@app.get("/", tags=["Prédiction"], include_in_schema=False)
def read_root():
    return {"message": "API de prédiction prête"}

@app.post("/predict", tags=["Prédiction"])
def predict(data: InputData):
    # Vérifier si le nombre de features est bien 3
    if len(data.features) != 3:
        raise HTTPException(status_code=400, detail="Le modèle attend exactement 3 features.")
    
    # Convertir les features en DataFrame avec les colonnes que le modèle attend
    features_df = pd.DataFrame([data.features], columns=["median_income", "latitude", "longitude"])

    # Prédiction dans l'espace log
    log_prediction = model.predict(features_df)

    # Retour à l'espace original via exp
    prediction = np.exp(log_prediction)
    
    # Retourner la prédiction transformée
    return {"prediction": prediction.tolist()}
