from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np

app = FastAPI()

templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

model = joblib.load("model.pkl")
threshold = 0.3



class PatientData(BaseModel):
    Fever: int
    Cough: int
    Fatigue: int
    Difficulty_Breathing: int
    Age: int
    Gender: int
    Blood_Pressure: int
    Cholesterol_Level: int



@app.get("/")
def home(request: Request):
    return templates.TemplateResponse( request=request, name="index.html")

# @app.get("/")
# def home():
#     return {"message": "Welcome to the COVID-19 Prediction API. Use the /predict endpoint to get predictions."}



@app.post("/predict")
def predict(data: PatientData):

    input_data = np.array([[

        data.Age,

        data.Fever,
        data.Cough,
        data.Fatigue,
        data.Difficulty_Breathing,

        data.Gender,

        # Blood Pressure (dummy)
        1 if data.Blood_Pressure == 0 else 0,   # Low
        1 if data.Blood_Pressure == 1 else 0,   # Normal

        # Cholesterol (dummy)
        1 if data.Cholesterol_Level == 0 else 0,  # Low
        1 if data.Cholesterol_Level == 1 else 0   # Normal

    ]])

    prob = model.predict_proba(input_data)[0][1]

    prediction = "Positive" if prob >= threshold else "Negative"

    return {
        "prediction": prediction,
        "probability": float(prob)
    }
