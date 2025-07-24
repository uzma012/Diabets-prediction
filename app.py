from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load the trained model
model = joblib.load("model.pkl")

# Create app
app = FastAPI()

# Allow CORS (needed for frontend calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Prediction route
@app.post("/predict")
def predict(data: PatientData):
    input_array = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])
    prediction = model.predict(input_array)[0]
    print(f"Prediction: {prediction}")  # Debugging output
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    return {"prediction": result}
