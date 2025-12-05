from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model
with open("model/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Churn Prediction API")

# Define input schema
class CustomerData(BaseModel):
    Age: float
    Tenure: float
    Balance: float
    Products: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running!"}

@app.post("/predict")
def predict(data: CustomerData):
    features = np.array([[data.Age, data.Tenure, data.Balance, data.Products, data.IsActiveMember, data.EstimatedSalary]])
    prediction = model.predict(features)[0]
    return {"Predicted_Churn": int(prediction)}
