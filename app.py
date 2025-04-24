from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for GUI communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
try:
    model = joblib.load("models/linear_regression_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    raise Exception(f"Error loading model or scaler: {str(e)}")

# Define input data model


class InputData(BaseModel):
    Make: int
    Model: int
    Vehicle_Class: int
    Engine_Size_L: float
    Transmission: int
    Fuel_Type: int
    Fuel_Consumption_Hwy_L_100km: float


@app.get("/")
async def root():
    return {"message": "CO2 Emissions Prediction API"}


@app.post("/predict")
async def predict(data: InputData):
    try:
        # Convert input data to array
        input_array = np.array([[
            data.Make,
            data.Model,
            data.Vehicle_Class,
            data.Engine_Size_L,
            data.Transmission,
            data.Fuel_Type,
            data.Fuel_Consumption_Hwy_L_100km
        ]])

        # Scale the numerical features (columns 3 and 6) while keeping others unchanged
        # Engine_Size_L and Fuel_Consumption_Hwy_L_100km
        numerical_indices = [3, 6]
        scaled_numerical = scaler.transform(input_array[:, numerical_indices])

        # Update the numerical columns in the input array
        input_array[:, numerical_indices] = scaled_numerical

        # Make prediction
        prediction = float(model.predict(input_array)[0])

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")
