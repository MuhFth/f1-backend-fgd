from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from typing import List

# --- MOCKING CLASSES FOR LOCAL TESTING ---
# In a real environment, you would load your actual joblib files here.
# These mocks allow the API endpoints to be tested without the .pkl files.

class MockScaler:
    """A placeholder for the actual scikit-learn scaler."""
    def transform(self, X):
        # Simply returns the input array, bypassing actual scaling.
        return X

class MockModel:
    """A placeholder for the actual joblib-loaded model."""
    def predict_proba(self, X):
        # Returns a mock probability array: [[prob_class_0, prob_class_1]].
        # We return a fixed probability of 80% for winning (class 1) for demonstration.
        return np.array([[0.2, 0.8]])

# Initialize the mock objects
model = MockModel()
scaler = MockScaler()
print("Mock Model and Scaler initialized for local testing.")

# --- FASTAPI SETUP ---

class F1Features(BaseModel):
    """Data model for the input features."""
    features: List[float]

app = FastAPI(title="F1 Winner Predictor API (Mock)")

allowed_origins = [
    "http://localhost:8501",  # Frontend Streamlit atau aplikasi web Anda
    "http://127.0.0.1:8501",
    "http://localhost:8000",  # Untuk pengujian
    "http://127.0.0.1:8000",
    # Jika Anda menggunakan semua origin, Anda bisa tetap menggunakan ["*"]
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins, # Menggunakan daftar spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    """Root endpoint for health checks."""
    return {
        "status": "ready",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "message": "F1 API is running with mock models for local testing."
    }

@app.post("/predict")
def predict(data: F1Features):
    """Receives feature data, scales it (mocked), and returns a win probability."""
    # Since we are using mock models, they are guaranteed to be loaded (model is not None)
    
    input_array = np.array(data.features).reshape(1, -1)
    
    expected_features = 20 
    if input_array.shape[1] != expected_features:
        # This check remains important to ensure the input data shape is correct
        raise HTTPException(
            status_code=400,
            detail=f"Incorrect number of features. Expected {expected_features}, but received {input_array.shape[1]}."
        )
    
    try:
        # 1. Scale the input data using the scaler (mocked)
        input_scaled = scaler.transform(input_array)
        
        # 2. Get the prediction probability (mocked: always 0.8)
        # We take the probability of the positive class (index 1)
        probability = model.predict_proba(input_scaled)[:, 1].item()
        
        return {"winner_probability": probability}

    except Exception as e:
        # General catch for unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")