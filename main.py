from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from typing import List
import uvicorn
import os

# --- MOCKING CLASSES FOR LOCAL TESTING ---
class MockScaler:
    """A placeholder for the actual scikit-learn scaler."""
    def transform(self, X):
        return X

class MockModel:
    """A placeholder for the actual joblib-loaded model."""
    def predict_proba(self, X):
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

# CORS configuration: Disesuaikan untuk Streamlit dan localhost
allowed_origins = [
    "http://localhost:8501",  # Frontend Streamlit lokal
    "http://127.0.0.1:8501",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://streamlit-f1-winner-predictor-app.streamlit.app",
    "*"  # Untuk testing, bisa dihapus di production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins, 
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

@app.get("/health")
def health_check():
    """Health check endpoint for Railway."""
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: F1Features):
    """Receives feature data, scales it (mocked), and returns a win probability."""
    
    input_array = np.array(data.features).reshape(1, -1)
    
    expected_features = 20 
    if input_array.shape[1] != expected_features:
        raise HTTPException(
            status_code=400,
            detail=f"Incorrect number of features. Expected {expected_features}, but received {input_array.shape[1]}."
        )
    
    try:
        input_scaled = scaler.transform(input_array)
        probability = model.predict_proba(input_scaled)[:, 1].item()
        
        return {"winner_probability": probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

# PENTING: Untuk Railway deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",  # Ganti "main" dengan nama file Python Anda
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
