from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from typing import List
import os

# --- MOCKING CLASSES FOR LOCAL TESTING ---
class MockScaler:
    """A placeholder for the actual scikit-learn scaler."""
    def transform(self, X):
        print(f"Scaler transform called with shape: {X.shape}")
        return X

class MockModel:
    """A placeholder for the actual joblib-loaded model."""
    def predict_proba(self, X):
        print(f"Model predict_proba called with shape: {X.shape}")
        return np.array([[0.2, 0.8]])

# Initialize the mock objects
try:
    model = MockModel()
    scaler = MockScaler()
    print("✓ Mock Model and Scaler initialized successfully")
except Exception as e:
    print(f"✗ Error initializing models: {e}")
    model = None
    scaler = None

# --- FASTAPI SETUP ---
class F1Features(BaseModel):
    """Data model for the input features."""
    features: List[float]

# Create FastAPI app
app = FastAPI(
    title="F1 Winner Predictor API",
    description="API for predicting F1 race winners",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk production, ganti dengan domain spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Runs when the application starts."""
    print("=" * 50)
    print("🚀 FastAPI Application Starting...")
    print(f"Model loaded: {model is not None}")
    print(f"Scaler loaded: {scaler is not None}")
    print(f"Port: {os.environ.get('PORT', 'Not set')}")
    print("=" * 50)

@app.get("/")
async def root():
    """Root endpoint - basic info."""
    return {
        "message": "F1 Winner Predictor API",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/test")
async def test():
    """Simple test endpoint."""
    return {"message": "API is working!", "test": "success"}

@app.post("/predict")
async def predict(data: F1Features):
    """Receives feature data and returns win probability."""
    
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model or scaler not loaded"
        )
    
    try:
        # Convert to numpy array
        input_array = np.array(data.features).reshape(1, -1)
        
        # Validate feature count
        expected_features = 20
        if input_array.shape[1] != expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected_features} features, got {input_array.shape[1]}"
            )
        
        # Scale and predict
        input_scaled = scaler.transform(input_array)
        probability = model.predict_proba(input_scaled)[:, 1].item()
        
        return {
            "winner_probability": round(probability, 4),
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
