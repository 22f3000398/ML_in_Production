"""
FastAPI app serving an Amazon review sentiment model.

Run locally:
    uvicorn main:app --reload

Test it:
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"review": "This product is amazing and I really enjoyed it!"}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


app = FastAPI(
    title="Amazon Review Sentiment API",
    description="A simple FastAPI prediction API for Amazon review sentiment.",
    version="1.0.0",
)


# Load the trained model once when the API starts
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


# Request format
class PredictionRequest(BaseModel):
    review: str = Field(
        ...,
        min_length=1,
        description="Amazon product review text"
    )


# Response format
class PredictionResponse(BaseModel):
    prediction: int
    sentiment: str


@app.get("/")
def root():
    return {
        "message": "Amazon Review Sentiment API is running.",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Make sure model.pkl exists."
        )

    # Get the review text
    review = [request.review]

    # Make prediction
    prediction = int(model.predict(review)[0])

    # Convert prediction to sentiment
    if prediction == 1:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return PredictionResponse(
        prediction=prediction,
        sentiment=sentiment
    )