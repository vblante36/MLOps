from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)

# Load model at startup
model = joblib.load("reddit_model_pipeline.joblib")

class CommentInput(BaseModel):
    text: str

@app.get("/")
def main():
    return {"message": "This is a model for classifying Reddit comments"}

@app.post("/predict")
def predict(input: CommentInput):
    pred = model.predict([input.text])[0]
    proba = model.predict_proba([input.text])[0][1]
    return {"text": input.text, "prediction": int(pred), "probability_remove": float(proba)}