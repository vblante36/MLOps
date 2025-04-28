from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn

# Load the model
model = mlflow.sklearn.load_model("/Users/victoriablante/1-MSDS/Spring Mod 2/MLOps/mlruns/4/54b5b9c59a1c456fb1ba8f2922ed6cb9/artifacts/better_models")

print(model)

app = FastAPI(
    title="ML Model Deployment",
    description="A FastAPI app to serve predictions from a trained Decision Tree model.",
    version="1.0"
)

# ✨ Define input format

class ModelInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float
    feature6: float
    feature7: float
    feature8: float
    feature9: float
    feature10: float
    feature11: float
    feature12: float
    feature13: float

# ✨ Adjust predict endpoint
@app.post("/predict/")
def predict(input: ModelInput):
    features = [[
        input.feature1, input.feature2, input.feature3, input.feature4,
        input.feature5, input.feature6, input.feature7, input.feature8,
        input.feature9, input.feature10, input.feature11, input.feature12,
        input.feature13
    ]]
    pred_proba = model.predict_proba(features)[0][1]
    prediction = int(pred_proba >= 0.5)
    return {
        "input_features": features,
        "predicted_probability": pred_proba,
        "predicted_class": prediction
    }