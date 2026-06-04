
from fastapi import FastAPI
from pydantic import BaseModel

from utils.inference import predict_feedback

app = FastAPI(title="SALC API")

class PredictRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "SALC API aktif"}

@app.post("/predict")
def predict(req: PredictRequest):

    result = predict_feedback(req.text)

    return result
