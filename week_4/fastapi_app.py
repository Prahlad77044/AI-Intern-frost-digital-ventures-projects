from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch.nn.functional as F

# -----------------------------
# Device & Model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "./imdb_lora_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

MAX_LENGTH = 512

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="IMDB Sentiment Analysis API (JSON Input)")

# -----------------------------
# Pydantic model for input
# -----------------------------
class TextRequest(BaseModel):
    text: str

# -----------------------------
# Predict function
# -----------------------------
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=1).item()

    label = "Positive" if predicted_class == 1 else "Negative"
    confidence = probs[0][predicted_class].item()
    return label, confidence

# -----------------------------
# Endpoint: accept JSON input
# -----------------------------
@app.post("/predict")
async def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text received")
    label, confidence = predict_sentiment(request.text)
    return {"text": request.text, "predicted_label": label, "confidence": confidence}
