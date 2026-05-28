# app.py
# Usage: uvicorn app:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from contextlib import asynccontextmanager
from utils import load_config
from data_processing import clean_text

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
model = None
tokenizer = None
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, device
    cfg = load_config()
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name"])
    m = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["model_name"], num_labels=cfg["model"]["num_labels"]
    )
    m.load_state_dict(torch.load(
        cfg["training"]["output_dir"] + "best_model.pt", map_location=device
    ))
    m.to(device)
    m.eval()
    model = m
    yield


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    scores: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    cleaned = clean_text(request.text)
    tokens = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()
        predicted_id = torch.argmax(probs).item()

    return PredictResponse(
        text=request.text,
        sentiment=LABEL_MAP[predicted_id],
        confidence=round(probs[predicted_id].item(), 4),
        scores={LABEL_MAP[i]: round(probs[i].item(), 4) for i in range(len(LABEL_MAP))}
    )
