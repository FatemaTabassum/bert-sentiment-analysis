# predict.py
# Usage: python predict.py "The doctor was very kind and attentive."

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import load_config
from data_processing import clean_text

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def load_trained_model(model_path, model_name, num_labels, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(text, model, tokenizer, device):
    cleaned = clean_text(text)
    tokens = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()
        predicted_id = torch.argmax(probs).item()

    return {
        "text": text,
        "sentiment": LABEL_MAP[predicted_id],
        "confidence": round(probs[predicted_id].item(), 4),
        "scores": {LABEL_MAP[i]: round(probs[i].item(), 4) for i in range(len(LABEL_MAP))}
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Your text here\"")
        sys.exit(1)

    text = sys.argv[1]
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = cfg["training"]["output_dir"] + "best_model.pt"
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name"])
    model = load_trained_model(
        model_path, cfg["model"]["model_name"], cfg["model"]["num_labels"], device
    )

    result = predict(text, model, tokenizer, device)
    print(f"\nText      : {result['text']}")
    print(f"Sentiment : {result['sentiment'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Scores    : {result['scores']}")


if __name__ == "__main__":
    main()
