# BERT Sentiment Analysis

![CI](https://github.com/FatemaTabassum/bert-sentiment-analysis/actions/workflows/ci.yml/badge.svg)

Fine-tuning `bert-base-uncased` for 3-class sentiment classification (positive, neutral, negative) with a config-driven pipeline, W&B experiment tracking, and FastAPI serving.

## Overview

This project fine-tunes a pre-trained BERT model on the [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) sentiment benchmark. It also includes a custom patient feedback dataset for domain-specific testing.

**Labels:** `0 = negative` · `1 = neutral` · `2 = positive`

## Project Structure

```
bert-sentiment-analysis/
├── main.py                  # Entry point — orchestrates the full pipeline
├── data_processing.py       # Data loading, cleaning, tokenization, dataloaders
├── model_training.py        # Training loop, evaluation, W&B logging
├── utils.py                 # Metrics (accuracy, F1) and config loader
├── predict.py               # CLI inference script
├── app.py                   # FastAPI REST API server
├── config/
│   ├── config.yaml          # Main config entry point
│   ├── data_config.yaml     # Dataset and preprocessing settings
│   ├── model_config.yaml    # Model architecture settings
│   └── training_config.yaml # Optimizer and training settings
├── data/
│   ├── patient_feedback.csv # Custom 30-sample patient feedback dataset
│   └── patient_feedback.json
├── requirements.txt
└── .gitignore
```

## Setup

```bash
# Clone the repo
git clone https://github.com/FatemaTabassum/bert-sentiment-analysis.git
cd bert-sentiment-analysis

# Install dependencies
pip install -r requirements.txt
```

## Configuration

All hyperparameters are managed via YAML files in `config/`. No hardcoded values in code.

| Config file | Key settings |
|---|---|
| `data_config.yaml` | `dataset_name`, `max_samples`, `batch_size`, `max_length` |
| `model_config.yaml` | `model_name`, `num_labels`, `freeze_bert_layers` |
| `training_config.yaml` | `learning_rate`, `num_epochs`, `optimizer` |

## Usage

```bash
# Train on HuggingFace tweet sentiment dataset (default)
python main.py

# To use the custom patient feedback dataset instead,
# uncomment Option B in main.py
```

## Experiment Tracking

This project uses [Weights & Biases](https://wandb.ai) to track training metrics.

To enable tracking, add your API key to a `.env` file:

```
WANDB_API_KEY=your_key_here
```

Each run logs: `train_loss`, `val_loss`, `val_accuracy` per epoch.

## Dataset

**Default:** `cardiffnlp/tweet_eval` (sentiment config) — ~45k tweets, 3 classes, loaded directly from HuggingFace.

**Custom fallback:** 30 patient feedback samples in `data/` — balanced across positive, neutral, and negative classes.

## REST API (FastAPI)

Start the API server:

```bash
uvicorn app:app --reload
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Predict sentiment for a text |

### Example request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "The doctor was very kind and attentive."}'
```

### Example response

```json
{
  "text": "The doctor was very kind and attentive.",
  "sentiment": "positive",
  "confidence": 0.8581,
  "scores": {
    "negative": 0.0363,
    "neutral": 0.1056,
    "positive": 0.8581
  }
}
```

Interactive API docs available at `http://127.0.0.1:8000/docs` when the server is running.

## Results

| Setting | Test Accuracy |
|---|---|
| Frozen BERT (classifier only) | 48.6% |
| Full fine-tuning (all layers) | **66.4%** |

## CI

This project uses GitHub Actions to run `ruff` linting and import checks on every push to `main`.

## Tech Stack

- [Transformers](https://huggingface.co/docs/transformers) — BERT model and tokenizer
- [PyTorch](https://pytorch.org) — custom training loop with MPS (Apple GPU) support
- [Weights & Biases](https://wandb.ai) — experiment tracking
- [FastAPI](https://fastapi.tiangolo.com) — REST API serving
- [scikit-learn](https://scikit-learn.org) — metrics and train/val/test split
- [PyYAML](https://pyyaml.org) — config management
- [ruff](https://docs.astral.sh/ruff/) — linting (CI)
