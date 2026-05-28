# main.py

# This is the entry point of the project.
# It will import functions from data_processing.py, model_training.py, and utils.py to
# orchestrate the entire fine-tuning workflow.

import wandb
from dotenv import load_dotenv
from utils import compute_metrics, load_config
from data_processing import load_and_preprocess_data, load_hf_sentiment_data, tokenize_data, create_dataloaders
from model_training import train_step, eval_step, run_training_loop, load_model, setup_training
import torch

load_dotenv()


def get_custom_data():
    """Small patient-feedback example for quick local testing."""
    return {
        "text": [
            "  The staff was very kind and attentive to my needs!!!  ",
            "The waiting time was too long, and the staff was rude. Visit us at http://hospitalreviews.com",
            "The doctor answered all my questions...but the facility was outdated.   ",
            "The nurse was compassionate & made me feel comfortable!! :) ",
            "I had to wait over an hour before being seen.  Unacceptable service! #frustrated",
            "The check-in process was smooth, but the doctor seemed rushed. Visit https://feedback.com",
            "Everyone I interacted with was professional and helpful.  "
        ],
        "label": ["positive", "negative", "neutral", "positive", "negative", "neutral", "positive"]
    }


def main():

    cfg = load_config()

    wandb.init(
        project=cfg["project_name"],
        config={
            **cfg["data"],
            **cfg["model"],
            **cfg["training"]
        }
    )

    # --- Data loading ---
    # Option A: HuggingFace tweet sentiment benchmark (0=negative, 1=neutral, 2=positive)
    print("Loading dataset from HuggingFace...")
    data_dict = load_hf_sentiment_data(
        split=cfg["data"]["split"],
        max_samples=cfg["data"]["max_samples"]
        )

    # Option B: small custom patient-feedback example (uncomment to use instead)
    # data_dict = get_custom_data()
    # data_dict = load_and_preprocess_data(data_dict)

    labels = torch.tensor(data_dict["label"].tolist())

    # --- Tokenize ---
    print("Tokenizing data...")
    tokens, tokenizer = tokenize_data(data_dict, model_name=cfg["model"]["model_name"])

    # --- DataLoaders ---
    print("Creating DataLoaders...")
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        tokens, labels,
        batch_size=cfg["data"]["batch_size"],
        random_state=cfg["data"]["random_state"]
    )

    # --- Model ---
    print("Loading model...")
    model = load_model(
        num_labels=cfg["model"]["num_labels"],
        model_name=cfg["model"]["model_name"]
    )

    # --- Training setup ---
    print("Setting up training...")
    device, optimizer, loss_fn = setup_training(
        model,
        learning_rate=cfg["training"]["learning_rate"],
        freeze_bert_layers=cfg["model"]["freeze_bert_layers"]
    )

    # --- Train ---
    print("Starting training loop...")
    run_training_loop(
        model, train_dataloader, val_dataloader,
        device, optimizer, loss_fn,
        num_epochs=cfg["training"]["num_epochs"],
        output_dir=cfg["training"]["output_dir"]
    )

    # --- Evaluate on test set ---
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = eval_step(test_dataloader, model, loss_fn, device)
    print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
