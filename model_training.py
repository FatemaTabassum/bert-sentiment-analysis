import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification
from utils import compute_metrics # Assuming utils.py is in the same directory
import numpy as np
from sklearn.metrics import accuracy_score # Import here for eval_step

def load_model(num_labels=3, model_name="bert-base-uncased"):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model

def setup_training(model, learning_rate=2e-5, freeze_bert_layers=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # print("Model parameters before freezing:")
    # for name, param in model.named_parameters():
    #     print(f"  {name}")
    #     print(f"  {param.shape}")
    #     print(f"  {param.requires_grad}")
    #     print(f"  {param.grad}")
    #     print(f"  {param.grad_fn}")

    if freeze_bert_layers:
        for name, param in model.named_parameters():
            if 'classifier' not in name:  # Keep the classifier layer trainable
                param.requires_grad = False


    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    return device, optimizer, loss_fn

def train_step(dataloader, model, loss_fn, optimizer, device):
    model.train()  # Set model to training mode
    total_loss = 0
    for batch in dataloader:
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        optimizer.zero_grad()  # Clear previous gradients

        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs.loss
        # logits = outputs.logits # Not directly used for loss calculation here as labels are passed to model

        total_loss += loss.item()

        loss.backward()
        optimizer.step()  # Update weights

    avg_train_loss = total_loss / len(dataloader)
    return avg_train_loss

def eval_step(dataloader, model, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():  # Disable gradient calculations
        for batch in dataloader:
            b_input_ids = batch[0].to(device)
            b_attention_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            true_labels = b_labels.cpu().numpy()

            all_predictions.extend(predictions)
            all_true_labels.extend(true_labels)

    avg_eval_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_true_labels, all_predictions)
    return avg_eval_loss, accuracy

def run_training_loop(model, train_dataloader, val_dataloader, device, optimizer, loss_fn, num_epochs=3):
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_step(train_dataloader, model, loss_fn, optimizer, device)
        print(f"  Training Loss: {train_loss:.4f}")

        val_loss, val_accuracy = eval_step(val_dataloader, model, loss_fn, device)
        print(f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

    wandb.finish()
    print("\nTraining finished!")



