# data_processing.py

# This file will contain all functions related to data cleaning, tokenization, 
# and splitting the dataset into training,
# validation, and test sets. It encapsulates the data preparation pipeline.
 
 
import re
from typing import Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def load_and_preprocess_data(data_dict):
    data = pd.DataFrame(data_dict)
    data["cleaned_text"] = data["text"].apply(clean_text)
    # Handle missing values in 'cleaned_text' by filling with 'missing'
    data['cleaned_text'].fillna('missing', inplace=True)
    label_map = {"positive": 0, "neutral": 1, "negative": 2}
    data["label"] = data["label"].map(label_map)
    return data

def load_hf_sentiment_data(split: str = "train", max_samples: Optional[int] = None) -> pd.DataFrame:
    """Load tweet sentiment data from cardiffnlp/tweet_eval on HuggingFace.

    Labels: 0=negative, 1=neutral, 2=positive.
    """
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment", split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    data = pd.DataFrame({"text": dataset["text"], "label": dataset["label"]})
    data["cleaned_text"] = data["text"].apply(clean_text)
    data["cleaned_text"] = data["cleaned_text"].fillna("missing")
    return data

def tokenize_data(data, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer(
        data['cleaned_text'].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=128
    )
    return tokens, tokenizer

def create_dataloaders(tokens, labels, batch_size=16, random_state=42):
    input_ids = tokens['input_ids']
    attention_masks = tokens['attention_mask']

    dataset_size = input_ids.shape[0]
    # print('dataset size: ', dataset_size)
    indices = list(range(dataset_size))
    # print('indices: ', indices)

    train_val_indices, test_indices = train_test_split(
        indices, test_size=0.1, random_state=random_state
    )
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=0.15, random_state=random_state
    )

    train_dataset = torch.utils.data.TensorDataset(
        input_ids[train_indices],
        attention_masks[train_indices],
        labels[train_indices]
    )
    val_dataset = torch.utils.data.TensorDataset(
        input_ids[val_indices],
        attention_masks[val_indices],
        labels[val_indices]
    )
    test_dataset = torch.utils.data.TensorDataset(
        input_ids[test_indices],
        attention_masks[test_indices],
        labels[test_indices]
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    # print('train_dataloader: ', train_dataloader)
    # print('val_dataloader: ', val_dataloader)
    # print('test_dataloader: ', test_dataloader)
    # print('train_dataloader length: ', len(train_dataloader))
    # print('val_dataloader length: ', len(val_dataloader))
    # print('test_dataloader length: ', len(test_dataloader))
    # print('train_dataloader batch size: ', train_dataloader.batch_size)
    # print('val_dataloader batch size: ', val_dataloader.batch_size)
    # print('test_dataloader batch size: ', test_dataloader.batch_size)
    # print('dataset length in train_dataloader: ', len(train_dataloader.dataset))
    # print('dataset length in val_dataloader: ', len(val_dataloader.dataset))
    # print('dataset length in test_dataloader: ', len(test_dataloader.dataset))
    # print('dataset in train_dataloader: ', train_dataloader.dataset)
    # print('dataset in val_dataloader: ', val_dataloader.dataset)
    # print('dataset in test_dataloader: ', test_dataloader.dataset)
    # print('dataset length in train_dataloader: ', len(train_dataloader.dataset))

    return train_dataloader, val_dataloader, test_dataloader