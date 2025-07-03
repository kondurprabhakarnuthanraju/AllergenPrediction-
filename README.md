# AllergenPrediction-
# 🧬 Allergen Prediction using Pre-trained Language Models (PLMs)

A deep learning-based project for predicting allergens from protein sequences using cutting-edge Pre-trained Language Models (PLMs). This tool aims to assist in the identification of allergenic proteins in food and biomedical research.

---

## 🚀 Overview

This project leverages state-of-the-art transformer-based protein language models (PLMs) such as **ProtBERT**, **ESM**, and **ProtT5** to predict whether a given protein sequence is an allergen or not.

---

## 🧠 Key Features

- ✅ **Preprocessing** of protein sequences
- 🔍 **Embedding** using PLMs (e.g., ProtBERT, ESM, ProtT5)
- 🧪 **Binary classification**: Allergen vs Non-Allergen
- 📈 **Evaluation metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 🧰 Compatible with GPU acceleration (via PyTorch)

---

## 📊 Dataset

We use a curated dataset of allergenic and non-allergenic protein sequences sourced from:

- **AllergenOnline**
- **UniProt**
- **NCBI Protein Database**

> 💡 Preprocessed data available in the `/data/` folder.

---

## 🛠️ Requirements

```bash
python>=3.8
torch>=1.10
transformers>=4.12.0
scikit-learn
pandas
numpy
matplotlib
tqdm
# 🧬 Allergen Prediction using Pre-trained Language Models (PLMs)

A deep learning-based project for predicting allergens from protein sequences using cutting-edge Pre-trained Language Models (PLMs). This tool assists in the identification of allergenic proteins in food and biomedical research.

---

## 🚀 Overview

This project leverages state-of-the-art transformer-based protein language models (PLMs) such as **ProtBERT** to predict whether a given protein sequence is an allergen or not.

---

## 🧠 Key Features

- ✅ Preprocessing of protein sequences
- 🔍 Embedding using ProtBERT
- 🧪 Binary classification: Allergen vs Non-Allergen
- 📈 Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 🧰 Compatible with GPU acceleration

---

## 📊 Dataset

A CSV file with labeled allergenic and non-allergenic protein sequences. Format:

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

MODEL_NAME = "Rostlab/prot_bert"
BATCH_SIZE = 8
EPOCHS = 5
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = " ".join(list(self.sequences[idx]))
        enc = self.tokenizer(seq, padding="max_length", max_length=MAX_LEN,
                             truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }

class AllergenClassifier(nn.Module):
    def __init__(self):
        super(AllergenClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        return torch.sigmoid(self.classifier(pooled_output))

df = pd.read_csv("dataset.csv")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["sequence"].values, df["label"].values, test_size=0.2, random_state=42
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_dataset = ProteinDataset(train_texts, train_labels, tokenizer)
val_dataset = ProteinDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = AllergenClassifier().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.BCELoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask).squeeze()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} loss: {total_loss/len(train_loader)}")

torch.save(model.state_dict(), "model.pt")


import torch
from sklearn.metrics import classification_report
from train import AllergenClassifier, ProteinDataset, tokenizer, DEVICE
import pandas as pd
from torch.utils.data import DataLoader

model = AllergenClassifier().to(DEVICE)
model.load_state_dict(torch.load("model.pt"))
model.eval()

df = pd.read_csv("dataset.csv")
sequences = df["sequence"].values
labels = df["label"].values
dataset = ProteinDataset(sequences, labels, tokenizer)
loader = DataLoader(dataset, batch_size=8)

all_preds, all_labels = [], []

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].numpy()

        outputs = model(input_ids, attention_mask).squeeze()
        preds = (outputs.cpu().numpy() > 0.5).astype(int)

        all_preds.extend(preds)
        all_labels.extend(labels)

print(classification_report(all_labels, all_preds, digits=4))
  import torch
from train import AllergenClassifier, tokenizer, DEVICE, MAX_LEN

def predict(sequence):
    model = AllergenClassifier().to(DEVICE)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    seq = " ".join(list(sequence))
    enc = tokenizer(seq, return_tensors="pt", truncation=True,
                    padding="max_length", max_length=MAX_LEN)

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        output = model(input_ids, attention_mask).squeeze().item()
        print(f"Predicted probability of being Allergen: {output:.4f}")
        if output >= 0.5:
            print("⚠️ Likely Allergen")
        else:
            print("✅ Likely Non-Allergen")

# Example usage
if __name__ == "__main__":
    predict("MKWVTFISLLFLFSSAYS")

