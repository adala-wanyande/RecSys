import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from bert4rec_model import BERT4Rec
from transformers import get_cosine_schedule_with_warmup
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--model_save_path", type=str, default="best_bert4rec.pth")
args = parser.parse_args()

# Constants
SEQUENCE_LENGTH = 100
MASK_PROB = 0.15
EPOCHS = 200
PATIENCE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASK_TOKEN = 1  # Reserved index for [MASK] token

class BERT4RecDataset(Dataset):
    def __init__(self, dataframe, vocab_size):
        self.sequences = dataframe["sequence"].tolist()
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_ids = sequence.copy()
        labels = [-100] * SEQUENCE_LENGTH

        for i in range(SEQUENCE_LENGTH):
            if input_ids[i] != 0 and random.random() < MASK_PROB:
                labels[i] = input_ids[i]
                rand = random.random()
                if rand < 0.8:
                    input_ids[i] = MASK_TOKEN
                elif rand < 0.9:
                    input_ids[i] = random.randint(2, self.vocab_size - 1)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = (input_ids != 0).long()

        return input_ids, attention_mask, labels

def compute_ndcg_k(logits, labels, k=10):
    ndcgs = []
    for logit, label_seq in zip(logits, labels):
        for pos, true_id in enumerate(label_seq):
            if true_id == -100:
                continue
            topk = torch.topk(logit[pos], k).indices.tolist()
            if true_id in topk:
                rank = topk.index(true_id) + 1
                ndcgs.append(1 / np.log2(rank + 1))
            else:
                ndcgs.append(0)
    return np.mean(ndcgs)

def train():
    print("Loading data...")
    train_df = pd.read_pickle("data/train.pkl")
    val_df = pd.read_pickle("data/val.pkl")

    max_item_id = max(max(seq) for seq in train_df["sequence"])
    vocab_size = max_item_id + 2

    model = BERT4Rec(
        num_items=vocab_size - 2,
        hidden_size=args.hidden_size,
        max_seq_len=SEQUENCE_LENGTH,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(DEVICE)

    train_dataset = BERT4RecDataset(train_df, vocab_size)
    val_dataset = BERT4RecDataset(val_df, vocab_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_ndcg = 0
    patience_counter = 0

    print("Training started.")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(input_ids, attention_mask)

            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")

        model.eval()
        all_logits, all_labels = [], []

        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                logits = model(input_ids, attention_mask)
                all_logits.extend(logits.cpu())
                all_labels.extend(labels)

        val_ndcg = compute_ndcg_k(all_logits, all_labels, k=10)
        print(f"Validation NDCG@10: {val_ndcg:.4f}")

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            patience_counter = 0
            torch.save(model.state_dict(), args.model_save_path)
            print("Saved new best model.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    train()
