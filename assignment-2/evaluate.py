import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from bert4rec_model import BERT4Rec
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
SEQ_LEN = 100
MASK_PROB = 0.15
NUM_LAYERS = 3
NUM_HEADS = 4
HIDDEN_SIZE = 256
DROPOUT = 0.2
BATCH_SIZE = 128
MODEL_PATH = "best_bert4rec.pth"

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, num_items):
        self.sequences = sequences
        self.num_items = num_items

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx].copy()
        input_ids = sequence.copy()
        labels = [-100] * SEQ_LEN

        for i in range(SEQ_LEN):
            if input_ids[i] != 0 and random.random() < MASK_PROB:
                labels[i] = input_ids[i]
                input_ids[i] = 1  # MASK token index

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor((np.array(input_ids) != 0).astype(int), dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        )

def recall_at_k(logits, labels, k=10):
    recalls = []
    for logit, label_seq in zip(logits, labels):
        for pos, true_id in enumerate(label_seq):
            if true_id == -100:
                continue
            topk = torch.topk(logit[pos], k).indices.tolist()
            recalls.append(1 if true_id in topk else 0)
    return np.mean(recalls)

def ndcg_at_k(logits, labels, k=10):
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

def evaluate():
    test_df = pd.read_pickle("data/test.pkl")
    sequences = test_df["sequence"].tolist()
    num_items = max(max(seq) for seq in sequences)

    model = BERT4Rec(
        num_items=num_items,
        hidden_size=HIDDEN_SIZE,
        max_seq_len=SEQ_LEN,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dataset = EvalDataset(sequences, num_items)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    all_logits, all_labels = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader, desc="Evaluating"):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            logits = model(input_ids, attention_mask)
            all_logits.extend(logits.cpu())
            all_labels.extend(labels)

    recall = recall_at_k(all_logits, all_labels, k=10)
    ndcg = ndcg_at_k(all_logits, all_labels, k=10)

    print("\nEvaluation Results:")
    print(f"  Recall@10: {recall:.4f}")
    print(f"  NDCG@10:   {ndcg:.4f}")

if __name__ == "__main__":
    evaluate()
