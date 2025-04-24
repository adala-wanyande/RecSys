import pandas as pd  # type: ignore
import os
import numpy as np  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

DATA_DIR = "data"
SEQUENCE_LENGTH = 100

def load_data():
    ratings_path = os.path.join(DATA_DIR, "ratings.dat")
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"]
    )
    return ratings

def preprocess_data(ratings):
    # Keep only ratings >= 4
    ratings = ratings[ratings["rating"] >= 4]

    # Sort by user and timestamp
    ratings = ratings.sort_values(by=["userId", "timestamp"])

    # Group into user sequences
    user_sequences = ratings.groupby("userId")["movieId"].apply(list)

    # Filter out users with fewer than 5 interactions
    user_sequences = user_sequences[user_sequences.apply(len) >= 5]

    return user_sequences

def pad_or_truncate(seq, max_len=SEQUENCE_LENGTH):
    if len(seq) >= max_len:
        return seq[-max_len:]
    else:
        return [0] * (max_len - len(seq)) + seq

def train_val_test_split(user_sequences):
    df = pd.DataFrame({'userId': user_sequences.index, 'sequence': user_sequences.values})
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    return train, val, test

def preprocess_and_save():
    ratings = load_data()
    user_sequences = preprocess_data(ratings)
    user_sequences = user_sequences.apply(lambda seq: pad_or_truncate(seq, SEQUENCE_LENGTH))
    train, val, test = train_val_test_split(user_sequences)

    train.to_pickle("data/train.pkl")
    val.to_pickle("data/val.pkl")
    test.to_pickle("data/test.pkl")

    print("Preprocessing complete. Saved train.pkl, val.pkl, and test.pkl.")

if __name__ == "__main__":
    preprocess_and_save()
