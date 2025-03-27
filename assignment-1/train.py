import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from neural_collaborative_model import NeuralCollaborativeFiltering
from training import train_model
from sklearn.model_selection import train_test_split

# Load data
train_df = pd.read_csv("./data/train_df.csv")
val_df = pd.read_csv("./data/val_df.csv")

# Encode user_id and movie_id to be contiguous integers
user_ids = pd.concat([train_df['user_id'], val_df['user_id']]).unique()
movie_ids = pd.concat([train_df['movie_id'], val_df['movie_id']]).unique()

user_id_map = {id_: idx for idx, id_ in enumerate(user_ids)}
movie_id_map = {id_: idx for idx, id_ in enumerate(movie_ids)}

train_df['user_id'] = train_df['user_id'].map(user_id_map)
train_df['movie_id'] = train_df['movie_id'].map(movie_id_map)
val_df['user_id'] = val_df['user_id'].map(user_id_map)
val_df['movie_id'] = val_df['movie_id'].map(movie_id_map)

# Create DataLoaders
def create_loader(df, batch_size):
    users = torch.LongTensor(df['user_id'].values)
    items = torch.LongTensor(df['movie_id'].values)
    labels = torch.FloatTensor(df['interaction'].values)
    dataset = TensorDataset(users, items, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = create_loader(train_df, batch_size=128)
val_loader = create_loader(val_df, batch_size=128)

# Model hyperparameters
num_users = len(user_id_map)
num_items = len(movie_id_map)
embed_size = 64
hidden_layers = [128, 64, 32]

# Initialize model
model = NeuralCollaborativeFiltering(num_users, num_items, embed_size, hidden_layers)

# Train model
train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, patience=3)

print("Training complete. Best model saved as 'best_model.pth'.")