import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, patience=3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []

        for user_ids, item_ids, labels in train_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(user_ids, item_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for user_ids, item_ids, labels in val_loader:
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                labels = labels.to(device).float()

                outputs = model(user_ids, item_ids)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break
