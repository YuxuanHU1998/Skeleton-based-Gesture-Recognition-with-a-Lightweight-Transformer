import numpy as np
import os
import csv
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def load_skeleton_data(folder_path, num_frames=60, segment_size=20):
    train_data = []
    train_label = []

    for file in os.listdir(folder_path):
        if not file.endswith(".csv"):
            continue

        current_file = []
        with open(os.path.join(folder_path, file), 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for line in reader:
                current_file.append([float(x) for x in line])
        current_file = np.array(current_file)

        indices = np.linspace(0, len(current_file) - 1, num_frames, dtype=int)
        sampled_frames = current_file[indices]

        # Normalize coordinates
        for i in range(9, 75, 3):
            sampled_frames[:, i] -= sampled_frames[:, 0]
        for i in range(10, 75, 3):
            sampled_frames[:, i] -= sampled_frames[:, 1]
        for i in range(11, 75, 3):
            sampled_frames[:, i] -= sampled_frames[:, 2]

        # Split into segments
        for start in range(0, num_frames, segment_size):
            segment = sampled_frames[start:start + segment_size, :-2].reshape(-1, 3)
            segment = segment.reshape(segment_size, -1, 3)
            train_data.append(segment)
            train_label.append(current_file[0, -2])
    
    combined = list(zip(train_data, train_label))
    random.seed(0)
    random.shuffle(combined)
    train_data, train_label = zip(*combined)

    return train_data, train_label

def uniform_subsample(sequence, target_len=30):
    T_i = sequence.shape[0]

    if T_i < target_len:
        pad_len = target_len - T_i
        pad = np.repeat(sequence[-1:], pad_len, axis=0)
        return np.concatenate([sequence, pad], axis=0)

    indices = np.linspace(0, T_i - 1, target_len).astype(float)
    return sequence[indices]

class GestureDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data      # shape: (N, T, J, 3)
        self.labels = labels  # shape: (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]                     # (T, J, 3)
        x = np.array(x)
        x = x.reshape(x.shape[0], -1)          # (T, J*3)
        y = self.labels[idx]
        x = np.asarray(x, dtype=np.float32)
        y = float(y)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.embedding = nn.Linear(input_dim, 64)
        self.pos_encoder = PositionalEncoding(64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.4,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y = y.clone().detach().long()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y = y.clone().detach().long()
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item() * y.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N, T, J = 570, 20, 25
    num_classes = 14

    train_data, train_label = load_skeleton_data('../trainset')
    all_test_data, all_test_label = load_skeleton_data('../testset')

    val_data, test_data, val_label, test_label = train_test_split(
        all_test_data,
        all_test_label,
        test_size=0.5,
        random_state=0,
        stratify=all_test_label
    )

    train_dataset = GestureDataset(train_data, train_label)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_dataset = GestureDataset(val_data, val_label)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    test_dataset = GestureDataset(test_data, test_label)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    model = TransformerClassifier(input_dim=J * 3, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    num_epochs = 450
    best_acc = 0
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Best Val Acc: {best_acc:.4f}")
    
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc : {test_acc:.4f}")
