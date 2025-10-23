import h5py
import torch
import numpy as np

# Build a neural net that takes in 1024 dim vectors and outputs 1024 dim vectors
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

class EmbeddingMapper(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, hidden_dim=256):  # Reduced from 512
        super().__init__()
        # self.fc1 = nn.Linear(input_dim, input_dim)  # 1024 -> 512 (after GLU halves it)
        # self.fc2 = nn.Linear(512, 512)
        # self.fc3 = nn.Linear(256, 1024)
        # self.fc4 = nn.Linear(512, output_dim)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 256)
        self.fc4 = nn.Linear(256, 1024)
        # self.fc5 = nn.Linear(256, 512)
        # self.fc6 = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(0.2)

        # nn.init.xavier_uniform_(self.fc3.weight, gain=0.1)
        # nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)

        x = F.gelu(self.fc2(x))
        x = self.dropout(x)

        x = F.gelu(self.fc3(x))
        x = self.dropout(x)

        x = F.gelu(self.fc4(x))
        x = self.dropout(x)

        # x = F.gelu(self.fc5(x))
        # x = self.dropout(x)

        # x = F.gelu(self.fc6(x))
        # x = self.dropout(x)

        x = F.normalize(x, p=2, dim=1)
        return x


class TripletDataset(Dataset):
    def __init__(self, h5_path: str):
        with h5py.File(h5_path, "r") as f:
            self.triplets = torch.from_numpy(f["triplets"][:]).float()

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor = self.triplets[idx, 0, :]
        positive = self.triplets[idx, 1, :]
        negative = self.triplets[idx, 2, :]
        return anchor, positive, negative


def main():

    model = EmbeddingMapper(input_dim=1024, output_dim=1024).to(DEVICE)

    # L1 regularization
    # L1_lambda = 1e-5

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    loss_function = nn.TripletMarginWithDistanceLoss(
        margin=0.1,
        distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
    )


    train_dataset = TripletDataset("../src/citeline/nn/np_vectors_train_triplets.h5")
    val_dataset = TripletDataset("../src/citeline/nn/np_vectors_val_triplets.h5")
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
