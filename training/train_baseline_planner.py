import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# CONFIG
# =========================
DATA_DIR = "nuplan_ml_dataset/samples"
NUM_SAMPLES = 50000      # subsample
BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-3

H = 10   # history
T = 20   # future

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATASET
# =========================
class EgoPlannerDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        ego_past = d["ego_past"]          # (H, 5)
        ego_future = d["ego_future"]      # (T, 5)

        return (
            torch.from_numpy(ego_past).float(),
            torch.from_numpy(ego_future).float(),
        )

# =========================
# MODEL
# =========================
class MLPPlanner(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(H * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, T * 5),
        )

    def forward(self, ego_past):
        x = ego_past.view(ego_past.size(0), -1)
        out = self.net(x)
        return out.view(-1, T, 5)

# =========================
# TRAINING
# =========================
def main():
    all_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    random.shuffle(all_files)
    files = all_files[:NUM_SAMPLES]

    dataset = EgoPlannerDataset(files)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = MLPPlanner().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f"Training on {len(dataset)} samples on {DEVICE}")

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for ego_past, ego_future in loader:
            ego_past = ego_past.to(DEVICE)
            ego_future = ego_future.to(DEVICE)

            pred = model(ego_past)
            loss = loss_fn(pred, ego_future)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "baseline_planner.pt")
    print("Saved model to baseline_planner.pt")

if __name__ == "__main__":
    main()
