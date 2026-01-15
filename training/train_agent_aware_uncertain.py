import os, glob, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os, glob, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# CONFIG
# =========================
DATA_DIR = "nuplan_ml_dataset/samples"
NUM_SAMPLES = 50000
BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-3

H = 10
T = 20
N = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATASET
# =========================
class PlannerDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        return (
            torch.from_numpy(d["ego_past"]).float(),
            torch.from_numpy(d["agents_past"]).float(),
            torch.from_numpy(d["agents_mask"]).float(),
            torch.from_numpy(d["ego_future"]).float(),
        )

# =========================
# MODEL (MC-Dropout)
# =========================
class UncertainPlanner(nn.Module):
    def __init__(self):
        super().__init__()

        self.ego_enc = nn.Sequential(
            nn.Linear(H * 5, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 256),
        )

        self.agent_enc = nn.Sequential(
            nn.Linear(H * 5, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 128),
        )

        self.decoder = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, T * 5),
        )

    def forward(self, ego, agents, mask):
        B = ego.size(0)

        ego_feat = self.ego_enc(ego.view(B, -1))

        agents_flat = agents.view(B, N, -1)
        agent_feat = self.agent_enc(agents_flat)

        mask = mask.unsqueeze(-1)
        pooled = (agent_feat * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)

        fused = torch.cat([ego_feat, pooled], dim=1)
        out = self.decoder(fused)

        return out.view(B, T, 5)

# =========================
# TRAINING
# =========================
def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    random.shuffle(files)
    files = files[:NUM_SAMPLES]

    loader = DataLoader(
        PlannerDataset(files),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = UncertainPlanner().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("Training uncertainty-aware planner")

    for epoch in range(EPOCHS):
        total = 0.0
        for ego, agents, mask, gt in loader:
            ego, agents, mask, gt = (
                ego.to(DEVICE),
                agents.to(DEVICE),
                mask.to(DEVICE),
                gt.to(DEVICE),
            )

            pred = model(ego, agents, mask)
            loss = loss_fn(pred, gt)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total/len(loader):.6f}")

    torch.save(model.state_dict(), "uncertain_planner.pt")
    print("Saved uncertain_planner.pt")

if __name__ == "__main__":
    main()
