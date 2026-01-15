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
NUM_SAMPLES = 50000
BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-3

H = 10
T = 20
N = 32   # max agents

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
            torch.from_numpy(d["ego_past"]).float(),        # (H, 5)
            torch.from_numpy(d["agents_past"]).float(),     # (N, H, 5)
            torch.from_numpy(d["agents_mask"]).float(),     # (N,)
            torch.from_numpy(d["ego_future"]).float(),      # (T, 5)
        )

# =========================
# MODEL
# =========================
class AgentAwarePlanner(nn.Module):
    def __init__(self):
        super().__init__()

        # Ego encoder
        self.ego_enc = nn.Sequential(
            nn.Linear(H * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # Agent encoder (shared)
        self.agent_enc = nn.Sequential(
            nn.Linear(H * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # Fusion + decoder
        self.decoder = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, T * 5),
        )

    def forward(self, ego_past, agents_past, agents_mask):
        B = ego_past.size(0)

        # Ego embedding
        ego_feat = self.ego_enc(ego_past.view(B, -1))  # (B, 256)

        # Agent embeddings
        agents_flat = agents_past.view(B, N, -1)       # (B, N, H*5)
        agent_feat = self.agent_enc(agents_flat)       # (B, N, 128)

        # Masked mean pooling
        mask = agents_mask.unsqueeze(-1)               # (B, N, 1)
        agent_feat = agent_feat * mask

        pooled = agent_feat.sum(dim=1) / (mask.sum(dim=1) + 1e-6)

        # Fuse
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

    dataset = PlannerDataset(files)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = AgentAwarePlanner().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f"Training agent-aware planner on {len(dataset)} samples")

    for epoch in range(EPOCHS):
        total = 0.0
        for ego, agents, mask, gt in loader:
            ego = ego.to(DEVICE)
            agents = agents.to(DEVICE)
            mask = mask.to(DEVICE)
            gt = gt.to(DEVICE)

            pred = model(ego, agents, mask)
            loss = loss_fn(pred, gt)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total/len(loader):.6f}")

    torch.save(model.state_dict(), "agent_aware_planner.pt")
    print("Saved agent_aware_planner.pt")

if __name__ == "__main__":
    main()
