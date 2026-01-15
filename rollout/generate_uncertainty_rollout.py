import numpy as np
import torch

# =====================
# CONFIG
# =====================
SAMPLE_PATH = "nuplan_ml_dataset/samples/sample_000200.npz"
MODEL_PATH = "uncertain_planner.pt"

H = 10
T = 20
N = 32
MC_SAMPLES = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# LOAD DATA
# =====================
d = np.load(SAMPLE_PATH)

ego_past = d["ego_past"]
ego_future = d["ego_future"]
agents = d["agents_past"]
mask = d["agents_mask"]

# =====================
# MODEL
# =====================
class UncertainPlanner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ego_enc = torch.nn.Sequential(
            torch.nn.Linear(H * 5, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 256),
        )
        self.agent_enc = torch.nn.Sequential(
            torch.nn.Linear(H * 5, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 128),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(256 + 128, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, T * 5),
        )

    def forward(self, ego, agents, mask):
        B = ego.size(0)
        ego_feat = self.ego_enc(ego.view(B, -1))
        agents_flat = agents.view(B, N, -1)
        agent_feat = self.agent_enc(agents_flat)
        mask = mask.unsqueeze(-1)
        pooled = (agent_feat * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        fused = torch.cat([ego_feat, pooled], dim=1)
        return self.decoder(fused).view(B, T, 5)

model = UncertainPlanner().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.train()  # keep dropout ON

# =====================
# MC-DROPOUT
# =====================
def predict(ego, agents, mask):
    preds = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            preds.append(
                model(ego, agents, mask)[0].cpu().numpy()
            )
    preds = np.stack(preds)
    return preds.mean(axis=0), preds.std(axis=0)

# =====================
# GENERATE ROLLOUT
# =====================
ego_t = torch.from_numpy(ego_past).float().unsqueeze(0).to(DEVICE)
agents_t = torch.from_numpy(agents).float().unsqueeze(0).to(DEVICE)
mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(DEVICE)

mean, std = predict(ego_t, agents_t, mask_t)

np.savez(
    "uncertainty_rollout_data.npz",
    ego_past=ego_past,
    ego_future=ego_future,
    agents=agents,
    mask=mask,
    mean=mean,
    std=std,
)

print("Saved uncertainty_rollout_data.npz")
