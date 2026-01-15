import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

ego_all = d["ego_past"]        # (H,5) at first step
ego_future = d["ego_future"]
agents_all = d["agents_past"]
mask_all = d["agents_mask"]

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
model.train()  # IMPORTANT: keep dropout ON

# =====================
# MC DROPOUT PREDICTION
# =====================
def predict_with_uncertainty(ego, agents, mask):
    preds = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            preds.append(
                model(ego, agents, mask)[0].cpu().numpy()
            )
    preds = np.stack(preds, axis=0)   # (K,T,5)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std

# =====================
# PLOTTING
# =====================
fig, ax = plt.subplots(figsize=(6,6))

def update(frame):
    ax.clear()

    ego = torch.from_numpy(ego_all).float().unsqueeze(0).to(DEVICE)
    agents = torch.from_numpy(agents_all).float().unsqueeze(0).to(DEVICE)
    mask = torch.from_numpy(mask_all).float().unsqueeze(0).to(DEVICE)

    mean, std = predict_with_uncertainty(ego, agents, mask)

    # Ego past
    ax.plot(ego_all[:,0], ego_all[:,1], "bo-", label="ego past")

    # GT future
    ax.plot(ego_future[:,0], ego_future[:,1], "go-", label="GT future")

    # Mean prediction
    ax.plot(mean[:,0], mean[:,1], "ro--", label="mean prediction")

    # Uncertainty band
    ax.fill_between(
        mean[:,0],
        mean[:,1] - std[:,1],
        mean[:,1] + std[:,1],
        color="red",
        alpha=0.2,
        label="uncertainty"
    )

    # Agents
    for i in range(len(mask_all)):
        if mask_all[i] > 0:
            ax.scatter(
                agents_all[i,0,0],
                agents_all[i,0,1],
                c="gray",
                s=20
            )

    ax.axhline(0,c="k")
    ax.axvline(0,c="k")
    ax.axis("equal")
    ax.set_title(f"Planner rollout (frame {frame})")
    ax.legend(loc="upper left")

update(0)  # force-render one frame
plt.savefig("uncertainty_rollout.png", dpi=150)
print("Saved uncertainty_rollout.png")



