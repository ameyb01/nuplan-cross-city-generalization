import numpy as np
import torch
import matplotlib.pyplot as plt

# =====================
# CONFIG
# =====================
SAMPLE_PATH = "nuplan_ml_dataset/samples/sample_000200.npz"
MODEL_PATH = "agent_aware_planner.pt"

H = 10
T = 20
N = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# LOAD SAMPLE
# =====================
d = np.load(SAMPLE_PATH)

ego_past = torch.from_numpy(d["ego_past"]).float().unsqueeze(0).to(DEVICE)
agents = torch.from_numpy(d["agents_past"]).float().unsqueeze(0).to(DEVICE)
mask = torch.from_numpy(d["agents_mask"]).float().unsqueeze(0).to(DEVICE)

ego_future = d["ego_future"]

# =====================
# MODEL
# =====================
class AgentAwarePlanner(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.ego_enc = torch.nn.Sequential(
            torch.nn.Linear(H * 5, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
        )

        self.agent_enc = torch.nn.Sequential(
            torch.nn.Linear(H * 5, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(256 + 128, 256),
            torch.nn.ReLU(),
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
        out = self.decoder(fused)

        return out.view(B, T, 5)

model = AgentAwarePlanner().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =====================
# PREDICT
# =====================
with torch.no_grad():
    pred = model(ego_past, agents, mask)[0].cpu().numpy()

# =====================
# PLOT
# =====================
plt.figure(figsize=(6,6))

# Ego past
plt.plot(d["ego_past"][:,0], d["ego_past"][:,1], "bo-", label="ego past")

# GT future
plt.plot(ego_future[:,0], ego_future[:,1], "go-", label="GT future")

# Predicted future
plt.plot(pred[:,0], pred[:,1], "ro--", label="Agent-aware pred")

# Agents (current positions)
for i in range(len(d["agents_mask"])):
    if d["agents_mask"][i] > 0:
        plt.scatter(d["agents_past"][i,0,0], d["agents_past"][i,0,1],
                    c="gray", s=15)

plt.axhline(0,c="k")
plt.axvline(0,c="k")
plt.axis("equal")
plt.legend()
plt.title("Agent-aware planner vs GT")
plt.savefig("agent_aware_prediction.png", dpi=150)
print("Saved plot to agent_aware_prediction.png")
