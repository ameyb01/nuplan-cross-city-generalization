import numpy as np
import torch
import matplotlib.pyplot as plt

# =====================
# CONFIG
# =====================
SAMPLE_PATH = "nuplan_ml_dataset/samples/sample_000200.npz"
MODEL_PATH = "baseline_planner.pt"

H = 10
T = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# LOAD SAMPLE
# =====================
d = np.load(SAMPLE_PATH)
ego_past = torch.from_numpy(d["ego_past"]).float().unsqueeze(0).to(DEVICE)
ego_future = d["ego_future"]

agents = d["agents_past"]
mask = d["agents_mask"]

# =====================
# MODEL
# =====================
class MLPPlanner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(H * 5, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, T * 5),
        )

    def forward(self, ego_past):
        x = ego_past.view(ego_past.size(0), -1)
        out = self.net(x)
        return out.view(-1, T, 5)

model = MLPPlanner().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =====================
# PREDICT
# =====================
with torch.no_grad():
    pred_future = model(ego_past)[0].cpu().numpy()

# =====================
# PLOT
# =====================
plt.figure(figsize=(6, 6))

# Ego past
plt.plot(
    d["ego_past"][:, 0],
    d["ego_past"][:, 1],
    "bo-",
    label="ego past"
)

# Ground truth future
plt.plot(
    ego_future[:, 0],
    ego_future[:, 1],
    "go-",
    label="GT future"
)

# Predicted future
plt.plot(
    pred_future[:, 0],
    pred_future[:, 1],
    "ro--",
    label="Predicted future"
)

# Agents (current positions)
for i in range(len(mask)):
    if mask[i] > 0:
        plt.scatter(
            agents[i, 0, 0],
            agents[i, 0, 1],
            c="gray",
            s=15
        )

plt.axhline(0, c="k")
plt.axvline(0, c="k")
plt.axis("equal")
plt.legend()
plt.title("Baseline planner prediction vs GT")
plt.savefig("baseline_prediction.png", dpi=150)
print("Saved plot to baseline_prediction.png")

