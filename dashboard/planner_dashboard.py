import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

d = np.load("uncertainty_rollout_data.npz")

ego_past = d["ego_past"]
ego_future = d["ego_future"]
agents = d["agents"]
mask = d["mask"]
mean = d["mean"]
std = d["std"]

t = st.slider("Future timestep", 0, len(mean) - 1, 0)

fig, ax = plt.subplots(figsize=(6,6))

# Ego past
ax.plot(ego_past[:,0], ego_past[:,1], "bo-", label="ego past")

# GT future (up to t)
ax.plot(ego_future[:t+1,0], ego_future[:t+1,1], "go-", label="GT future")

# Mean prediction
ax.plot(mean[:t+1,0], mean[:t+1,1], "ro--", label="mean pred")

# Uncertainty band
ax.fill_between(
    mean[:t+1,0],
    mean[:t+1,1] - std[:t+1,1],
    mean[:t+1,1] + std[:t+1,1],
    color="red",
    alpha=0.25,
)

# Agents
for i in range(len(mask)):
    if mask[i] > 0:
        ax.scatter(agents[i,0,0], agents[i,0,1], c="gray", s=20)

ax.axhline(0,c="k")
ax.axvline(0,c="k")
ax.axis("equal")
ax.legend()

st.pyplot(fig)
