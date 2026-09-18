"""
Problem 2: Distributed State Estimation of a Target Intruder via a UAV Swarm
Algorithm: Distributed Gradient Descent (DGD) with Metropolis-Hastings Weights
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

np.random.seed(42)

# ============================================================
# 1. PROBLEM PARAMETERS
# ============================================================
N = 8                       # Number of UAV drones (N < 20)
p_er = 0.40                 # Edge probability for Erdős-Rényi graph
T_max = 40                  # Time horizon t = 0, ..., T_max
dim = 3                     # Spatial coordinates (x, y, z)

# Target prior: z0 ~ N(z0_bar, Sigma0)
z0_bar = np.array([0.0, 0.0, 0.0])
Sigma0 = np.diag([2.0, 2.0, 0.2])

# Process noise covariance (ship moves along surface with small vertical fluctuation)
Sigma_w = np.diag([0.25, 0.25, 0.01])

# DGD Consensus hyperparameters
K_inner = 150               # Consensus iterations per time step
alpha = 0.015               # Step size

# ============================================================
# 2. GENERATE CONNECTED ERDŐS-RÉNYI GRAPH
# ============================================================
while True:
    G = nx.erdos_renyi_graph(N, p_er, seed=42)
    if nx.is_connected(G):
        break

neighbors = {i: list(G.neighbors(i)) for i in range(N)}
deg = dict(G.degree())

# ============================================================
# 3. METROPOLIS-HASTINGS DOUBLY-STOCHASTIC MATRIX W
# ============================================================
W = np.zeros((N, N))
for i in range(N):
    for j in neighbors[i]:
        W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
    W[i, i] = 1.0 - np.sum(W[i, :])

assert np.allclose(W, W.T), "Weight matrix must be symmetric"
assert np.allclose(W.sum(axis=1), 1.0), "Weight matrix rows must sum to 1"

# ============================================================
# 4. UAV LOCATIONS x_i & SENSOR NOISE COVARIANCES
# ============================================================
# Drones positioned around the surveillance area at altitudes z in [5, 9] m
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
radius = np.random.uniform(8.0, 12.0, size=N)
drone_x = radius * np.cos(angles) + np.random.uniform(-1.0, 1.0, size=N)
drone_y = radius * np.sin(angles) + np.random.uniform(-1.0, 1.0, size=N)
drone_z = np.random.uniform(5.5, 9.0, size=N)
X = np.column_stack([drone_x, drone_y, drone_z])

# Heterogeneous sensor noise variances
sigma_i2 = np.random.uniform(0.3, 1.8, size=N)
Sigma_v_inv = [np.eye(dim) / s2 for s2 in sigma_i2]

# ============================================================
# 5. SIMULATE GROUND-TRUTH INTRUDER TRAJECTORY z(t)
# ============================================================
z_true = np.zeros((T_max + 1, dim))
z_true[0] = np.random.multivariate_normal(z0_bar, Sigma0)
L_w = np.linalg.cholesky(Sigma_w)

for t in range(T_max):
    # Directed drift + random walk process noise
    drift = np.array([0.35, 0.25, 0.0])
    w_t = L_w @ np.random.randn(dim)
    z_true[t + 1] = z_true[t] + drift + w_t

# ============================================================
# 6. GENERATE SENSOR MEASUREMENTS: y_i(t) = x_i - z(t) + v_i
# ============================================================
Y = np.zeros((T_max + 1, N, dim))
for t in range(T_max + 1):
    for i in range(N):
        v_i = np.sqrt(sigma_i2[i]) * np.random.randn(dim)
        Y[t, i] = X[i] - z_true[t] + v_i

# ============================================================
# 7. DISTRIBUTED GRADIENT DESCENT (DGD)
# ============================================================
z_hat = np.zeros((T_max + 1, dim))
z_agents = np.tile(z0_bar, (N, 1))

for t in range(T_max + 1):
    # Unbiased direct reconstruction: target_obs_i = x_i - y_i(t)
    z_obs = X - Y[t]

    if t > 0:
        z_agents = np.tile(z_hat[t - 1], (N, 1))

    for k in range(K_inner):
        grads = np.zeros((N, dim))
        for i in range(N):
            grads[i] = Sigma_v_inv[i] @ (z_agents[i] - z_obs[i])

        # Consensus mixing + local gradient descent step
        z_agents = W @ z_agents - alpha * grads

    z_hat[t] = np.mean(z_agents, axis=0)

# ============================================================
# 8. ERROR ANALYSIS
# ============================================================
errors = z_hat - z_true
error_norm = np.linalg.norm(errors, axis=1)

print("=" * 60)
print(f"Mean Estimation Error Norm: {np.mean(error_norm):.4f} m")
print(f"Final Estimation Error Norm: {error_norm[-1]:.4f} m")
print("=" * 60)

# ============================================================
# 9. EXPORT PLOTS
# ============================================================
# Plot 1: Communication Graph
fig1, ax1 = plt.subplots(figsize=(6, 5))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, ax=ax1, node_color="#2b5c8f", node_size=650)
nx.draw_networkx_edges(G, pos, ax=ax1, edge_color="dimgray", width=1.8)
nx.draw_networkx_labels(G, pos, ax=ax1, font_color="white", font_weight="bold", font_size=11)
ax1.set_title(f"Connected UAV Communication Graph $\\mathcal{{G}}(\\mathcal{{V}}, \\mathcal{{E}})$ ($N = {N}$)", fontsize=11)
ax1.axis("off")
fig1.tight_layout()
fig1.savefig("plot1_communication_graph.png", dpi=200)

# Plot 2: 3D Trajectory
fig2 = plt.figure(figsize=(9, 7))
ax2 = fig2.add_subplot(111, projection="3d")
ax2.scatter(X[:, 0], X[:, 1], X[:, 2], color="black", marker="^", s=90, label=r"UAV Drones ($x_i$)")
for i in range(N):
    ax2.text(X[i, 0], X[i, 1], X[i, 2] + 0.4, f"UAV {i+1}", fontsize=8)

ax2.plot(z_true[:, 0], z_true[:, 1], z_true[:, 2], color="crimson", linewidth=2.5, label=r"True Target Trajectory $z(t)$")
ax2.plot(z_hat[:, 0], z_hat[:, 1], z_hat[:, 2], color="royalblue", linestyle="--", linewidth=2.0, label=r"DGD Estimate $\hat{z}(t)$")
ax2.set_xlabel("X (m)", labelpad=8)
ax2.set_ylabel("Y (m)", labelpad=8)
ax2.set_zlabel("Z (m)", labelpad=8)
ax2.set_title("3D Distributed State Estimation: Target vs. UAV Swarm", fontsize=12)
ax2.legend(loc="upper left")
ax2.view_init(elev=28, azim=-55)
fig2.tight_layout()
fig2.savefig("plot2_trajectory_3d.png", dpi=200)

# Plot 3: Error Norm vs. Time
fig3, ax3 = plt.subplots(figsize=(7.5, 4.2))
ax3.plot(range(T_max + 1), error_norm, color="#d95f02", linewidth=2, marker="o", markersize=3.5, label=r"$\|e(t)\|_2 = \|\hat{z}(t) - z(t)\|_2$")
ax3.axhline(np.mean(error_norm), color="black", linestyle="--", alpha=0.7, label=f"Mean Error = {np.mean(error_norm):.3f} m")
ax3.set_xlabel("Time step $t$", fontsize=10)
ax3.set_ylabel("Estimation Error (m)", fontsize=10)
ax3.set_title(r"Estimation Error Norm $\|e(t)\|_2$ over Time", fontsize=11)
ax3.grid(True, linestyle=":", alpha=0.6)
ax3.legend(loc="upper right")
fig3.tight_layout()
fig3.savefig("plot3_error_norm.png", dpi=200)

plt.close("all")