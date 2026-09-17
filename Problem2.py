"""
Problem 2: Distributed State Estimation of a Target Intruder via a UAV Swarm
=============================================================================

Target dynamics (random walk):
    z(t+1) = z(t) + w,          w ~ N(0, Sigma_w)

Sensor model (drone i, located at x_i):
    y_i(t) = x_i - z(t) + v_i,  v_i ~ N(0, Sigma_v^(i)), independent across i

Distributed estimation: Distributed Gradient Descent (DGD) over a connected
Erdos-Renyi communication network with Metropolis-Hastings weights.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3D projection)

np.random.seed(7)

# ============================================================
# 1. PROBLEM PARAMETERS
# ============================================================
N = 8                      # number of drones (< 20)
p_er = 0.45                # Erdos-Renyi edge probability
T_max = 50                 # time horizon
dim = 3                    # state dimension in R^3

# Target prior: z0 ~ N(z0_bar, Sigma0)
z0_bar = np.array([0.0, 0.0, 0.0])
Sigma0 = 4.0 * np.eye(dim)

# Process noise covariance (random-walk drift)
Sigma_w = 0.05 * np.eye(dim)

# Distributed Gradient Descent parameters
K_inner = 300              # inner consensus iterations per time step
alpha = 0.02               # gradient descent step size

# ============================================================
# 2. GENERATE CONNECTED ERDOS-RENYI COMMUNICATION GRAPH
# ============================================================
while True:
    G = nx.erdos_renyi_graph(N, p_er)
    if nx.is_connected(G):
        break

print("=" * 65)
print("DISTRIBUTED TARGET TRACKING VIA UAV SWARM (DGD CONSENSUS)")
print("=" * 65)
print(f"Number of drones : {N}")
print(f"Number of edges  : {G.number_of_edges()}")
print(f"Graph connected  : {nx.is_connected(G)}")

neighbors = {i: list(G.neighbors(i)) for i in range(N)}

# ============================================================
# 3. METROPOLIS-HASTINGS DOUBLY-STOCHASTIC WEIGHT MATRIX W
# ============================================================
deg = dict(G.degree())
W = np.zeros((N, N))

for i in range(N):
    for j in neighbors[i]:
        W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
    W[i, i] = 1.0 - np.sum(W[i, :])

assert np.allclose(W, W.T), "Weight matrix W must be symmetric"
assert np.allclose(W.sum(axis=1), 1.0), "Weight matrix W rows must sum to 1"

# ============================================================
# 4. DRONE LOCATIONS x_i AND SENSOR NOISE COVARIANCES
# ============================================================
drone_xy = np.random.uniform(-12, 12, size=(N, 2))
drone_z = np.random.uniform(6, 10, size=(N, 1))
X = np.hstack([drone_xy, drone_z])          # fixed drone positions x_i in R^3

# Heterogeneous sensor noise quality
sigma_i2 = np.random.uniform(0.4, 3.0, size=N)
Sigma_v = [s2 * np.eye(dim) for s2 in sigma_i2]
Sigma_v_inv = [np.linalg.inv(S) for S in Sigma_v]

print("\nDrone sensor noise variances (sigma_i^2):")
print(np.round(sigma_i2, 3))

# ============================================================
# 5. SIMULATE GROUND-TRUTH TARGET TRAJECTORY z(t)
# ============================================================
z_true = np.zeros((T_max + 1, dim))
z_true[0] = np.random.multivariate_normal(z0_bar, Sigma0)

L_w = np.linalg.cholesky(Sigma_w)
for t in range(T_max):
    w_t = L_w @ np.random.randn(dim)
    z_true[t + 1] = z_true[t] + w_t

# ============================================================
# 6. GENERATE NOISY SENSOR MEASUREMENTS y_i(t)
# ============================================================
Y = np.zeros((T_max + 1, N, dim))
for t in range(T_max + 1):
    for i in range(N):
        L_v = np.linalg.cholesky(Sigma_v[i])
        v_i = L_v @ np.random.randn(dim)
        Y[t, i] = X[i] - z_true[t] + v_i

# ============================================================
# 7. DISTRIBUTED STATE ESTIMATION (DGD)
# ============================================================
z_hat = np.zeros((T_max + 1, dim))
z_agents = np.tile(z0_bar, (N, 1))

for t in range(T_max + 1):
    # Local direct observation: h_i(t) = x_i - y_i(t)
    h = X - Y[t]

    # Warm start from previous consensus state
    if t > 0:
        z_agents = np.tile(z_hat[t - 1], (N, 1))

    # Inner DGD consensus loop
    for k in range(K_inner):
        grads = np.zeros((N, dim))
        for i in range(N):
            grads[i] = Sigma_v_inv[i] @ (z_agents[i] - h[i])

        z_agents = W @ z_agents - alpha * grads

    z_hat[t] = z_agents.mean(axis=0)

    if t % 10 == 0 or t == T_max:
        disagreement = np.max(np.linalg.norm(z_agents - z_hat[t], axis=1))
        print(f"t={t:3d}  Consensus Disagreement max||z_i - z_hat|| = {disagreement:.5f}")

# ============================================================
# 8. ESTIMATION ERROR ANALYSIS
# ============================================================
error = z_hat - z_true
error_norm = np.linalg.norm(error, axis=1)

print("\n" + "=" * 65)
print(f"Mean Error Norm ||e(t)||_2 over all t : {np.mean(error_norm):.4f} m")
print(f"Final Error Norm ||e(T_max)||_2      : {error_norm[-1]:.4f} m")
print("=" * 65)

# ============================================================
# 9. PLOT 1: COMMUNICATION TOPOLOGY GRAPH
# ============================================================
fig1, ax1 = plt.subplots(figsize=(6, 5))
pos = nx.spring_layout(G, seed=7)
nx.draw(
    G, pos, ax=ax1, with_labels=True,
    node_color="#2b5c8f", node_size=600,
    font_color="white", font_weight="bold",
    edge_color="gray", width=1.5
)
ax1.set_title(f"Connected Communication Graph G(V, E) (N = {N})")
fig1.tight_layout()
fig1.savefig("plot1_communication_graph.png", dpi=150)

# ============================================================
# 10. PLOT 2: 3D SPATIAL TRAJECTORY
# ============================================================
fig2 = plt.figure(figsize=(9, 7))
ax2 = fig2.add_subplot(111, projection="3d")

ax2.scatter(
    X[:, 0], X[:, 1], X[:, 2],
    c="black", marker="^", s=90, label="Drone Positions $x_i$"
)
ax2.plot(
    z_true[:, 0], z_true[:, 1], z_true[:, 2],
    color="crimson", linewidth=2, label="True Intruder Trajectory $z(t)$"
)
ax2.plot(
    z_hat[:, 0], z_hat[:, 1], z_hat[:, 2],
    color="royalblue", linewidth=2, linestyle="--",
    label=r"Consensus Estimate $\hat{z}(t)$"
)

ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_zlabel("Z (m)")
ax2.set_title("3D Distributed Tracking: UAV Swarm vs Intruder")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle=":", alpha=0.5)
fig2.tight_layout()
fig2.savefig("plot2_trajectory_3d.png", dpi=150)

# ============================================================
# 11. PLOT 3: ESTIMATION ERROR NORM VS TIME
# ============================================================
fig3, ax3 = plt.subplots(figsize=(8, 4.5))
ax3.plot(range(T_max + 1), error_norm, color="#d95f02", linewidth=2, marker="o", markersize=3)
ax3.axhline(np.mean(error_norm), color="black", linestyle="--", alpha=0.7, 
            label=f"Mean Error = {np.mean(error_norm):.3f} m")
ax3.set_xlabel("Time Step $t$")
ax3.set_ylabel(r"$\|e(t)\|_2 = \|\hat{z}(t) - z(t)\|_2$ (m)")
ax3.set_title("Distributed State Estimation Error Norm vs. Time")
ax3.grid(True, linestyle=":", alpha=0.6)
ax3.legend()
fig3.tight_layout()
fig3.savefig("plot3_error_norm.png", dpi=150)

print("\nPlots successfully saved as:")
print("  - plot1_communication_graph.png")
print("  - plot2_trajectory_3d.png")
print("  - plot3_error_norm.png")

plt.show()