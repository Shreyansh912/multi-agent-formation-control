"""
Problem 3, Part 2: Distributed Task Allocation via ADMM over a Communication Graph
====================================================================================

N agents, M > N tasks. Each agent i has capacity b_i (sum b_i = M) and a
local cost row C_i for performing each task. We solve the (relaxed) task
allocation LP:

    min_{X in R^{NxM}}  sum_i sum_j C_ij * x_ij
    s.t.  sum_i x_ij = 1        for all j   (each task done exactly once)
          sum_j x_ij <= b_i     for all i   (agent capacity)
          0 <= x_ij <= 1

in a fully distributed way: agents exchange information with graph neighbors
via exchange-ADMM + average consensus (Metropolis-Hastings weights).
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import linprog

np.random.seed(3)


# ============================================================
# 1. CHOOSE N (agents) AND M (tasks), M > N
# ============================================================
N = 6
M = 20
assert M > N, "Task count M must exceed agent count N"

# ============================================================
# 2. CONNECTED ERDOS-RENYI COMMUNICATION GRAPH
# ============================================================
p_er = 0.5
while True:
    G = nx.erdos_renyi_graph(N, p_er)
    if nx.is_connected(G):
        break

neighbors = {i: list(G.neighbors(i)) for i in range(N)}
deg = dict(G.degree())

# Metropolis-Hastings doubly-stochastic weight matrix
W = np.zeros((N, N))
for i in range(N):
    for j in neighbors[i]:
        W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
    W[i, i] = 1.0 - np.sum(W[i, :])

print("=" * 65)
print("DISTRIBUTED TASK ALLOCATION VIA ADMM + GRAPH CONSENSUS")
print("=" * 65)
print(f"N (agents) = {N},  M (tasks) = {M}")
print(f"Graph edges: {G.number_of_edges()} | connected: {nx.is_connected(G)}")


# ============================================================
# 3. RANDOM AGENT CAPACITIES b_i, sum_i b_i = M
# ============================================================
cuts = np.sort(np.random.choice(range(1, M), N - 1, replace=False))
b = np.diff(np.concatenate(([0], cuts, [M])))
b = b.astype(int)
assert b.sum() == M and np.all(b >= 1), "Capacities must be positive integers summing to M"

print(f"Agent capacities b_i: {b} (sum = {b.sum()})")


# ============================================================
# 4. PROJECTION ONTO X_i = { x in [0,1]^M : sum(x) <= b_i }
# ============================================================
def project_capped_simplex(v, cap):
    """Euclidean projection onto box-capped simplex via bisection."""
    x_box = np.clip(v, 0.0, 1.0)
    if x_box.sum() <= cap + 1e-12:
        return x_box

    lo, hi = np.min(v) - 1.0, np.max(v)
    for _ in range(100):
        tau = 0.5 * (lo + hi)
        x = np.clip(v - tau, 0.0, 1.0)
        if x.sum() > cap:
            lo = tau
        else:
            hi = tau
    return np.clip(v - hi, 0.0, 1.0)


def run_admm(C, rho=1.0, K_admm=300, K_consensus=60, verbose_tag=""):
    """Distributed exchange-ADMM for task allocation LP with graph consensus."""
    a_over_N = (1.0 / N) * np.ones(M)

    x = np.zeros((N, M))
    xbar = np.tile(a_over_N.copy(), (N, 1))
    u = np.zeros((N, M))

    cost_hist = []
    resid_hist = []

    for k in range(K_admm):
        x_new = np.zeros((N, M))
        for i in range(N):
            v_i = x[i] - xbar[i] + a_over_N - u[i] - (1.0 / rho) * C[i]
            x_new[i] = project_capped_simplex(v_i, b[i])

        # Distributed average consensus over the graph
        z = x_new.copy()
        for _ in range(K_consensus):
            z = W @ z
        xbar_new = z

        # Dual update tracked locally per agent
        u_new = u + xbar_new - a_over_N

        x, xbar, u = x_new, xbar_new, u_new

        cost_hist.append(np.sum(C * x))
        resid_hist.append(np.linalg.norm(x.sum(axis=0) - 1.0))

    print(f"[{verbose_tag}] final constraint residual ||sum_i x_i - 1||: {resid_hist[-1]:.6f}")
    print(f"[{verbose_tag}] final relaxed cost: {cost_hist[-1]:.4f}")

    return x, cost_hist, resid_hist


def round_allocation(x, b_caps):
    """Greedy rounding into a binary assignment respecting capacities b_i."""
    N_, M_ = x.shape
    remaining = b_caps.copy()
    assign = -np.ones(M_, dtype=int)

    task_order = np.argsort(-x.max(axis=0))
    for j in task_order:
        for i in np.argsort(-x[:, j]):
            if remaining[i] > 0:
                assign[j] = i
                remaining[i] -= 1
                break

    X_bin = np.zeros((N_, M_))
    for j, i in enumerate(assign):
        X_bin[i, j] = 1
    return X_bin, assign


def centralized_lp_reference(C):
    """Centralized LP relaxation baseline using scipy for validation."""
    N_, M_ = C.shape
    c_vec = C.flatten()

    A_eq = np.zeros((M_, N_ * M_))
    for j in range(M_):
        for i in range(N_):
            A_eq[j, i * M_ + j] = 1.0
    b_eq = np.ones(M_)

    A_ub = np.zeros((N_, N_ * M_))
    for i in range(N_):
        A_ub[i, i * M_:(i + 1) * M_] = 1.0
    b_ub = b.astype(float)

    res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, 1), method="highs")
    X_ref = res.x.reshape(N_, M_)
    return X_ref, res.fun


# ============================================================
# 5. EXPERIMENT 1: RANDOM COST MATRIX C1
# ============================================================
C1 = np.random.uniform(1.0, 10.0, size=(N, M))
assert np.all(C1 > 0), "All cost entries must be strictly positive"

x1, cost_hist1, resid_hist1 = run_admm(C1, verbose_tag="C1")
X1_bin, assign1 = round_allocation(x1, b)
cost1_binary = np.sum(C1 * X1_bin)
X1_ref, cost1_ref = centralized_lp_reference(C1)

print(f"[C1] Centralized LP optimal cost   : {cost1_ref:.4f}")
print(f"[C1] ADMM relaxed cost (converged): {cost_hist1[-1]:.4f}")
print(f"[C1] Rounded binary cost          : {cost1_binary:.4f}")


# ============================================================
# 6. EXPERIMENT 2: DIFFERENT RANDOM COST MATRIX C2
# ============================================================
C2 = np.random.uniform(1.0, 10.0, size=(N, M))
assert np.all(C2 > 0), "All cost entries must be strictly positive"

x2, cost_hist2, resid_hist2 = run_admm(C2, verbose_tag="C2")
X2_bin, assign2 = round_allocation(x2, b)
cost2_binary = np.sum(C2 * X2_bin)
X2_ref, cost2_ref = centralized_lp_reference(C2)

print(f"[C2] Centralized LP optimal cost   : {cost2_ref:.4f}")
print(f"[C2] ADMM relaxed cost (converged): {cost_hist2[-1]:.4f}")
print(f"[C2] Rounded binary cost          : {cost2_binary:.4f}")

frac_changed = np.mean(assign1 != assign2)
print(f"\nFraction of tasks reassigned under C2: {frac_changed:.2%}")


# ============================================================
# 7. VISUALIZATIONS
# ============================================================
# Plot 1: Communication Graph
fig1, ax1 = plt.subplots(figsize=(5.5, 5.5))
pos = nx.spring_layout(G, seed=3)
nx.draw(
    G, pos, ax=ax1, with_labels=True, node_color="#55A868",
    node_size=700, font_color="white", font_weight="bold",
    edge_color="gray", width=1.5
)
ax1.set_title(f"Communication Graph G(V, E) (N={N} agents, connected)")
fig1.tight_layout()
fig1.savefig("p3_plot1_graph.png", dpi=150)

# Plot 2: Convergence Curves
fig2, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.5))
axA.plot(cost_hist1, label="Cost matrix C1", color="tab:blue")
axA.plot(cost_hist2, label="Cost matrix C2", color="tab:red")
axA.axhline(cost1_ref, color="tab:blue", linestyle=":", alpha=0.6, label="C1 Centralized Opt")
axA.axhline(cost2_ref, color="tab:red", linestyle=":", alpha=0.6, label="C2 Centralized Opt")
axA.set_xlabel("ADMM Iteration $k$")
axA.set_ylabel(r"Total Relaxed Cost $\sum_{i,j} C_{ij} x_{ij}$")
axA.set_title("ADMM Objective Cost Convergence")
axA.legend(fontsize=8)
axA.grid(True, linestyle=":", alpha=0.5)

axB.plot(resid_hist1, label="Cost matrix C1", color="tab:blue")
axB.plot(resid_hist2, label="Cost matrix C2", color="tab:red")
axB.set_xlabel("ADMM Iteration $k$")
axB.set_ylabel(r"Residual $\|\sum_i x_i - \mathbf{1}\|_2$")
axB.set_title("Constraint Feasibility Residual")
axB.set_yscale("log")
axB.legend(fontsize=8)
axB.grid(True, linestyle=":", alpha=0.5)

fig2.tight_layout()
fig2.savefig("p3_plot2_convergence.png", dpi=150)

# Plot 3: Allocation Heatmaps
fig3, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, X_bin, title in zip(axes, [X1_bin, X2_bin], ["Allocation for C1", "Allocation for C2"]):
    im = ax.imshow(X_bin, cmap="Greens", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("Task Index $j$")
    ax.set_ylabel("Agent Index $i$")
    ax.set_title(title)
    ax.set_yticks(range(N))
    ax.set_yticklabels([f"Agent {i} ($b_i={b[i]}$)" for i in range(N)])

fig3.colorbar(im, ax=axes, shrink=0.7, label="Assigned (1) / Unassigned (0)")
fig3.tight_layout()
fig3.savefig("p3_plot3_allocation_heatmaps.png", dpi=150)

print("\nFiles generated successfully:")
print("  - p3_plot1_graph.png")
print("  - p3_plot2_convergence.png")
print("  - p3_plot3_allocation_heatmaps.png")

plt.show()