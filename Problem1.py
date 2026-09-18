"""
Problem 1: Multi-Agent Formation Control via Distributed Consensus
Spelling: S -> H -> R -> E -> Y -> A -> N -> S -> H
Number of agents: N = 20
Graph: Connected Erdős-Rényi Graph
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.optimize import linear_sum_assignment

# ============================================================
# 1. PARAMETERS
# ============================================================
N = 20
p = 0.25
dt = 0.04
gain = 2.0
centroid_gain = 0.5
steps_per_letter = 220

np.random.seed(42)

# ============================================================
# 2. GENERATE CONNECTED ERDŐS-RÉNYI GRAPH
# ============================================================
while True:
    G = nx.erdos_renyi_graph(N, p, seed=42)
    if nx.is_connected(G):
        break

neighbors = {i: list(G.neighbors(i)) for i in range(N)}

# ============================================================
# 3. INITIAL POSITIONS
# ============================================================
positions = np.random.uniform(-6.0, 6.0, size=(N, 2))

# ============================================================
# 4. LETTER FORMATIONS (EXACTLY 20 DISTINCT, CLEAN POINTS)
# ============================================================
letters = {}

# S: Clean digital 5-segment serif shape (exactly 20 points, aligned on grid)
letters["S"] = np.array([
    # 1. Top bar: left to right (y = 3.2)
    [-2.2,  3.2],
    [-0.8,  3.2],
    [ 0.8,  3.2],
    [ 2.2,  3.2],

    # 2. Upper-left vertical spine (x = -2.2)
    [-2.2,  2.4],
    [-2.2,  1.6],
    [-2.2,  0.8],

    # 3. Middle horizontal bar: left to right (y = 0.0)
    [-2.2,  0.0],
    [-0.8,  0.0],
    [ 0.8,  0.0],
    [ 2.2,  0.0],

    # 4. Lower-right vertical spine (x = 2.2)
    [ 2.2, -0.8],
    [ 2.2, -1.6],
    [ 2.2, -2.4],

    # 5. Bottom horizontal bar: right to left (y = -3.2)
    [ 2.2, -3.2],
    [ 1.3, -3.2],
    [ 0.4, -3.2],
    [-0.5, -3.2],
    [-1.4, -3.2],
    [-2.2, -3.2]
], dtype=float)

# H: Left vertical (7), Right vertical (7), Middle crossbar (6)
letters["H"] = np.array([
    # Left vertical column
    [-2.5, 3.0], 
    [-2.5, 2.0], 
    [-2.5, 1.0], 
    [-2.5, 0.0], 
    [-2.5, -1.0], 
    [-2.5, -2.0], 
    [-2.5, -3.0],
    # Right vertical column
    [ 2.5, 3.0], 
    [ 2.5, 2.0], 
    [ 2.5, 1.0], 
    [ 2.5, 0.0], 
    [ 2.5, -1.0], 
    [ 2.5, -2.0], 
    [ 2.5, -3.0],
    # Evenly spaced middle crossbar
    [-1.8, 0.0], 
    [-1.1, 0.0], 
    [-0.4, 0.0], 
    [ 0.4, 0.0], 
    [ 1.1, 0.0], 
    [ 1.8, 0.0]
], dtype=float)

# R: Left spine (7), Top bar (3), Upper loop (5), Diagonal leg (5) -> 20 points
letters["R"] = np.array([
    # 1. Full left vertical spine (x = -2.2)
    [-2.2,  3.0],
    [-2.2,  2.0],
    [-2.2,  1.0],
    [-2.2,  0.0],
    [-2.2, -1.0],
    [-2.2, -2.0],
    [-2.2, -3.0],

    # 2. Top horizontal bar: moving right (y = 3.0)
    [-1.1,  3.0],
    [ 0.0,  3.0],
    [ 1.1,  3.0],

    # 3. Outer curve of the upper loop & closure at y = 0.0
    [ 2.2,  2.4],
    [ 2.2,  1.4],
    [ 2.2,  0.4],
    [ 1.1,  0.0],
    [ 0.0,  0.0],

    # 4. Straight diagonal leg branching down-right
    [ 0.4, -0.6],
    [ 0.9, -1.2],
    [ 1.4, -1.8],
    [ 1.9, -2.4],
    [ 2.4, -3.0]
], dtype=float)

# E: Vertical spine (7), Top bar (4), Middle bar (3), Bottom bar (6) -> 20 points
letters["E"] = np.array([
    # 1. Left vertical spine (x = -2.2)
    [-2.2,  3.0],
    [-2.2,  2.0],
    [-2.2,  1.0],
    [-2.2,  0.0],
    [-2.2, -1.0],
    [-2.2, -2.0],
    [-2.2, -3.0],

    # 2. Top horizontal bar (y = 3.0)
    [-1.1,  3.0],
    [ 0.0,  3.0],
    [ 1.1,  3.0],
    [ 2.2,  3.0],

    # 3. Middle horizontal bar (y = 0.0, slightly shorter)
    [-1.0,  0.0],
    [ 0.1,  0.0],
    [ 1.2,  0.0],

    # 4. Bottom horizontal bar (y = -3.0)
    [-1.3, -3.0],
    [-0.4, -3.0],
    [ 0.5, -3.0],
    [ 1.4, -3.0],
    [ 2.2, -3.0],
    [ 0.0, -3.0]
], dtype=float)

# Y: Left branch (7), Right branch (7), Center stem (6)
letters["Y"] = np.array([
    # Left diagonal branch
    [-2.6, 3.0], 
    [-2.1, 2.4],
    [-1.6, 1.8], 
    [-1.1, 1.2], 
    [-0.6, 0.6], 
    [-0.2, 0.1], 
    [-1.3, 1.5],
    # Right diagonal branch
    [ 2.6, 3.0], 
    [ 2.1, 2.4], 
    [ 1.6, 1.8], 
    [ 1.1, 1.2], 
    [ 0.6, 0.6], 
    [ 0.2, 0.1], 
    [ 1.3, 1.5],
    # Vertical tail
    [0.0, -0.4], 
    [0.0, -0.9], 
    [0.0, -1.4], 
    [0.0, -2.0], 
    [0.0, -2.6], 
    [0.0, -3.2]
], dtype=float)

# A: Left slant (7), Right slant (7), Crossbar (5), Apex peak (1)
letters["A"] = np.array([
    # Left leg
    [-2.5, -3.0], 
    [-2.1, -2.0], 
    [-1.7, -1.0], 
    [-1.3, 0.0], 
    [-0.9, 1.0], 
    [-0.5, 2.0],
    # Peak
    [ 0.0, 3.2],
    # Right leg
    [ 0.5, 2.0], 
    [ 0.9, 1.0], 
    [ 1.3, 0.0], 
    [ 1.7, -1.0], 
    [ 2.1, -2.0], 
    [ 2.5, -3.0],
    # Horizontal crossbar
    [-1.5, -0.6], 
    [-0.75, -0.6], 
    [ 0.0, -0.6], 
    [ 0.75, -0.6], 
    [ 1.5, -0.6],
    # Support anchors on legs
    [-1.9, -1.5], 
    [ 1.9, -1.5]
], dtype=float)

# N: Left vertical (7), Right vertical (7), Diagonal stroke (6)
letters["N"] = np.array([
    # Left vertical column
    [-2.4, -3.0], 
    [-2.4, -2.0], 
    [-2.4, -1.0], 
    [-2.4, 0.0], 
    [-2.4, 1.0], 
    [-2.4, 2.0], 
    [-2.4, 3.0],
    # Right vertical column
    [ 2.4, -3.0], 
    [ 2.4, -2.0], 
    [ 2.4, -1.0], 
    [ 2.4, 0.0], 
    [ 2.4, 1.0], 
    [ 2.4, 2.0], 
    [ 2.4, 3.0],
    # Clean single diagonal line
    [-1.6, 2.0], 
    [-0.9, 1.0], 
    [-0.2, 0.1], 
    [ 0.5, -0.9], 
    [ 1.2, -1.8], 
    [ 1.8, -2.6]
], dtype=float)

# Center all letter patterns at the origin
for k in letters:
    letters[k] -= np.mean(letters[k], axis=0)
# ============================================================
# 5. SIMULATION SEQUENCE
# ============================================================
sequence = ["S", "H", "R", "E", "Y", "A", "N", "S", "H"]
current_positions = positions.copy()
trajectory = [current_positions.copy()]
target_trajectory = []

for letter in sequence:
    desired_targets = letters[letter]
    
    # Optimal assignment between current agent positions and target slots
    cost_matrix = np.linalg.norm(current_positions[:, None, :] - desired_targets[None, :, :], axis=2)
    _, col_ind = linear_sum_assignment(cost_matrix)
    d = desired_targets[col_ind]

    for step in range(steps_per_letter):
        target_trajectory.append(d.copy())
        u = np.zeros_like(current_positions)
        
        for i in range(N):
            # Correct relative-displacement consensus (negative feedback)
            for j in neighbors[i]:
                u[i] += (current_positions[j] - current_positions[i]) - (d[j] - d[i])
            
            # Centroid grounding term to prevent boundary drift
            u[i] -= centroid_gain * (current_positions[i] - d[i])

        current_positions += dt * gain * u
        trajectory.append(current_positions.copy())

trajectory = np.array(trajectory)
target_trajectory = np.array(target_trajectory)

# ============================================================
# 6. ANIMATION & EXPORT
# ============================================================
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-6.5, 6.5)
ax.set_ylim(-6.5, 6.5)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.5)

agents_scatter = ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], s=180, color="#1f77b4", zorder=4)
edge_lines = [ax.plot([], [], color="gray", lw=0.8, alpha=0.3)[0] for _ in G.edges()]
title = ax.set_title("", fontsize=12, fontweight="bold")

def update(frame):
    current = trajectory[frame]
    agents_scatter.set_offsets(current)
    
    for line, (i, j) in zip(edge_lines, G.edges()):
        line.set_data([current[i, 0], current[j, 0]], [current[i, 1], current[j, 1]])
        
    letter_idx = min(frame // steps_per_letter, len(sequence) - 1)
    title.set_text(f"UAV Formation Control (N={N}) | Name: SHREYANSH | Letter: {sequence[letter_idx]}")
    return [agents_scatter, title] + edge_lines

anim = FuncAnimation(fig, update, frames=len(trajectory), interval=25, blit=False)

os.makedirs("output", exist_ok=True)
output_path = os.path.join("output", "SHREYANSH_formation_animation.mp4")
anim.save(output_path, writer=FFMpegWriter(fps=30))
print(f"Animation successfully exported to {output_path}")