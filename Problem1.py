import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ============================================================
# 1. PARAMETERS & GRAPH GENERATION
# ============================================================
N = 20
p = 0.20
dt = 0.05
steps_per_letter = 250
gain = 1.0
np.random.seed(42)

while True:
    G = nx.erdos_renyi_graph(N, p)
    if nx.is_connected(G):
        break

positions = np.random.uniform(-8, 8, size=(N, 2))
initial_positions = positions.copy()

# ============================================================
# 2. EXACT NON-OVERLAPPING 20-POINT LETTER PATTERNS
# ============================================================
letters = {}

letters["S"] = np.array([
    [-2.0,  3.0], [-1.0,  3.0], [ 0.0,  3.0], [ 1.0,  3.0], [ 2.0,  3.0],
    [-2.0,  2.0], [-2.0,  1.0], [-1.0,  1.0], [ 0.0,  1.0], [ 1.0,  1.0],
    [ 2.0,  0.0], [ 2.0, -1.0], [ 1.0, -1.0], [ 0.0, -1.0], [-1.0, -1.0],
    [-2.0, -1.0], [-2.0, -2.0], [-2.0, -3.0], [-1.0, -3.0], [ 0.0, -3.0]
], dtype=float)

letters["H"] = np.array([
    [-2.0,  3.0], [-2.0,  2.0], [-2.0,  1.0], [-2.0,  0.0], [-2.0, -1.0], [-2.0, -2.0], [-2.0, -3.0],
    [ 2.0,  3.0], [ 2.0,  2.0], [ 2.0,  1.0], [ 2.0,  0.0], [ 2.0, -1.0], [ 2.0, -2.0], [ 2.0, -3.0],
    [-1.4,  0.0], [-0.8,  0.0], [-0.2,  0.0], [ 0.4,  0.0], [ 1.0,  0.0], [ 1.5,  0.0]
], dtype=float)

# FIXED LETTER R: Clean loop and isolated diagonal leg (no 13/20 overlap)
letters["R"] = np.array([
    [-2.0,  3.0], [-2.0,  2.0], [-2.0,  1.0], [-2.0,  0.0], [-2.0, -1.0], [-2.0, -2.0], [-2.0, -3.0], # Spine (1-7)
    [-1.0,  3.0], [ 0.0,  3.0], [ 1.2,  2.6], [ 1.8,  1.8], [ 1.2,  0.8], [ 0.1,  0.3],               # Top loop (8-13)
    [-0.8, -0.2], [-0.3, -0.8], [ 0.2, -1.4], [ 0.7, -2.0], [ 1.2, -2.6], [ 1.7, -3.2], [ 2.2, -3.8]  # Leg (14-20)
], dtype=float)

letters["E"] = np.array([
    [-2.0,  3.0], [-2.0,  2.0], [-2.0,  1.0], [-2.0,  0.0], [-2.0, -1.0], [-2.0, -2.0], [-2.0, -3.0],
    [-1.1,  3.0], [-0.2,  3.0], [ 0.7,  3.0], [ 1.6,  3.0],
    [-1.1,  0.0], [-0.2,  0.0], [ 0.7,  0.0], [ 1.5,  0.0],
    [-1.1, -3.0], [-0.3, -3.0], [ 0.4, -3.0], [ 1.1, -3.0], [ 1.8, -3.0]
], dtype=float)

letters["Y"] = np.array([
    [-2.5,  3.0], [-2.0,  2.4], [-1.5,  1.8], [-1.0,  1.2], [-0.6,  0.6], [-0.2,  0.2],
    [ 2.5,  3.0], [ 2.0,  2.4], [ 1.5,  1.8], [ 1.0,  1.2], [ 0.6,  0.6], [ 0.2,  0.2],
    [ 0.0, -0.2], [ 0.0, -0.6], [ 0.0, -1.0], [ 0.0, -1.5], [ 0.0, -2.0], [ 0.0, -2.4], [ 0.0, -2.8], [ 0.0, -3.2]
], dtype=float)

# FIXED LETTER A: Shortened inner crossbar to prevent touching outer legs
letters["A"] = np.array([
    [ 0.0,  3.5],                                                                            # Apex (1)
    [-0.4,  2.4], [-0.8,  1.3], [-1.2,  0.2], [-1.6, -0.9], [-2.0, -2.0], [-2.4, -3.1],      # Left leg (2-7)
    [ 0.4,  2.4], [ 0.8,  1.3], [ 1.2,  0.2], [ 1.6, -0.9], [ 2.0, -2.0], [ 2.4, -3.1],      # Right leg (8-13)
    [-0.7,  0.0], [-0.35, 0.0], [ 0.0,  0.0], [ 0.35, 0.0], [ 0.7,  0.0],                    # Crossbar (14-18)
    [-1.2, -3.1], [ 1.2, -3.1]                                                               # Base stabilizers (19-20)
], dtype=float)

letters["N"] = np.array([
    [-2.0, -3.0], [-2.0, -2.0], [-2.0, -1.0], [-2.0,  0.0], [-2.0,  1.0], [-2.0,  2.0], [-2.0,  3.0],
    [-1.3,  1.8], [-0.7,  0.8], [ 0.0, -0.1], [ 0.6, -1.0], [ 1.2, -1.9], [ 1.7, -2.6],
    [ 2.0, -3.0], [ 2.0, -2.0], [ 2.0, -1.0], [ 2.0,  0.0], [ 2.0,  1.0], [ 2.0,  2.0], [ 2.0,  3.0]
], dtype=float)

for key in letters:
    letters[key] -= np.mean(letters[key], axis=0)

# ============================================================
# 3. CONTROL SIMULATION
# ============================================================
sequence = ["S", "H", "R", "E", "Y", "A", "N", "S", "H"]
neighbors = {i: list(G.neighbors(i)) for i in range(N)}

all_trajectory = [positions.copy()]
all_targets = []
current_positions = positions.copy()

for letter in sequence:
    desired_relative = letters[letter]
    current_center = np.mean(current_positions, axis=0)
    desired_positions = desired_relative + current_center

    for step in range(steps_per_letter):
        all_targets.append(desired_positions.copy())
        new_positions = current_positions.copy()

        for i in range(N):
            u_i = np.zeros(2)
            for j in neighbors[i]:
                actual_rel = current_positions[j] - current_positions[i]
                desired_rel = desired_relative[j] - desired_relative[i]
                u_i += gain * (actual_rel - desired_rel)

            new_positions[i] = current_positions[i] + dt * u_i

        current_positions = new_positions
        all_trajectory.append(current_positions.copy())

trajectory = np.array(all_trajectory)
all_targets = np.array(all_targets)

# ============================================================
# 4. ANIMATION RENDERING
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))
center_view = np.mean(initial_positions, axis=0)
ax.set_xlim(center_view[0] - 10, center_view[0] + 10)
ax.set_ylim(center_view[1] - 10, center_view[1] + 10)
ax.set_aspect("equal")
ax.grid(True, alpha=0.25)

target_points = ax.scatter(all_targets[0][:, 0], all_targets[0][:, 1], marker="x", s=80, color="red", alpha=0.4, label="Target")
agents = ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], s=220, color="#1f77b4", label="Agents", zorder=4)

labels = [ax.text(trajectory[0, i, 0], trajectory[0, i, 1], str(i + 1), ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=5) for i in range(N)]

edge_lines = []
for i, j in G.edges():
    line, = ax.plot([trajectory[0, i, 0], trajectory[0, j, 0]], [trajectory[0, i, 1], trajectory[0, j, 1]], color="gray", linewidth=0.4, alpha=0.2, zorder=1)
    edge_lines.append((line, i, j))

title = ax.set_title("Multi-Agent Formation Control", fontsize=14, fontweight="bold")
ax.legend(loc="upper right")

def update(frame):
    curr_pos = trajectory[frame]
    curr_target = all_targets[min(frame, len(all_targets) - 1)]
    letter_idx = min(frame // steps_per_letter, len(sequence) - 1)

    agents.set_offsets(curr_pos)
    target_points.set_offsets(curr_target)

    for i in range(N):
        labels[i].set_position((curr_pos[i, 0], curr_pos[i, 1]))

    for line, i, j in edge_lines:
        line.set_data([curr_pos[i, 0], curr_pos[j, 0]], [curr_pos[i, 1], curr_pos[j, 1]])

    title.set_text(f"20-Agent Formation Control | Name: SHREYANSH | Letter: {sequence[letter_idx]}")
    return [agents, target_points] + labels + [line for line, _, _ in edge_lines] + [title]

anim = FuncAnimation(fig, update, frames=len(trajectory), interval=30, blit=False)

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "SHREYANSH_formation_animation.mp4")

writer = FFMpegWriter(fps=30)
anim.save(output_file, writer=writer)
print(f"Animation saved successfully to {output_file}")