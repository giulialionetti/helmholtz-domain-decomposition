#!/usr/bin/env python3
"""
Animated visualization of boundary edges being constructed
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys

# --- 1. ROBUST PATH SETUP ---
# Find the project root by looking for the 'src' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
while not os.path.exists(os.path.join(project_root, 'src')):
    parent = os.path.dirname(project_root)
    if parent == project_root: # Reached filesystem root without finding src
        # Fallback: assume standard structure
        project_root = os.path.abspath(os.path.join(current_dir, "../../"))
        break
    project_root = parent

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 2. IMPORTS FROM SRC ---
from src.common.mesh import local_mesh, local_boundary

# --- 3. PARAMETERS ---
Lx, Ly = 1.0, 2.0
nx_global, ny_global = 33, 65
j = 1  # Middle subdomain for better visualization
J = 4

# --- 4. DATA GENERATION ---
# Get local mesh and boundaries
vtxj, eltj = local_mesh(Lx, Ly, nx_global, ny_global, j, J)
nx_local = nx_global
ny_local = len(np.unique(vtxj[:, 1]))
beltj_phys, beltj_artf = local_boundary(nx_local, ny_local, j, J)

# --- 5. ANIMATION SETUP ---
fig, ax = plt.subplots(figsize=(10, 8))

def animate(frame):
    ax.clear()
    
    # Plot all vertices (gray background)
    ax.scatter(vtxj[:, 0], vtxj[:, 1], c='lightgray', s=30, zorder=1, alpha=0.5)
    
    # Animate physical edges
    n_phys = len(beltj_phys)
    n_artf = len(beltj_artf)
    
    # Physical edges phase
    if frame < n_phys:
        # Draw completed physical edges
        for i in range(frame):
            edge = beltj_phys[i]
            v1, v2 = vtxj[edge[0]], vtxj[edge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-', linewidth=2, zorder=3)
        
        # Draw current active edge
        if frame > 0:
            edge = beltj_phys[frame-1]
            v1, v2 = vtxj[edge[0]], vtxj[edge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'b-', linewidth=4, zorder=4) # Blue highlight
            ax.scatter([v1[0], v2[0]], [v1[1], v2[1]], c='blue', s=100, zorder=5)
        
        title = f'Building Physical Edges: {frame}/{n_phys}'
        
    # Artificial edges phase
    else:
        # Draw all physical edges
        for edge in beltj_phys:
            v1, v2 = vtxj[edge[0]], vtxj[edge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-', linewidth=2, zorder=3, alpha=0.3)
        
        # Draw completed artificial edges
        artf_frame = frame - n_phys
        for i in range(min(artf_frame, n_artf)):
            edge = beltj_artf[i]
            v1, v2 = vtxj[edge[0]], vtxj[edge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'r--', linewidth=2, zorder=4)
        
        # Draw current active edge
        if 0 < artf_frame <= n_artf:
            edge = beltj_artf[artf_frame-1]
            v1, v2 = vtxj[edge[0]], vtxj[edge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'r-', linewidth=4, zorder=5) # Red highlight
            ax.scatter([v1[0], v2[0]], [v1[1], v2[1]], c='red', s=100, zorder=5)
        
        title = f'Building Artificial Edges: {min(artf_frame, n_artf)}/{n_artf}'
    
    ax.set_title(f'Subdomain {j}\n{title}', fontsize=14, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='k', linewidth=2, label='Physical'),
        Line2D([0], [0], color='r', linewidth=2, linestyle='--', label='Artificial'),
        Line2D([0], [0], color='blue', marker='o', label='Active Node'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')


total_frames = len(beltj_phys) + len(beltj_artf) + 10 
anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=50, repeat=False)


plots_dir = os.path.join(project_root, 'plots')
os.makedirs(plots_dir, exist_ok=True)
output_path = os.path.join(plots_dir, 'edges_animation.gif')

print(f"Generating animation ({total_frames} frames)...")
try:
    anim.save(output_path, writer='pillow', fps=15)
    print(f"Animation saved to: {output_path}")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("   (Ensure 'pillow' is installed: pip install pillow)")