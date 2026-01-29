#!/usr/bin/env python3
"""
Animated visualization of mesh construction: vertices -> edges -> triangles
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys

# --- ROBUST PATH SETUP ---
# Find the project root by looking for the 'src' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
while not os.path.exists(os.path.join(project_root, 'src')):
    parent = os.path.dirname(project_root)
    if parent == project_root:
        # Fallback: assume standard structure
        project_root = os.path.abspath(os.path.join(current_dir, "../../"))
        break
    project_root = parent

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.common.mesh import local_mesh, local_boundary

# Parameters
Lx, Ly = 1.0, 2.0
nx_global, ny_global = 33, 65 
j = 1
J = 4

# Get mesh
vtxj, eltj = local_mesh(Lx, Ly, nx_global, ny_global, j, J)
nx_local = nx_global
ny_local = len(np.unique(vtxj[:, 1]))
beltj_phys, beltj_artf = local_boundary(nx_local, ny_local, j, J)

# Animation phases
n_vertices = len(vtxj)
n_edges = len(beltj_phys) + len(beltj_artf)
n_triangles = len(eltj)

fig, ax = plt.subplots(figsize=(12, 8))

def animate(frame):
    ax.clear()
    
    # Phase 1: Show vertices (frames 0-20)
    if frame < 20:
        progress = min(frame / 20.0, 1.0)
        n_show = int(progress * n_vertices)
        ax.scatter(vtxj[:n_show, 0], vtxj[:n_show, 1], c='blue', s=100, zorder=5)
        title = f'Phase 1: Vertices ({n_show}/{n_vertices})'
    
    # Phase 2: Draw boundary edges (frames 20-50)
    elif frame < 50:
        ax.scatter(vtxj[:, 0], vtxj[:, 1], c='lightgray', s=50, zorder=1, alpha=0.5)
        
        progress = (frame - 20) / 30.0
        n_show_phys = int(progress * len(beltj_phys))
        n_show_artf = int(progress * len(beltj_artf))
        
        for edge in beltj_phys[:n_show_phys]:
            v1, v2 = vtxj[edge[0]], vtxj[edge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-', linewidth=2, zorder=3)
        
        for edge in beltj_artf[:n_show_artf]:
            v1, v2 = vtxj[edge[0]], vtxj[edge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'r--', linewidth=2, zorder=3)
        
        title = f'Phase 2: Boundary Edges ({n_show_phys + n_show_artf}/{n_edges})'
    
    # Phase 3: Fill triangles (frames 50-120)
    else:
        ax.scatter(vtxj[:, 0], vtxj[:, 1], c='lightgray', s=30, zorder=1, alpha=0.3)
        
        progress = (frame - 50) / 70.0
        n_show = int(progress * n_triangles)
        
        # Draw filled triangles
        for i, tri in enumerate(eltj[:n_show]):
            vertices = vtxj[tri]
            color = plt.cm.viridis(i / n_triangles)
            ax.fill(vertices[:, 0], vertices[:, 1], color=color, alpha=0.6, edgecolor='black', linewidth=0.5)
        
        # Highlight current triangle
        if n_show > 0 and n_show <= n_triangles:
            tri = eltj[n_show-1]
            vertices = vtxj[tri]
            ax.fill(vertices[:, 0], vertices[:, 1], color='red', alpha=0.8, edgecolor='red', linewidth=2)
        
        # Draw boundaries on top
        for edge in beltj_phys:
            v1, v2 = vtxj[edge[0]], vtxj[edge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-', linewidth=3, zorder=10)
        
        for edge in beltj_artf:
            v1, v2 = vtxj[edge[0]], vtxj[edge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'r--', linewidth=3, zorder=10)
        
        title = f'Phase 3: Triangles ({n_show}/{n_triangles})'
    
    ax.set_title(f'Subdomain {j} - {title}', fontsize=14, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, Lx+0.05)
    ax.set_ylim(vtxj[:, 1].min()-0.05, vtxj[:, 1].max()+0.05)
    ax.set_aspect('equal')

# Create animation
total_frames = 130
anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=100, repeat=True)

# Save
plots_dir = os.path.join(project_root, 'plots')
os.makedirs(plots_dir, exist_ok=True)
output_path = os.path.join(plots_dir, 'mesh_construction.gif')

print("Creating animation... (this may take a minute)")
try:
    anim.save(output_path, writer='pillow', fps=10)
    print(f"Saved animation to {output_path}")
except Exception as e:
    print(f"Error saving animation: {e}")

plt.close()