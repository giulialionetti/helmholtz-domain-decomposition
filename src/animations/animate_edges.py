#!/usr/bin/env python3
"""
Animated visualization of boundary edges being constructed
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from src.helmholtz import local_mesh, local_boundary

# Parameters
Lx, Ly = 1.0, 2.0
nx_global, ny_global = 17, 33
j = 1  # Middle subdomain for better visualization
J = 4

# Get local mesh and boundaries
vtxj, eltj = local_mesh(Lx, Ly, nx_global, ny_global, j, J)
nx_local = nx_global
ny_local = len(np.unique(vtxj[:, 1]))
beltj_phys, beltj_artf = local_boundary(nx_local, ny_local, j, J)

# Setup figure
fig, ax = plt.subplots(figsize=(12, 8))

def animate(frame):
    ax.clear()
    
    # Plot all vertices (gray)
    ax.scatter(vtxj[:, 0], vtxj[:, 1], c='lightgray', s=50, zorder=1, alpha=0.5)
    
    # Animate physical edges
    n_phys = len(beltj_phys)
    n_artf = len(beltj_artf)
    total_edges = n_phys + n_artf
    
    # Physical edges (0 to n_phys frames)
    if frame < n_phys:
        # Draw completed physical edges
        for i in range(frame):
            edge = beltj_phys[i]
            v1, v2 = vtxj[edge[0]], vtxj[edge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-', linewidth=3, zorder=3)
        
        # Draw current physical edge being added (animated)
        if frame > 0:
            edge = beltj_phys[frame-1]
            v1, v2 = vtxj[edge[0]], vtxj[edge[1]]
            # Highlight vertices
            ax.scatter([v1[0], v2[0]], [v1[1], v2[1]], c='blue', s=200, zorder=5, marker='o')
            # Add labels
            ax.text(v1[0], v1[1]+0.03, f'v{edge[0]}', ha='center', fontsize=8, fontweight='bold')
            ax.text(v2[0], v2[1]+0.03, f'v{edge[1]}', ha='center', fontsize=8, fontweight='bold')
        
        title = f'Physical Edges: {frame}/{n_phys}'
        
    # Artificial edges (n_phys to total_edges frames)
    else:
        # Draw all physical edges
        for edge in beltj_phys:
            v1, v2 = vtxj[edge[0]], vtxj[edge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-', linewidth=3, zorder=3, alpha=0.5)
        
        # Draw completed artificial edges
        artf_frame = frame - n_phys
        for i in range(min(artf_frame, n_artf)):
            edge = beltj_artf[i]
            v1, v2 = vtxj[edge[0]], vtxj[edge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'r--', linewidth=3, zorder=4)
        
        # Draw current artificial edge being added
        if 0 < artf_frame <= n_artf:
            edge = beltj_artf[artf_frame-1]
            v1, v2 = vtxj[edge[0]], vtxj[edge[1]]
            ax.scatter([v1[0], v2[0]], [v1[1], v2[1]], c='red', s=200, zorder=5, marker='o')
            ax.text(v1[0], v1[1]+0.03, f'v{edge[0]}', ha='center', fontsize=8, fontweight='bold')
            ax.text(v2[0], v2[1]+0.03, f'v{edge[1]}', ha='center', fontsize=8, fontweight='bold')
        
        title = f'Artificial Edges: {min(artf_frame, n_artf)}/{n_artf} (Physical: {n_phys} complete)'
    
    ax.set_title(f'Subdomain {j} - {title}', fontsize=14, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, Lx+0.05)
    ax.set_ylim(vtxj[:, 1].min()-0.05, vtxj[:, 1].max()+0.05)
    ax.set_aspect('equal')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='k', linewidth=3, label='Physical boundary'),
        Line2D([0], [0], color='r', linewidth=3, linestyle='--', label='Artificial interface'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=10, label='Current vertices')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

# Create animation
total_frames = len(beltj_phys) + len(beltj_artf) + 10  # +10 for pause at end
anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=200, repeat=True)

# Save
plots_dir = os.path.join('..', 'plots')
os.makedirs(plots_dir, exist_ok=True)
output_path = os.path.join(plots_dir, 'edges_animation.gif')
anim.save(output_path, writer='pillow', fps=5)

plt.close()
