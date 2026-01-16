#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
from src.helmholtz import Rj_matrix

nx, ny, J = 9, 17, 4

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for j in range(J):
    Rj = Rj_matrix(nx, ny, j, J)
    
    ax = axes[j]
    
    # Visualize as heatmap
    ax.spy(Rj, markersize=2, color='blue')
    ax.set_title(f'Rj for Subdomain {j}\nShape: {Rj.shape[0]} local × {Rj.shape[1]} global DOFs', 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Global DOF index')
    ax.set_ylabel('Local DOF index')
    
    # Add info text
    info_text = f'Maps {Rj.shape[1]} global → {Rj.shape[0]} local\nNon-zeros: {Rj.nnz}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plots_dir = os.path.join('..', 'plots')
os.makedirs(plots_dir, exist_ok=True)
output_path = os.path.join(plots_dir, 'Rj_matrices.png')
plt.savefig(output_path, dpi=150)
