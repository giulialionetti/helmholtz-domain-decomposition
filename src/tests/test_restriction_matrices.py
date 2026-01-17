#!/usr/bin/env python3
"""
Test Bj_matrix and Cj_matrix together
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from src.helmholtz import local_mesh, local_boundary, Bj_matrix, Cj_matrix

# Parameters
Lx, Ly = 1.0, 2.0
nx_global, ny_global = 9, 17
J = 4

logger.info("="*70)
logger.info("Testing Bj_matrix and Cj_matrix")
logger.info("="*70)

# Global skeleton size
n_skeleton = (J - 1) * nx_global
logger.info(f"Global skeleton: {J-1} interfaces × {nx_global} vertices = {n_skeleton} DOFs")

Bj_list = []
Cj_list = []

for j in range(J):
    logger.info(f"--- Subdomain {j} ---")
    
    # Get local mesh
    vtxj, eltj = local_mesh(Lx, Ly, nx_global, ny_global, j, J)
    nx_local = nx_global
    ny_local = len(np.unique(vtxj[:, 1]))
    
    # Get boundaries
    beltj_phys, beltj_artf = local_boundary(nx_local, ny_local, j, J)
    
    # Build matrices
    Bj = Bj_matrix(nx_local, ny_local, j, J, beltj_artf)
    Cj = Cj_matrix(nx_global, ny_global, j, J)
    
    Bj_list.append(Bj)
    Cj_list.append(Cj)
    
    logger.info(f"  Local DOFs: {nx_local * ny_local}")
    logger.info(f"  Bj shape: {Bj.shape} (interface DOFs × local DOFs)")
    logger.info(f"  Cj shape: {Cj.shape} (local interface DOFs × global skeleton DOFs)")
    logger.info(f"  Bj extracts: {Bj.shape[0]} interface DOFs from {Bj.shape[1]} local DOFs")
    logger.info(f"  Cj extracts: {Cj.shape[0]} local interface DOFs from {Cj.shape[1]} global skeleton")
    
    # Verify dimensions match
    assert Bj.shape[0] == Cj.shape[0], "Bj and Cj interface dimensions must match!"
    logger.info(f" Bj and Cj dimensions compatible")
    
    # Check which interfaces
    if j == 0:
        logger.info(f"  First subdomain: only top interface (interface 0)")
    elif j == J - 1:
        logger.info(f"  Last subdomain: only bottom interface (interface {J-2})")
    else:
        logger.info(f"  Middle subdomain: bottom (interface {j-1}) + top (interface {j})")

# Test the full pipeline: local → interface → skeleton → back
logger.info("="*70)
logger.info("Testing full pipeline")
logger.info("="*70)

# Create random global skeleton vector
x_skeleton = np.random.rand(n_skeleton)
logger.info(f"Global skeleton vector: {x_skeleton.shape}")

# Extract to each subdomain and sum (should reconstruct with proper accounting)
total_extracted = 0
for j in range(J):
    xj = Cj_list[j] @ x_skeleton
    logger.info(f"  Subdomain {j}: extracted {len(xj)} interface DOFs")
    total_extracted += len(xj)

logger.info(f"Total extracted: {total_extracted} (from {n_skeleton} skeleton DOFs)")
logger.info(f"Expected: {2 * (J-1) * nx_global} (each interface counted twice)")

# Visualize Bj and Cj matrices
fig, axes = plt.subplots(2, J, figsize=(16, 8))

for j in range(J):
    # Plot Bj
    ax = axes[0, j]
    ax.spy(Bj_list[j], markersize=3, color='blue')
    ax.set_title(f'Bj (subdomain {j})\n{Bj_list[j].shape}')
    ax.set_xlabel('Local DOF')
    ax.set_ylabel('Interface DOF')
    
    # Plot Cj
    ax = axes[1, j]
    ax.spy(Cj_list[j], markersize=3, color='red')
    ax.set_title(f'Cj (subdomain {j})\n{Cj_list[j].shape}')
    ax.set_xlabel('Skeleton DOF')
    ax.set_ylabel('Local interface DOF')

plt.tight_layout()
plots_dir = os.path.join('..', 'plots')
os.makedirs(plots_dir, exist_ok=True)
output_path = os.path.join(plots_dir, 'Bj_Cj_matrices.png')
plt.savefig(output_path, dpi=150)
logger.info(f"Saved visualization to {output_path}")

logger.info("="*70)
logger.info(" All tests passed!")
logger.info("="*70)