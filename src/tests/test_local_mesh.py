#!/usr/bin/env python3
"""
Test script for local_mesh function
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
import os, sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from src.helmholtz_base import mesh, boundary, mass, stiffness, point_source, plot_mesh
from src.helmholtz import local_mesh

# Test parameters
Lx = 1.0
Ly = 2.0
nx = 9  # Keep small for visualization
ny = 17  # 16 intervals -> divisible by J=4
J = 4

logger.info("="*70)
logger.info("Testing local_mesh function")
logger.info("="*70)
logger.info(f"Global mesh: {nx} x {ny} points")
logger.info(f"Domain: [{Lx}] x [{Ly}]")
logger.info(f"Number of subdomains: {J}")
logger.info(f"Expected intervals per subdomain: {(ny-1)//J}")

# Generate global mesh for comparison
vtx_global, elt_global = mesh(nx, ny, Lx, Ly)
logger.info(f"Global mesh: {vtx_global.shape[0]} vertices, {elt_global.shape[0]} triangles")

# Test each subdomain
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for j in range(J):
    logger.info(f"\n--- Subdomain {j} ---")
    
    # Generate local mesh
    vtxj, eltj = local_mesh(Lx, Ly, nx, ny, j, J)
    
    logger.info(f"  Local vertices: {vtxj.shape[0]}")
    logger.info(f"  Local triangles: {eltj.shape[0]}")
    logger.info(f"  Y-range: [{vtxj[:, 1].min():.4f}, {vtxj[:, 1].max():.4f}]")
    logger.info(f"  X-range: [{vtxj[:, 0].min():.4f}, {vtxj[:, 0].max():.4f}]")
    
    # Expected y-range
    y_start = j * Ly / J
    y_end = (j + 1) * Ly / J
    logger.info(f"  Expected Y-range: [{y_start:.4f}, {y_end:.4f}]")
    
    # Plot
    ax = axes[j]
    plot_mesh(vtxj, eltj)
    ax.set_title(f'Subdomain {j}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    
plots_dir = os.path.join('..', 'plots')
os.makedirs(plots_dir, exist_ok=True)
plt.tight_layout()
output_path = os.path.join(plots_dir, 'test_local_mesh.png')
plt.savefig(output_path, dpi=150)

# Check total vertices (should add up with overlap)
total_local_vertices = sum([local_mesh(Lx, Ly, nx, ny, j, J)[0].shape[0] for j in range(J)])
expected_with_overlap = nx * ny + nx * (J - 1)  # Interfaces counted twice
logger.info(f"\nTotal local vertices (with overlap): {total_local_vertices}")
logger.info(f"Expected (global + interfaces): {expected_with_overlap}")
logger.info(f"Global vertices: {nx * ny}")
logger.info(f"Extra due to overlap: {total_local_vertices - nx * ny} (should be {nx * (J-1)})")
