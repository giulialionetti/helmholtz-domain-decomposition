#!/usr/bin/env python3
"""
Test mesh refinement and analyze GMRES convergence
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from src.helmholtz_base import mesh, boundary, mass, stiffness, point_source

# Problem parameters
Lx = 1.0
Ly = 2.0
k = 16.0
ns = 8
np.random.seed(42)
sp = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in range(ns)]

# Test different mesh sizes (keeping aspect ratio)
mesh_sizes = [
    (17, 33),   # coarse
    (33, 65),   # medium
    (49, 97),   # fine
    (65, 129),  # finer
]

results = []

logger.info("="*70)
logger.info("Mesh Refinement Study")
logger.info("="*70)

for nx, ny in mesh_sizes:
    logger.info(f"\nTesting mesh: nx={nx}, ny={ny}")
    
    # Check isotropy (aspect ratio of mesh cells)
    hx = Lx / (nx - 1)
    hy = Ly / (ny - 1)
    aspect_ratio = max(hx, hy) / min(hx, hy)
    logger.info(f"  Mesh spacing: hx={hx:.6f}, hy={hy:.6f}, aspect ratio={aspect_ratio:.3f}")
    
    if aspect_ratio > 1.5:
        logger.warning(f" Mesh is anisotropic (aspect ratio > 1.5)!")
    
    # Build problem
    vtx, elt = mesh(nx, ny, Lx, Ly)
    belt = boundary(nx, ny)
    M = mass(vtx, elt)
    Mb = mass(vtx, belt)
    K = stiffness(vtx, elt)
    A = K - k**2 * M - 1j*k*Mb
    b = M @ point_source(sp, k)(vtx)
    
    ndof = nx * ny
    logger.info(f"  DOFs: {ndof}")
    
    # GMRES solve
    residuals = []
    def callback(x):
        residuals.append(x)
    
    tol = 1e-10
    x, info = spla.gmres(A, b, rtol=tol, callback=callback, callback_type='pr_norm', maxiter=5000)
    
    n_iter = len(residuals)
    logger.info(f"  GMRES iterations: {n_iter}")
    logger.info(f"  Final residual: {residuals[-1]:.6e}")
    logger.info(f"  Converged: {info == 0}")
    
    results.append({
        'nx': nx,
        'ny': ny,
        'ndof': ndof,
        'h': max(hx, hy),
        'aspect_ratio': aspect_ratio,
        'iterations': n_iter,
        'residuals': residuals,
        'converged': info == 0
    })


plots_dir = os.path.join('..', 'plots')
os.makedirs(plots_dir, exist_ok=True)

# Plot 1: Iterations vs DOFs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ndofs = [r['ndof'] for r in results]
iters = [r['iterations'] for r in results]
hs = [r['h'] for r in results]

ax1.plot(ndofs, iters, 'o-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of DOFs')
ax1.set_ylabel('GMRES Iterations')
ax1.set_title('Iterations vs Problem Size')
ax1.grid(True, alpha=0.3)
for i, (nd, it) in enumerate(zip(ndofs, iters)):
    ax1.annotate(f'{results[i]["nx"]}×{results[i]["ny"]}', 
                 (nd, it), textcoords="offset points", xytext=(0,10), ha='center')

ax2.loglog(hs, iters, 's-', linewidth=2, markersize=8)
ax2.set_xlabel('Mesh size h')
ax2.set_ylabel('GMRES Iterations')
ax2.set_title('Iterations vs Mesh Size (log-log)')
ax2.grid(True, alpha=0.3, which='both')
ax2.invert_xaxis()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'mesh_refinement_iterations.png'), dpi=150)

# Plot 2: Convergence curves
fig, ax = plt.subplots(figsize=(10, 6))

for r in results:
    label = f"{r['nx']}×{r['ny']} ({r['ndof']} DOFs)"
    ax.semilogy(r['residuals'], label=label, linewidth=2)

ax.set_xlabel('Iteration')
ax.set_ylabel('Residual norm')
ax.set_title('GMRES Convergence for Different Mesh Refinements')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'mesh_refinement_convergence.png'), dpi=150)


logger.info("\n" + "="*70)
logger.info("Summary")
logger.info("="*70)
logger.info(f"{'Mesh':<15} {'DOFs':<10} {'h':<12} {'Aspect':<10} {'Iterations':<12}")
logger.info("-"*70)
for r in results:
    mesh_str = f"{r['nx']}×{r['ny']}"
    logger.info(f"{mesh_str:<15} {r['ndof']:<10} {r['h']:<12.6f} {r['aspect_ratio']:<10.3f} {r['iterations']:<12}")

logger.info("\nKey observations:")
logger.info(f"  - Iterations increase from {results[0]['iterations']} to {results[-1]['iterations']}")
logger.info(f"  - Growth rate: {results[-1]['iterations']/results[0]['iterations']:.2f}x for {results[-1]['ndof']/results[0]['ndof']:.2f}x more DOFs")
logger.info(f"  - All meshes remain isotropic (aspect ratio ≈ 1.0)")
logger.info("="*70)