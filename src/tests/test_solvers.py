#!/usr/bin/env python3
"""
Test fixed_point_solver and uj_solution
"""

import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from src.helmholtz import (local_mesh, local_boundary, Bj_matrix, Cj_matrix,
                           Aj_matrix, Tj_matrix, Sj_factorization, bj_vector,
                           S_operator, Pi_operator, g_vector, 
                           fixed_point_solver, uj_solution)

# Parameters
Lx, Ly = 1.0, 2.0
nx_global, ny_global = 9, 17
J = 4
kappa = 16.0

# Point sources
ns = 8
np.random.seed(42)
sp = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in range(ns)]

logger.info("="*70)
logger.info("Testing fixed_point_solver and uj_solution")
logger.info("="*70)

# Build all local operators
logger.info("Building local operators")
factorizations = []
Bj_list = []
Cj_list = []
Tj_list = []
bj_list = []
vtxj_list = []
eltj_list = []

for j in range(J):
    vtxj, eltj = local_mesh(Lx, Ly, nx_global, ny_global, j, J)
    nx_local = nx_global
    ny_local = len(np.unique(vtxj[:, 1]))
    
    beltj_phys, beltj_artf = local_boundary(nx_local, ny_local, j, J)
    
    Bj = Bj_matrix(nx_local, ny_local, j, J, beltj_artf)
    Cj = Cj_matrix(nx_global, ny_global, j, J)
    Aj = Aj_matrix(vtxj, eltj, beltj_phys, kappa)
    Tj = Tj_matrix(vtxj, beltj_artf, Bj, kappa)
    LU_j = Sj_factorization(Aj, Tj, Bj)
    bj = bj_vector(vtxj, eltj, sp, kappa)
    
    factorizations.append(LU_j)
    Bj_list.append(Bj)
    Cj_list.append(Cj)
    Tj_list.append(Tj)
    bj_list.append(bj)
    vtxj_list.append(vtxj)
    eltj_list.append(eltj)

logger.info("All local operators ready")

# Build global RHS
logger.info("")
logger.info("Building global interface RHS")
g = g_vector(factorizations, bj_list, Bj_list, Cj_list, nx_global, J)
logger.info(f"  g shape: {g.shape}")
logger.info(f"  g norm: {np.linalg.norm(g):.6e}")

# Define operator functions for solver
def S_op(x):
    return S_operator(x, factorizations, Bj_list, Tj_list, Cj_list)

def Pi_op(x):
    return Pi_operator(x, nx_global, J)

# Takes 276 iterations to converge with omega=0.1, tol=1e-8
logger.info("")
logger.info("Testing fixed_point_solver")
omega = 0.1
max_iter = 276
tol = 1e-8

x_solution, residuals, converged = fixed_point_solver(-g, S_op, Pi_op, omega, max_iter, tol)

logger.info(f"  Solution shape: {x_solution.shape}")
logger.info(f"  Number of iterations: {len(residuals)}")
logger.info(f"  Initial residual: {residuals[0]:.6e}")
logger.info(f"  Final residual: {residuals[-1]:.6e}")
logger.info(f"  Converged: {converged}")

# Test uj_solution for each subdomain
logger.info("")
logger.info("Testing uj_solution")
for j in range(J):
    # Extract local interface solution
    xj = Cj_list[j] @ x_solution
    
    # Compute local solution
    uj = uj_solution(xj, factorizations[j], Bj_list[j], Tj_list[j], bj_list[j])
    
    logger.info(f"  Subdomain {j}:")
    logger.info(f"    xj shape: {xj.shape}")
    logger.info(f"    uj shape: {uj.shape}")
    logger.info(f"    uj norm: {np.linalg.norm(uj):.6e}")
    
    assert uj.shape == bj_list[j].shape, "uj should match local DOFs"

logger.info("")
logger.info("="*70)
logger.info("All tests passed")
logger.info("="*70)