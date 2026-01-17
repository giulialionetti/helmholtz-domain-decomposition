#!/usr/bin/env python3
"""
Test S_operator, Pi_operator, and g_vector
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
                           S_operator, Pi_operator, g_vector)

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
logger.info("Testing S_operator, Pi_operator, and g_vector")
logger.info("="*70)

# Build all local operators
factorizations = []
Bj_list = []
Cj_list = []
Tj_list = []
bj_list = []

logger.info("Building local operators for all subdomains")
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
    
    logger.info(f"  Subdomain {j}: LU factorization and RHS ready")

# Test operators
logger.info("")
logger.info("Testing operators")
n_skeleton = (J - 1) * nx_global
x_test = np.random.rand(n_skeleton) + 1j * np.random.rand(n_skeleton)
Px = Pi_operator(x_test, nx_global, J)
logger.info(f"  Input shape: {x_test.shape}")
logger.info(f"  Output shape: {Px.shape}")
logger.info(f"  Is identity: {np.allclose(Px, x_test)}")
assert Px.shape == x_test.shape, "Pi should preserve shape"

# Test S_operator
logger.info("")
logger.info("Testing S_operator")
Sx = S_operator(x_test, factorizations, Bj_list, Tj_list, Cj_list)
logger.info(f"  Input shape: {x_test.shape}")
logger.info(f"  Output shape: {Sx.shape}")
logger.info(f"  Output norm: {np.linalg.norm(Sx):.6e}")
assert Sx.shape == x_test.shape, "S should preserve shape"
assert np.iscomplexobj(Sx), "S output should be complex"



# Test g_vector
logger.info("")
logger.info("Testing g_vector")
g = g_vector(factorizations, bj_list, Bj_list, Cj_list, nx_global, J)
logger.info(f"  g shape: {g.shape}")
logger.info(f"  Expected: ({n_skeleton},)")
logger.info(f"  g norm: {np.linalg.norm(g):.6e}")
assert g.shape == (n_skeleton,), "g should be skeleton vector"
assert np.iscomplexobj(g), "g should be complex"

# Test interface problem residual
logger.info("")
logger.info("Testing interface problem: (I + Pi S) x = -g")
x_zero = np.zeros(n_skeleton, dtype=complex)
residual = S_operator(x_zero, factorizations, Bj_list, Tj_list, Cj_list) + g
residual = Pi_operator(residual, nx_global, J)
logger.info(f"  Residual at x=0: {np.linalg.norm(residual):.6e}")
logger.info(f"  This should equal norm(g): {np.linalg.norm(g):.6e}")

logger.info("")
logger.info("="*70)
logger.info("All tests passed")
logger.info("="*70)