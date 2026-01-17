#!/usr/bin/env python3
"""
Test Tj_matrix, Sj_factorization, and bj_vector
"""


import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from src.helmholtz import (local_mesh, local_boundary, Bj_matrix, 
                           Aj_matrix, Tj_matrix, Sj_factorization, bj_vector)

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
logger.info("Testing Tj_matrix, Sj_factorization, and bj_vector")
logger.info("="*70)

for j in range(J):
    logger.info(f"Subdomain {j}")
    
    # Get local mesh
    vtxj, eltj = local_mesh(Lx, Ly, nx_global, ny_global, j, J)
    nx_local = nx_global
    ny_local = len(np.unique(vtxj[:, 1]))
    nv_local = nx_local * ny_local
    
    # Get boundaries
    beltj_phys, beltj_artf = local_boundary(nx_local, ny_local, j, J)
    
    # Build matrices
    Bj = Bj_matrix(nx_local, ny_local, j, J, beltj_artf)
    Aj = Aj_matrix(vtxj, eltj, beltj_phys, kappa)
    Tj = Tj_matrix(vtxj, beltj_artf, Bj, kappa)
    
    # Test Tj
    logger.info(f"  Tj shape: {Tj.shape}")
    logger.info(f"  Expected: ({Bj.shape[0]}, {Bj.shape[0]})")
    logger.info(f"  Tj non-zeros: {Tj.nnz}")
    assert Tj.shape == (Bj.shape[0], Bj.shape[0]), "Tj should be interface x interface"
    
    # Test Sj factorization
    LU_j = Sj_factorization(Aj, Tj, Bj)
    logger.info(f"  LU factorization successful")
    logger.info(f"  LU shape: ({LU_j.shape[0]}, {LU_j.shape[1]})")
    assert LU_j.shape == (nv_local, nv_local), "LU should match Aj dimensions"
    
    # Test bj_vector
    bj = bj_vector(vtxj, eltj, sp, kappa)
    logger.info(f"  bj shape: {bj.shape}")
    logger.info(f"  Expected: ({nv_local},)")
    logger.info(f"  bj norm: {np.linalg.norm(bj):.6e}")
    assert bj.shape == (nv_local,), "bj should be vector of local DOFs"
    
    # Test that we can solve with the factorization
    test_rhs = np.random.rand(nv_local) + 1j * np.random.rand(nv_local)
    test_sol = LU_j.solve(test_rhs)
    logger.info(f"  Test solve successful, solution norm: {np.linalg.norm(test_sol):.6e}")
    
    logger.info("")

logger.info("="*70)
logger.info("All tests passed")
logger.info("="*70)