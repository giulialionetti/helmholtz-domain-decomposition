#!/usr/bin/env python3
"""
Test Aj_matrix
"""

import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from src.helmholtz import local_mesh, local_boundary, Aj_matrix

# Parameters
Lx, Ly = 1.0, 2.0
nx_global, ny_global = 9, 17
J = 4
kappa = 16.0

logger.info("="*70)
logger.info("Testing Aj_matrix")
logger.info("="*70)

for j in range(J):
    logger.info(f"Subdomain {j}")
    
    # Get local mesh
    vtxj, eltj = local_mesh(Lx, Ly, nx_global, ny_global, j, J)
    nx_local = nx_global
    ny_local = len(np.unique(vtxj[:, 1]))
    
    # Get boundaries
    beltj_phys, beltj_artf = local_boundary(nx_local, ny_local, j, J)
    
    # Build Aj
    Aj = Aj_matrix(vtxj, eltj, beltj_phys, kappa)
    
    nv_local = nx_local * ny_local
    logger.info(f"  Aj shape: {Aj.shape}")
    logger.info(f"  Expected: ({nv_local}, {nv_local})")
    logger.info(f"  Non-zeros: {Aj.nnz}")
    logger.info(f"  Sparsity: {Aj.nnz / (nv_local**2) * 100:.2f}%")
    logger.info(f"  Is complex: {np.iscomplexobj(Aj.data)}")
    logger.info(f"  Physical boundary edges: {len(beltj_phys)}")
    
    # Check properties
    assert Aj.shape == (nv_local, nv_local), "Aj should be square"
    assert np.iscomplexobj(Aj.data), "Aj should be complex (due to ikMb term)"
    
    logger.info(f"  Matrix constructed successfully")
    logger.info("")

logger.info("="*70)
logger.info("All tests passed")
logger.info("="*70)