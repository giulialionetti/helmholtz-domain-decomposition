#!/usr/bin/env python3
"""
Test local_boundary function
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

from src.helmholtz import local_mesh, local_boundary

# Test parameters
Lx = 1.0
Ly = 2.0
nx_global = 9
ny_global = 17
J = 4

logger.info("="*70)
logger.info("Testing local_boundary function")
logger.info("="*70)

for j in range(J):
    logger.info(f"\nSubdomain {j}:")
    
    # Get local mesh
    vtxj, eltj = local_mesh(Lx, Ly, nx_global, ny_global, j, J)
    nx_local = nx_global
    ny_local = len(np.unique(vtxj[:, 1]))
    
    # Get boundaries
    beltj_phys, beltj_artf = local_boundary(nx_local, ny_local, j, J)
    
    logger.info(f"  Physical edges: {len(beltj_phys)}")
    logger.info(f"  Artificial edges: {len(beltj_artf)}")
    
    # Verify edges
    if j == 0:
        # First subdomain: bottom is physical
        expected_phys = 3 * (nx_local - 1) + 2 * (ny_local - 1)  # bottom + left + right
        expected_artf = nx_local - 1  # top
        logger.info(f"  Expected: {expected_phys} physical (bottom+left+right), {expected_artf} artificial (top)")
    elif j == J - 1:
        # Last subdomain: top is physical
        expected_phys = 3 * (nx_local - 1) + 2 * (ny_local - 1)  # top + left + right
        expected_artf = nx_local - 1  # bottom
        logger.info(f"  Expected: {expected_phys} physical (top+left+right), {expected_artf} artificial (bottom)")
    else:
        # Middle subdomain: both top and bottom are artificial
        expected_phys = 2 * (nx_local - 1) + 2 * (ny_local - 1)  # left + right
        expected_artf = 2 * (nx_local - 1)  # top + bottom
        logger.info(f"  Expected: {expected_phys} physical (left+right), {expected_artf} artificial (top+bottom)")
    
    # Check unique vertices
    phys_verts = np.unique(beltj_phys.flatten()) if len(beltj_phys) > 0 else np.array([])
    artf_verts = np.unique(beltj_artf.flatten()) if len(beltj_artf) > 0 else np.array([])
    
    logger.info(f"  Physical boundary vertices: {len(phys_verts)}")
    logger.info(f"  Artificial boundary vertices: {len(artf_verts)}")
    
    # Check overlap (should only be corner vertices)
    overlap = np.intersect1d(phys_verts, artf_verts)
    if len(overlap) > 0:
        # For first/last subdomain: 2 corners (where left/right meet top/bottom)
        # For middle subdomains: 4 corners (where left/right meet both interfaces)
        expected_overlap = 2 if (j == 0 or j == J - 1) else 4
        if len(overlap) == expected_overlap:
            logger.info(f"   {len(overlap)} corner vertices shared (expected)")
        else:
            logger.error(f" ERROR: {len(overlap)} vertices overlap, expected {expected_overlap}")
    else:
        logger.info(f"No overlap between physical and artificial boundaries")
