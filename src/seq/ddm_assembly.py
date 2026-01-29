"""
Functions related to the assembly of the Domain Decomposition Method in the sequential case
"""

import numpy as np
from src.common.mesh import local_boundary, local_mesh
from src.common.helmholtz.helmholtz_param import HelmholtzParameters
from src.common.ddm_operators import (Bj_matrix, Cj_matrix, Aj_matrix, Tj_matrix,
                           Sj_factorization, bj_vector, g_vector)

def build_ddm_solver(nx, ny, J, param : HelmholtzParameters):
    """
    Builds all DDM components and aggregates them into lists to perform the Domain
    Decomposition in a sequential setting
    """
    factorizations = []
    Bj_list = []
    Cj_list = []
    Tj_list = []
    bj_list = []
    vtxj_list = []
    eltj_list = []
    
    for j in range(J):
        vtxj, eltj = local_mesh(param.Lx, param.Ly, nx, ny, j, J)
        nx_local = nx
        ny_local = len(np.unique(vtxj[:, 1]))
        
        beltj_phys, beltj_artf = local_boundary(nx_local, ny_local, j, J)
        
        Bj = Bj_matrix(nx_local, ny_local, j, J, beltj_artf)
        Cj = Cj_matrix(nx, ny, j, J)
        Aj = Aj_matrix(vtxj, eltj, beltj_phys, param.kappa)
        Tj = Tj_matrix(vtxj, beltj_artf, Bj, param.kappa)
        LU_j = Sj_factorization(Aj, Tj, Bj)
        bj = bj_vector(vtxj, eltj, param.sp, param.kappa)
        
        factorizations.append(LU_j)
        Bj_list.append(Bj)
        Cj_list.append(Cj)
        Tj_list.append(Tj)
        bj_list.append(bj)
        vtxj_list.append(vtxj)
        eltj_list.append(eltj)
    
    g = g_vector(factorizations, bj_list, Bj_list, Cj_list, nx, J)
    
    return factorizations, Bj_list, Cj_list, Tj_list, bj_list, vtxj_list, eltj_list, g