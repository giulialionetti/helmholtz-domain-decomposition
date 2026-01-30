import scipy.sparse.linalg as spla
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_path = current_dir
while not os.path.exists(os.path.join(project_root, 'src')):
    parent = os.path.dirname(project_root)
    if parent == project_root:
        project_root = os.path.abspath(os.path.join(current_dir, "../"))
        break
    project_root = parent

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.common.mesh import local_boundary, local_mesh
from src.common.helmholtz.helmholtz_param import HelmholtzParameters
from src.common.ddm_operators import (Bj_matrix, Cj_matrix, Aj_matrix, Tj_matrix,
                           Sj_factorization, bj_vector, g_vector)
from src.common.helmholtz.helmholtz_param import HelmholtzParameters
from src.common.ddm_operators import(S_operator, Pi_operator)

class HelmholtzSolver:
    def __init__(self, param : HelmholtzParameters, nx, ny, j, totalJ):
        self._param = param
        self._nx = nx
        self._ny = ny
        self._j = j
        self._totalJ = totalJ

        self._x = 0
        self._residuals = []

    def assembly(self):
        """
        Builds all DDM components and aggregates them into lists to perform the Domain
        Decomposition in a sequential setting
        """
        vtxj, eltj = local_mesh(self._param.Lx, self._param.Ly, self._nx, self._ny, self._j, self._totalJ)
        nx_local = self._nx
        ny_local = len(np.unique(vtxj[:, 1]))
        
        beltj_phys, beltj_artf = local_boundary(nx_local, ny_local, self._j, self._totalJ)
        
        Bj = Bj_matrix(nx_local, ny_local, self._j, self._totalJ, beltj_artf)
        Cj = Cj_matrix(self._nx, self._ny, self._j, self._totalJ)
        Aj = Aj_matrix(vtxj, eltj, beltj_phys, self._param.kappa)
        Tj = Tj_matrix(vtxj, beltj_artf, Bj, self._param.kappa)
        LU_j = Sj_factorization(Aj, Tj, Bj)
        bj = bj_vector(vtxj, eltj, self._param.sp, self._param.kappa)
            
        
        self._g = g_vector(self._factorizations, self._bj_list, self._Bj_list, self._Cj_list, self._nx, self._totalJ)
        
        # return factorizations, Bj_list, Cj_list, Tj_list, bj_list, vtxj_list, eltj_list, g
    
    def solve(self):
        def S_op(x):
            return S_operator(x, self._factorizations, self._Bj_list, self._Tj_list, self._Cj_list)
        
        def Pi_op(x):
            return Pi_operator(x, self._nx, self._totalJ)
        
        def matvec(x):
            return x + Pi_op(S_op(x))
        
        n_skeleton = len(self._g)
        A_op = spla.LinearOperator((n_skeleton, n_skeleton), matvec=matvec, dtype=complex)
        
        self._residuals = []
        def callback(rk):
            self._residuals.append(rk)
        
        x, info = spla.gmres(A_op, -self._g, rtol=1e-10, callback=callback, 
                            callback_type='pr_norm', maxiter=500)
        
        self._x = x
        return x, self._residuals, info, S_op, Pi_op
    
    def getX(self):
        return self._x
    
    def getComponents(self):
        return (self._factorizations, self._Bj_list, self._Cj_list, self._Tj_list,
                self._bj_list, self._vtxj_list, self._eltj_list, self._g)