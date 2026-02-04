import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix
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

from src.seq.mesh.mesh import FullBoundary, FullMesh
from src.common.helmholtz.helmholtz_param import HelmholtzParameters
from src.seq.operators.b_operator import FullBOperator
from src.seq.operators.a_operator import FullAOperator
from src.seq.operators.t_operator import FullTOperator
from src.seq.operators.c_operator import FullCOperator
from src.seq.operators.s_operator import FullSFactorization, FullSOperator
from src.seq.operators.pi_operator import FullPiOperator
from src.seq.operators.bv_operator import FullBVecOperator
from src.seq.operators.g_operator import FullGVecOperator
from src.common.helmholtz.helmholtz_param import HelmholtzParameters

class HelmholtzSolver:
    def __init__(self, params : HelmholtzParameters, J: int):
        self._params = params
        # self._nx = nx
        # self._ny = ny
        self._J = J

        self._x = 0
        self._residuals = []

    def assembly(self):
        """
        Builds all DDM components and aggregates them into lists to perform the Domain
        Decomposition in a sequential setting
        """
        # self._factorizations = []
        # self._Bj_list = []
        # self._Cj_list = []
        # self._Tj_list = []
        # self._bj_list = []
        # self._vtxj_list = []
        # self._eltj_list = []
        
        # for j in range(self._J):
        #     vtxj, eltj = local_mesh(self._param.Lx, self._param.Ly, self._nx, self._ny, j, self._J)
        #     nx_local = self._nx
        #     ny_local = len(np.unique(vtxj[:, 1]))
            
        #     beltj_phys, beltj_artf = local_boundary(nx_local, ny_local, j, self._J)
            
        #     Bj = Bj_matrix(nx_local, ny_local, j, self._J, beltj_artf)
        #     Cj = Cj_matrix(self._nx, self._ny, j, self._J)
        #     Aj = Aj_matrix(vtxj, eltj, beltj_phys, self._param.kappa)
        #     Tj = Tj_matrix(vtxj, beltj_artf, Bj, self._param.kappa)
        #     LU_j = Sj_factorization(Aj, Tj, Bj)
        #     bj = bj_vector(vtxj, eltj, self._param.sp, self._param.kappa)
            
        #     self._factorizations.append(LU_j)
        #     self._Bj_list.append(Bj)
        #     self._Cj_list.append(Cj)
        #     self._Tj_list.append(Tj)
        #     self._bj_list.append(bj)
        #     self._vtxj_list.append(vtxj)
        #     self._eltj_list.append(eltj)
        
        # self._g = g_vector(self._factorizations, self._bj_list, self._Bj_list, self._Cj_list, self._nx, self._J)
        self._mesh = FullMesh(self._J, self._params.nx, self._params.ny, self._params.Lx, self._params.Ly)
        self._boundary = FullBoundary(self._J, self._params.nx, self._mesh)

        self._B = FullBOperator(self._J, self._mesh, self._boundary)
        self._Q = FullCOperator(self._J, self._mesh)
        self._T = FullTOperator(self._J, self._mesh, self._boundary, self._B, self._params)
        self._A = FullAOperator(self._J, self._mesh, self._boundary, self._params)
        self._s_factorization = FullSFactorization(self._J, self._A, self._T, self._B)
        self._BVec = FullBVecOperator(self._J, self._mesh, self._params)
        

        self._mesh.build()
        self._boundary.build()

        self._B.build()
        self._Q.build()
        self._A.build()
        self._T.build()
        self._s_factorization.build()
        self._BVec.build()
    
        self._g = FullGVecOperator().applyGlobal(self._s_factorization, self._BVec, self._B, self._Q, self._params.nx, self._J)

        self._S = FullSOperator(self._J, self._s_factorization, self._B, self._T, self._Q)
        self._Pi = FullPiOperator(self._J, self._params.nx)
        
        # return factorizations, Bj_list, Cj_list, Tj_list, bj_list, vtxj_list, eltj_list, g
    
    def solve(self, callback = None, callback_type: str = "pr_norm"):
        S = FullSOperator(self._J, self._s_factorization, self._B, self._T, self._Q)
        # def S_op(x):
        #     return S_operator(x, self._factorizations, self._Bj_list, self._Tj_list, self._Cj_list)
        
        Pi = FullPiOperator(self._J, self._params.nx)
        # def Pi_op(x):
        #     return Pi_operator(x, self._nx, self._J)
        
        def matvec(x):
            return x + Pi.applyGlobal(S.applyGlobal(x))
        
        n_skeleton = len(self._g)
        self._A_op = spla.LinearOperator((n_skeleton, n_skeleton), matvec=matvec, dtype=complex)
        
        self._residuals = []
        if callback == None:
            def _callback(rk):
                self._residuals.append(rk)
            callback = _callback
        
        x, info = spla.gmres(self._A_op, -self._g, rtol=1e-10, callback=callback, 
                            callback_type=callback_type, maxiter=500)
        
        self._x = x
        return x, self._residuals, info, S, Pi
    
    def getX(self):
        return self._x
    
    def getComponents(self):
        return (self._s_factorization, self._B, self._Q, self._T,
                self._BVec, self._mesh, self._g, self._S, self._Pi)
    
    def getIterationMatrix(self):
        return self._A_op