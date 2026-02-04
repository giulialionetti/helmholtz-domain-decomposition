import numpy as np
from scipy.sparse import csr_matrix

from src.common.helmholtz.helmholtz_param import HelmholtzParameters
from src.common.helmholtz.system_assembly import mass
from src.common.operators.operators import TOperator
from src.seq.mesh.mesh import FullMesh, FullBoundary
from src.seq.operators.base_operators import FullBlockDiagOperator
from src.seq.operators.b_operator import FullBOperator


class FullTOperator[T](TOperator, FullBlockDiagOperator):
    def __init__(self, num_blocks: int, mesh: FullMesh, boundary: FullBoundary, 
                 B: FullBOperator, params: HelmholtzParameters):
        super(FullTOperator, self).__init__(mesh=mesh, boundary=boundary, B=B, params=params, num_blocks=num_blocks)

    def build(self):
        for j in range(self._num_blocks):
            vtxj = self._mesh.getLocal(j)[0]
            beltj_artf = self._boundary.getLocal(j)[1]
            if len(beltj_artf) == 0:
                # No artificial interfaces
                self.setBlock(j, csr_matrix((0,0)))
                continue
            
            # Build mass matrix on artificial interfaces
            M_interface = mass(vtxj, beltj_artf)
            
            # Restrict to interface DOFs: Tj = κ * Bj @ M_interface @ Bj^T
            Bj = self._B.getBlock(j)
            Tj = self._params.kappa * (Bj @ M_interface @ Bj.T)
            
            # self._block_list[j] = csr_matrix(Tj)
            self.setBlock(j, csr_matrix(Tj))

    def buildLocal(self, j: int):
        vtxj = self._mesh.getLocal(j)[0]
        beltj_artf = self._boundary.getLocal(j)[1]
        if len(beltj_artf) == 0:
            # No artificial interfaces
            self.setBlock(j, csr_matrix((0,0)))
            return
        
        # Build mass matrix on artificial interfaces
        M_interface = mass(vtxj, beltj_artf)
        
        # Restrict to interface DOFs: Tj = κ * Bj @ M_interface @ Bj^T
        Bj = self._B.getBlock(j)
        Tj = self._params.kappa * (Bj @ M_interface @ Bj.T)
        
        # self._block_list[j] = csr_matrix(Tj)
        self.setBlock(j, csr_matrix(Tj))

    
    def applyGlobalNorm(self, v):
        return np.linalg.norm(np.dot(self.applyGlobal(v), v))
