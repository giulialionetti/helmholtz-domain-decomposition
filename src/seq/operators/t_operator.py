import numpy as np
from scipy.sparse import csr_matrix

from src.seq.operators.base_operators import FullBlockDiagOperator
from src.common.helmholtz.system_assembly import mass
from src.common.operators.operators import TOperator
from src.seq.mesh.mesh import FullMesh, FullBoundary


class FullTOperator[T](TOperator, FullBlockDiagOperator):
    def __init__(self, num_blocks: int, mesh: FullMesh, boundary: FullBoundary):
        super(TOperator, self).__init__(mesh=mesh, boundary=boundary, num_blocks=num_blocks)

    def buildLocal(self, j: int, vtxj: np.ndarray, beltj_artf: np.ndarray, 
                   Bj: csr_matrix, kappa: float):
        if len(beltj_artf) == 0:
            # No artificial interfaces
            return csr_matrix((0, 0))
        
        # Build mass matrix on artificial interfaces
        M_interface = mass(vtxj, beltj_artf)
        
        # Restrict to interface DOFs: Tj = κ * Bj @ M_interface @ Bj^T
        Tj = kappa * (Bj @ M_interface @ Bj.T)
        
        self._block_list[j] = csr_matrix(Tj)
