import numpy as np
from src.common.helmholtz.system_assembly import mass, point_source
from src.seq.operators.base_operators import FullBlockDiagOperator

class BVecOperator[T](FullBlockDiagOperator[T]):
    def __init__(self, num_blocks: int):
        super(BVecOperator, self).__init__(num_blocks)
    
    def buildLocal(self, j: int, vtxj: np.ndarray, eltj: np.ndarray, sp: list, kappa: float):
        """
        Construct local right-hand side vector bj.
        
        Parameters:
        -----------
        vtxj : ndarray
            Local vertex coordinates
        eltj : ndarray
            Local triangle connectivity
        sp : list
            Point source specifications
        kappa : float
            Wavenumber k
        """
        # Build mass matrix
        M = mass(vtxj, eltj)
        
        # Evaluate point sources at local vertices
        f = point_source(sp, kappa)(vtxj)
        
        # RHS: bj = M @ f
        bj = M @ f

        self._block_list[j] = bj

    def buildGlobal(self, sp: list, kappa: float):
        pass
        
