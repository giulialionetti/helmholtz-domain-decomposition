import numpy as np
from src.common.helmholtz.helmholtz_param import HelmholtzParameters
from src.common.helmholtz.system_assembly import mass, point_source
from src.seq.operators.base_operators import FullBlockDiagOperator
from src.common.operators.operators import BVecOperator
from src.seq.mesh.mesh import FullMesh

class FullBVecOperator[T](BVecOperator, FullBlockDiagOperator[T]):
    def __init__(self, num_blocks: int, mesh: FullMesh, params: HelmholtzParameters):
        super(BVecOperator, self).__init__(mesh=mesh, params=params, num_blocks=num_blocks)
        self._mesh = mesh
        self._params = params
    
    def build(self):
        for j in range(self._num_blocks):
            vtxj, eltj = self._mesh.getLocal(j)
            # Build mass matrix
            M = mass(vtxj, eltj)
            
            # Evaluate point sources at local vertices
            f = point_source(self._params.sp, self._params.kappa)(vtxj)
            
            # RHS: bj = M @ f
            bj = M @ f

            self._block_list[j] = bj


    def buildLocal(self, j: int):
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
        vtxj, eltj = self._mesh.getLocal(j)
        # Build mass matrix
        M = mass(vtxj, eltj)
        
        # Evaluate point sources at local vertices
        f = point_source(self._params.sp, self._params.kappa)(vtxj)
        
        # RHS: bj = M @ f
        bj = M @ f

        self._block_list[j] = bj

    def buildGlobal(self, sp: list, kappa: float):
        pass
        
