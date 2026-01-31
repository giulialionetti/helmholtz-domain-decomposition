import numpy as np
from scipy.sparse import csr_matrix

from src.common.helmholtz.helmholtz_param import HelmholtzParameters
from src.common.operators.operators import AOperator
from src.common.helmholtz.system_assembly import mass, stiffness
from src.seq.operators.base_operators import FullBlockDiagOperator
from src.seq.mesh.mesh import FullBoundary, FullMesh

    
class FullAOperator[T](AOperator, FullBlockDiagOperator[T]):
    def __init__(self, num_blocks, mesh: FullMesh, boundary: FullBoundary, params: HelmholtzParameters):
        super(FullAOperator, self).__init__(mesh=mesh, boundary=boundary, params=params, num_blocks=num_blocks)

    def build(self):
        # Build local mass and stiffness matrices
        for j in range(self._num_blocks):
            vtxj, eltj = self._mesh.getLocal(j)
            beltj_phys = self._boundary.getLocal(j)[0]

            M = mass(vtxj, eltj)
            K = stiffness(vtxj, eltj)
            
            # Build boundary mass matrix (only on physical boundaries)
            if len(beltj_phys) > 0:
                Mb = mass(vtxj, beltj_phys)
            else:
                Mb = csr_matrix((len(vtxj), len(vtxj)))
            
            # Construct Helmholtz operator: A = K - k²M - ikMb
            Aj = K - self._params.kappa**2 * M - 1j * self._params.kappa * Mb
            
            self._block_list[j] = csr_matrix(Aj)


    def buildLocal(self, j:int, vtxj: np.ndarray, eltj: np.ndarray, 
                   beltj_phys: np.ndarray, kappa: float):
        """
        Construct local problem matrix Aj for subdomain j.
        
        Parameters:
        -----------
        vtxj : ndarray
            Local vertex coordinates
        eltj : ndarray
            Local triangle connectivity
        beltj_phys : ndarray
            Physical boundary edges
        kappa : float
            Wavenumber k
        """
        # Build local mass and stiffness matrices
        M = mass(vtxj, eltj)
        K = stiffness(vtxj, eltj)
        
        # Build boundary mass matrix (only on physical boundaries)
        if len(beltj_phys) > 0:
            Mb = mass(vtxj, beltj_phys)
        else:
            Mb = csr_matrix((len(vtxj), len(vtxj)))
        
        # Construct Helmholtz operator: A = K - k²M - ikMb
        Aj = K - kappa**2 * M - 1j * kappa * Mb
        
        self._block_list[j] = csr_matrix(Aj)
