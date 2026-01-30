import numpy as np
from scipy.sparse import csr_matrix

from src.common.helmholtz.system_assembly import mass, stiffness
from src.seq.operators.base_operators import FullBlockDiagOperator


def Aj_matrix(vtxj: np.ndarray, eltj: np.ndarray, 
              beltj_phys: np.ndarray, kappa: float) -> csr_matrix:
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
    
    Returns:
    --------
    Aj : sparse matrix
        Local Helmholtz operator K - k²M - ikMb
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
    
    return csr_matrix(Aj)
    
class AOperator[T](FullBlockDiagOperator[T]):
    def __init__(self, num_blocks):
        super(AOperator, self).__init__(num_blocks)