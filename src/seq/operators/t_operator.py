import numpy as np
from scipy.sparse import csr_matrix

from src.seq.operators.base_operators import FullBlockDiagOperator
from src.common.helmholtz.system_assembly import mass

def Tj_matrix(vtxj: np.ndarray, beltj_artf: np.ndarray, 
              Bj: csr_matrix, kappa: float) -> csr_matrix:
    """
    Construct local transmission matrix Tj.
    
    Tj = κ * (mass matrix on artificial interfaces)
    
    Parameters:
    -----------
    vtxj : ndarray
        Local vertex coordinates
    beltj_artf : ndarray
        Artificial boundary edges
    Bj : sparse matrix
        Interface restriction matrix
    kappa : float
        Wavenumber k
    
    Returns:
    --------
    Tj : sparse matrix
        Transmission matrix on Σj
    """
    if len(beltj_artf) == 0:
        # No artificial interfaces
        return csr_matrix((0, 0))
    
    # Build mass matrix on artificial interfaces
    M_interface = mass(vtxj, beltj_artf)
    
    # Restrict to interface DOFs: Tj = κ * Bj @ M_interface @ Bj^T
    Tj = kappa * (Bj @ M_interface @ Bj.T)
    
    return csr_matrix(Tj)

class TOperator[T](FullBlockDiagOperator):
    def __init__(self, num_blocks: int):
        super(TOperator, self).__init__(num_blocks)