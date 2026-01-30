import numpy as np
from scipy.sparse import csr_matrix

from src.seq.operators.base_operators import FullRowBlockOperator

def Rj_matrix(nx: int, ny: int, j: int, J: int) -> csr_matrix:
    r"""
    Construct local restriction matrix Rj.
    
    Rj maps global degrees of freedom V(Ω) to local degrees of freedom V(Ωj).
    
    Parameters:
    -----------
    nx, ny : int
        Number of points for GLOBAL mesh
    j : int
        Subdomain index
    J : int
        Total number of subdomains
    
    Returns:
    --------
    Rj : sparse matrix
        Restriction matrix of size |V(Ωj)| X |V(Ω)|
    
    Notes:
    ------
    - This is a boolean matrix (0s and 1s)
    - Each row has exactly one 1, selecting one global DOF
    - The ordering follows the local mesh numbering
    """
    # Number of y-intervals per subdomain
    ny_per_subdomain = (ny - 1) // J
    
    # Local number of points
    ny_local = ny_per_subdomain + 1
    nv_local = nx * ny_local
    nv_global = nx * ny
    
    # Starting y-index for subdomain j in global mesh
    y_start = j * ny_per_subdomain
    
    # Build mapping from local to global indices
    row_indices = []
    col_indices = []
    
    for iy_local in range(ny_local):
        iy_global = y_start + iy_local
        for ix in range(nx):
            local_idx = ix + iy_local * nx
            global_idx = ix + iy_global * nx
            row_indices.append(local_idx)
            col_indices.append(global_idx)
    
    # Create sparse restriction matrix
    data = np.ones(len(row_indices))
    Rj = csr_matrix((data, (row_indices, col_indices)), shape=(nv_local, nv_global))
    
    return Rj

# CHECK IF IT IS FullRowBlockOperator or FullRowQuasiDiagBlockOperator
# class ROperator[T](FullRowBlockOperator[T]):
#     def __init__(self, num_blocks: int):
#         super(ROperator, self).__init__(num_blocks)