import numpy as np
from scipy.sparse import csr_matrix
from src.seq.operators.base_operators import FullBlockDiagOperator


class BOperator[T](FullBlockDiagOperator[T]):
    def __init__(self, J: int):
        super(BOperator, self).__init__(J)

    def buildLocal(self, j: int, nx: int, ny: int, beltj_artf: np.ndarray):
        r"""
        Construct interface restriction matrix Bj.
        
        Bj maps local degrees of freedom V(Ωj) to interface DOFs V(Σj).
        
        Parameters:
        -----------
        nx, ny : int
            Number of points for LOCAL mesh
        j : int
            Subdomain index
        J : int
            Total number of subdomains
        beltj_artf : ndarray
            Artificial boundary edges
        
        Notes:
        ------
        - Σj = ∂Ωj \ ∂Ω (artificial interfaces only)
        - This extracts interface DOFs from the local solution
        """
        nv_local = nx * ny
        
        if len(beltj_artf) == 0:
            return csr_matrix((0, nv_local))
        
        # Extract unique vertex indices on artificial interfaces
        interface_vertices = np.unique(beltj_artf.flatten())
        n_interface = len(interface_vertices)
        
        # Build restriction matrix
        row_indices = np.arange(n_interface)
        col_indices = interface_vertices
        data = np.ones(n_interface)
        
        Bj = csr_matrix((data, (row_indices, col_indices)), shape=(n_interface, nv_local))
        
        self._block_list[j] = Bj
