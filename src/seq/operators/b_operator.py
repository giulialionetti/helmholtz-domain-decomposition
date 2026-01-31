import numpy as np
from scipy.sparse import csr_matrix

from src.seq.operators.base_operators import FullBlockDiagOperator
from src.seq.mesh.mesh import FullMesh, FullBoundary
from src.common.operators.operators import BOperator


class FullBOperator[T](BOperator, FullBlockDiagOperator[T]):
    def __init__(self, num_blocks: int, mesh: FullMesh, boundary: FullBoundary):
        super(BOperator, self).__init__(mesh=mesh, boundary=boundary, num_blocks=num_blocks)
        self._mesh = mesh
        self._boundary = boundary

    def build(self):
        
        for j in range(self._num_blocks):
            nx_local = self._mesh.getNxLocal(j)
            ny_local = self._mesh.getNyLocal(j)

            nv_local = nx_local * ny_local
            
            beltj_artf = self._boundary.getLocal(j)[1]

            # if len(beltj_artf) == 0:
            #     return csr_matrix((0, nv_local))
            
            # Extract unique vertex indices on artificial interfaces
            interface_vertices = np.unique(beltj_artf.flatten())
            n_interface = len(interface_vertices)
            
            # Build restriction matrix
            row_indices = np.arange(n_interface)
            col_indices = interface_vertices
            data = np.ones(n_interface)
            
            Bj = csr_matrix((data, (row_indices, col_indices)), shape=(n_interface, nv_local))
            
            self._block_list[j] = Bj

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
        
        # if len(beltj_artf) == 0:
        #     return csr_matrix((0, nv_local))
        
        # Extract unique vertex indices on artificial interfaces
        interface_vertices = np.unique(beltj_artf.flatten())
        n_interface = len(interface_vertices)
        
        # Build restriction matrix
        row_indices = np.arange(n_interface)
        col_indices = interface_vertices
        data = np.ones(n_interface)
        
        Bj = csr_matrix((data, (row_indices, col_indices)), shape=(n_interface, nv_local))
        
        self._block_list[j] = Bj
