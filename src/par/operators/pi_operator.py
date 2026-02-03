import numpy as np
from mpi4py import MPI
from src.common.operators.operators import PiOperator

class SparsePiOperator(PiOperator):
    def __init__(self, J: int, comm: MPI.Intercomm, nx: int):
        super(SparsePiOperator, self).__init__()
        self._J = J
        self._comm = comm
        self._nx = nx

    def applyGlobal(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the exchange operator Π to vector x.
        
        For every interface k:
        - Swaps Block 2k (Side 0) with Block 2k+1 (Side 1).
        """
        # Px = np.zeros_like(x)
        # n_interfaces = self._J - 1
        
        # for k in range(n_interfaces):
        #     # Indices for Side 0 (belonging to subdomain below)
        #     idx_side0 = slice((2 * k) * self._nx, (2 * k + 1) * self._nx)
            
        #     # Indices for Side 1 (belonging to subdomain above)
        #     idx_side1 = slice((2 * k + 1) * self._nx, (2 * k + 2) * self._nx)
            
        #     # Perform Swap
        #     Px[idx_side0] = x[idx_side1]
        #     Px[idx_side1] = x[idx_side0]
        # rank = self._comm.Get_rank()
        # if rank == 0:
        #     # Exchange with processor 1
            
        # elif rank == self._J - 1:
        #     # Exchange with processor J - 2

        # else:
        #     # Exchange with processor rank - 1 and rank + 1

        # return Px
        return np.zeros(0)
