import numpy as np
from mpi4py import MPI
from src.par.operators.s_operator import SparseSOperator
from src.par.operators.pi_operator import SparsePiOperator

class MPILinearOperator:
    """
    Wrapper for MPI-distributed linear operator to use with scipy.sparse.linalg.gmres.
    
    Implements the operator: (I + Π S)
    """
    
    def __init__(self, comm: MPI.Comm, S_op: SparseSOperator, Pi_op: SparsePiOperator, 
                 local_size: int):
        """
        Parameters
        ----------
        comm : MPI.Comm
            MPI communicator
        S_op : SparseSOperator
            Schur complement operator
        Pi_op : SparsePiOperator
            Exchange operator
        local_size : int
            Size of local vector portion
        """
        self.comm = comm
        self.S_op = S_op
        self.Pi_op = Pi_op
        self.local_size = local_size
        self.shape = (local_size, local_size)
        self.dtype = np.complex128
    
    def matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Compute (I + Π S) @ x in distributed fashion.
        
        Each process computes locally and communicates via Π.
        """
        Sx = self.S_op.applyGlobal(x)
        Pi_Sx = self.Pi_op.applyGlobal(Sx)
        return x + Pi_Sx
    
    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        return self.matvec(x)