import numpy as np
from mpi4py import MPI
from src.common.operators.operators import GVecOperator
from src.seq.operators.s_operator import FullSFactorization
from src.seq.operators.b_operator import FullBOperator
from src.seq.operators.c_operator import FullCOperator
from src.par.operators.pi_operator import SparsePiOperator
from src.seq.operators.bv_operator import FullBVecOperator


class SparseGVecOperator:
    """
    Construct global RHS vector g in MPI-distributed setting.
    
    Each process computes its local contribution:
    g_j = C_j^T B_j A_j^{-1} b_j
    
    Then applies Π exchange operator.
    """
    
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.rank = comm.Get_rank()
    
    def applyGlobal(self, s_factorization, BVec, B, C, nx: int, J: int) -> np.ndarray:
        """
        Construct local portion of global RHS vector g.
        
        Parameters
        ----------
        s_factorization : FullSFactorization
            Local S factorization
        BVec : FullBVecOperator
            Local RHS vector b_j
        B : FullBOperator
            Local restriction matrix B_j
        C : FullCOperator
            Local assembly matrix C_j
        nx : int
            Grid points per interface
        J : int
            Total subdomains
            
        Returns
        -------
        g_local : np.ndarray
            Local contribution to g after Π exchange
        """
        j = self.rank
        
        # 1. Solve local problem: A_j^{-1} b_j
        uj = s_factorization.applyLocal(j, BVec.getBlock(j))
        
        # 2. Extract boundary trace: B_j @ u_j
        interface_vals = B.applyLocal(j, uj)
        
        # 3. Map to global interface positions: C_j^T @ (B_j @ u_j)
        g_temp_local = C.T.applyLocal(j, interface_vals)
        
        # 4. Apply exchange operator Π
        # NOTE: In sparse setting, we need to communicate to apply Π correctly
        # Since Π swaps interface sides, we need to exchange with neighbors
        
        Pi_op = SparsePiOperator(self.comm, J, nx)
        g_local = Pi_op.applyGlobal(g_temp_local)
        
        return g_local