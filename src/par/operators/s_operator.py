import numpy as np
from mpi4py import MPI
from src.seq.operators.b_operator import FullBOperator
from src.seq.operators.t_operator import FullTOperator
from src.seq.operators.c_operator import FullCOperator

class SparseSOperator:
    """
    Schur complement operator S in MPI-distributed setting.
    
    S is block-diagonal, so each process computes its local block Sj independently.
    The only communication needed is during the application to assemble global results.
    """
    
    def __init__(self, comm: MPI.Comm, J: int, s_factorization, B: FullBOperator, T: FullTOperator, C: FullCOperator):
        """
        Parameters
        ----------
        comm : MPI.Comm
            MPI communicator
        J : int
            Total number of subdomains
        s_factorization : FullSFactorization
            Local S-factorization (each rank has its own)
        B : FullBOperator
            Local B operator
        T : FullTOperator
            Local T operator
        C : FullCOperator
            Local C operator (maps local → global interface indices)
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.J = J
        
        assert self.size == J
        
        # Store local operators (each process has j = rank)
        self._s_fact = s_factorization
        self._B = B
        self._T = T
        self._C = C
    
    def applyGlobal(self, x_global: np.ndarray) -> np.ndarray:
        """
        Apply Schur complement S to a distributed global vector.
        
        Each process:
        1. Extracts its local portion via C_j
        2. Computes S_j @ x_j locally
        3. All processes send their contributions to be summed
        
        NOTE: In the MPI version, we can avoid explicit global assembly by
        having each process hold only its LOCAL interface portion.
        
        Parameters
        ----------
        x_global : np.ndarray
            Local interface vector (size depends on rank position)
            
        Returns
        -------
        Sx_local : np.ndarray
            Local result after applying S
        """
        j = self.rank
        
        # Extract local interface portion
        # In sparse version, x_global is already local to this process
        # so C.applyLocal just identifies which part of x_global belongs to j
        xj = self._C.applyLocal(j, x_global)
        
        # Compute: Tj @ xj
        rhs = self._T.applyLocal(j, xj)
        
        # Solve: (Aj - iB*j Tj Bj)^(-1) @ (B*j @ Tj @ xj)
        local_sol = self._s_fact.applyLocal(j, self._B.T.applyLocal(j, rhs))
        
        # Apply: Bj @ local_sol
        Sj_xj = self._B.applyLocal(j, local_sol)
        
        # Map back to global interface positions
        # In sparse setting, C.T.applyLocal returns the local contribution
        Sx_local = self._C.T.applyLocal(j, Sj_xj)
        
        return Sx_local
