import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse.linalg as spla

from src.seq.operators.base_operators import FullBlockDiagOperator
from src.seq.operators.t_operator import TOperator
from src.seq.operators.b_operator import BOperator
from src.seq.operators.c_operator import QOperator

def Sj_factorization(Aj: csr_matrix, Tj: csr_matrix, Bj: csr_matrix):
    """
    Factorize the local problem matrix Aj - iB*j Tj Bj.
    
    Parameters:
    -----------
    Aj : sparse matrix
        Local problem matrix
    Tj : sparse matrix
        Transmission matrix
    Bj : sparse matrix
        Interface restriction matrix
    
    Returns:
    --------
    LU : SuperLU object
        LU factorization for efficient solves
    """
    # Construct modified local matrix: Aj - i * Bj^T @ Tj @ Bj
    if Bj.shape[0] > 0: # If there are artificial interfaces        # type: ignore
        modified_Aj = Aj - 1j * (Bj.T @ Tj @ Bj) 
    else:
        modified_Aj = Aj # No modification needed
    
    # LU factorization
    LU = spla.splu(csc_matrix(modified_Aj))
    
    return LU

class SFactorization[T](FullBlockDiagOperator[T]):
    def __init__(self, num_blocks: int):
        super(SFactorization, self).__init__(num_blocks)

    def applyLocal(self, j: int, xj: np.ndarray) -> np.ndarray:
        return self._block_list[j].solve(xj)  
    
    def applyGlobal(self, x: np.ndarray) -> np.ndarray:
        res = np.zeros((0,0))
        accumulate_cols = 0
        for j in range(self._num_blocks):

            res = np.concatenate(res, 
                                 self.applyLocal(j, 
                                                 x[accumulate_cols:accumulate_cols+self._shapes[j][1]]
                                                )
                                )
            accumulate_cols += self._shapes[j][1]
        return res

def S_operator(x: np.ndarray, s_factorization: SFactorization, B: BOperator, 
               T: TOperator, Q: QOperator) -> np.ndarray:
    """
    Apply the global Schur complement operator S to vector x.
    
    S is block diagonal: S = diag(S1, S2, ..., SJ)
    where Sj xj = Bj (Aj - iB*j Tj Bj)^(-1) B*j Tj xj
    
    Parameters:
    -----------
    x : ndarray
        Global interface vector
    factorizations : list
        List of LU factorizations for each subdomain
    Bj_list : list
        List of Bj matrices
    Tj_list : list
        List of Tj matrices
    Cj_list : list
        List of Cj matrices
    
    Returns:
    --------
    Sx : ndarray
        Result of S @ x
    """
    J = s_factorization.getNumBlocks()
    Sx = np.zeros_like(x)
    
    for j in range(J):
        # Extract local interface portion: xj = Cj @ x
        xj = Q.applyLocal(j, x)
        
        # Compute: Tj @ xj
        rhs = T.applyLocal(j, xj)
        
        # Solve: (Aj - iB*j Tj Bj)^(-1) @ (B*j @ Tj @ xj)
        local_sol = s_factorization.applyLocal(j, B.T.applyLocal(j, rhs))
        
        # Apply: Bj @ local_sol
        Sj_xj = B.applyLocal(j, local_sol)
        
        # Assemble back to global skeleton: C*j @ Sj_xj
        Sx += Q.T.applyLocal(j,Sj_xj)
    
    return Sx

class SOperator[T](FullBlockDiagOperator[T]):
    def __init__(self, num_blocks: int):
        super(SOperator, self).__init__(num_blocks)
