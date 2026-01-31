import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse.linalg as spla

from src.common.operators.operators import SFactorization, SOperator
from src.seq.operators.base_operators import FullBlockDiagOperator
from src.seq.operators.t_operator import FullTOperator
from src.seq.operators.b_operator import FullBOperator
from src.seq.operators.q_operator import FullQOperator
from src.seq.operators.a_operator import FullAOperator

class FullSFactorization[T](SFactorization[T], FullBlockDiagOperator[T]):
    def __init__(self, num_blocks: int, A: FullAOperator, T: FullTOperator, B: FullBOperator):
        super(SFactorization, self).__init__(A=A, T=T, B=B, num_blocks=num_blocks)
        self._A = A
        self._T = T
        self._B = B

    def build(self):
        for j in range(self._num_blocks):
            # Construct modified local matrix: Aj - i * Bj^T @ Tj @ Bj
            Aj = self._A.getBlock(j)
            Bj = self._B.getBlock(j)
            if Bj.shape[0] > 0: # If there are artificial interfaces        # type: ignore
                Tj = self._T.getBlock(j)
                modified_Aj = Aj - 1j * (Bj.T @ Tj @ Bj) 
            else:
                modified_Aj = Aj # No modification needed
            
            # LU factorization
            self._block_list[j] = spla.splu(csc_matrix(modified_Aj))
            

    def buildLocal(self, j: int, Aj: T, Tj: T, Bj: T):
        # Construct modified local matrix: Aj - i * Bj^T @ Tj @ Bj
        if Bj.shape[0] > 0: # If there are artificial interfaces        # type: ignore
            modified_Aj = Aj - 1j * (Bj.T @ Tj @ Bj) 
        else:
            modified_Aj = Aj # No modification needed
        
        # LU factorization
        self._block_list[j] = spla.splu(csc_matrix(modified_Aj))


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


class FullSOperator[T](SOperator, FullBlockDiagOperator[T]):
    def __init__(self, num_blocks: int, s_fact: FullSFactorization, B: FullBOperator, 
                 T: FullTOperator, Q: FullQOperator):
        super(SOperator, self).__init__(num_blocks=num_blocks)
        self._s_factorization = s_fact
        self._B = B
        self._T = T
        self._Q = Q

        self.T = None

    def applyGlobal(self, x: np.ndarray) -> np.ndarray:
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
        Sx = np.zeros_like(x)
        
        for j in range(self._num_blocks):
            # Extract local interface portion: xj = Cj @ x
            xj = self._Q.applyLocal(j, x)
            
            # Compute: Tj @ xj
            rhs = self._T.applyLocal(j, xj)
            
            # Solve: (Aj - iB*j Tj Bj)^(-1) @ (B*j @ Tj @ xj)
            local_sol = self._s_factorization.applyLocal(j, self._B.T.applyLocal(j, rhs))
            
            # Apply: Bj @ local_sol
            Sj_xj = self._B.applyLocal(j, local_sol)
            
            # Assemble back to global skeleton: C*j @ Sj_xj
            Sx += self._Q.T.applyLocal(j,Sj_xj)
        
        return Sx
