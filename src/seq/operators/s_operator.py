import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse.linalg as spla

from src.common.operators.operators import SFactorization, SOperator
from src.seq.operators.base_operators import FullBlockDiagOperator
from src.seq.operators.t_operator import FullTOperator
from src.seq.operators.b_operator import FullBOperator
from src.seq.operators.c_operator import FullCOperator
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
            self.setBlock(j, spla.splu(csc_matrix(modified_Aj)))
            

    def buildLocal(self, j: int):
        # Construct modified local matrix: Aj - i * Bj^T @ Tj @ Bj
        Aj = self._A.getBlock(j)
        Bj = self._B.getBlock(j)
        if Bj.shape[0] > 0: # If there are artificial interfaces        # type: ignore
            Tj = self._T.getBlock(j)
            modified_Aj = Aj - 1j * (Bj.T @ Tj @ Bj) 
        else:
            modified_Aj = Aj # No modification needed
        
        # LU factorization
        self.setBlock(j, spla.splu(csc_matrix(modified_Aj)))


    def applyLocal(self, j: int, xj: np.ndarray) -> np.ndarray:
        return self._block_list[j].solve(xj)  
    
    def applyGlobal(self, x: np.ndarray) -> np.ndarray:
        res = np.zeros(0)
        accumulate_cols = 0
        for j in range(self._num_blocks):

            res = np.concatenate([res, 
                                 self.applyLocal(j, 
                                                 x[accumulate_cols:accumulate_cols+self._shapes[j][1]]
                                                )]
                                )
            accumulate_cols += self._shapes[j][1]
        return res


class FullSOperator[T](SOperator, FullBlockDiagOperator[T]):
    def __init__(self, num_blocks: int, s_fact: FullSFactorization, B: FullBOperator, 
                 T: FullTOperator, Q: FullCOperator):
        super(SOperator, self).__init__(num_blocks=num_blocks)
        self._s_factorization = s_fact
        self._B = B
        self._T = T
        self._Q = Q

        self.T = None

    def applyGlobal(self, x: np.ndarray) -> np.ndarray:
        
        Sx = x.astype(complex).copy() 
        
        for j in range(self._num_blocks):
            # Extract local interface portion
            xj = self._Q.applyLocal(j, x)
            
            # Compute RHS: Tj @ xj (T includes kappa)
            rhs = self._T.applyLocal(j, xj)
            
            # Solve local problem
            local_sol = self._s_factorization.applyLocal(j, self._B.T.applyLocal(j, rhs))
            
            # Extract trace: Bj @ local_sol
            trace_u = self._B.applyLocal(j, local_sol)
            
            correction = 2j * trace_u
            
            # Add to global vector
            Sx += self._Q.T.applyLocal(j, correction)
        
        return Sx
