import numpy as np
from scipy.sparse import csr_matrix

from src.seq.operators.base_operators import FullBlockDiagOperator
from src.seq.operators.b_operator import BOperator
from src.seq.operators.t_operator import TOperator

def uj_solution(xj: np.ndarray, LU_j, Bj: csr_matrix, 
                Tj: csr_matrix, bj: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full local solution u_j using the converged interface data x_j.

    This function acts as the final step of the DDM solver. Once the interface 
    variables x_j (incoming impedance traces) are found, they are used as 
    boundary conditions to solve the local volumetric problem one last time.

    Mathematical Formulation
    ------------------------
    The local system being solved is:
        A_{local} u_j = b_j + B_j^T T_j x_j

    Where:
        u_j = A_{local}^{-1} (b_j + B_j^T T_j x_j)

    Parameters
    ----------
    xj : np.ndarray
        The converged interface data (incoming Robin traces) for this subdomain.
    LU_j : scipy.sparse.linalg.SuperLU
        The pre-computed LU factorization of the local operator 
        (A_j + B_j^T T_j B_j).
    Bj : scipy.sparse.csr_matrix
        The restriction matrix mapping volume DOFs to boundary DOFs.
    Tj : scipy.sparse.csr_matrix
        The transmission (impedance) matrix.
    bj : np.ndarray
        The original local volume source vector (right-hand side).

    Returns
    -------
    np.ndarray
        The fully reconstructed solution vector u_j on the subdomain mesh.
    """
    
    # Map interface data back to volume source terms
    rhs = bj + Bj.T @ (Tj @ xj)
    
    # Solve the local volume problem
    uj = LU_j.solve(rhs)
    
    return uj

class UOperator[T](FullBlockDiagOperator[T]):
    def __init__(self, num_blocks: int, B: BOperator[T], T: TOperator[T], b: np.ndarray, LU):
        super(UOperator, self).__init__(num_blocks)
        self._T = T
        self._B = B
        self._b = b
        self._LU = LU

        self.T = None


    def applyLocal(self, j: int, xj: np.ndarray) -> np.ndarray:
        # Map interface data back to volume source terms
        # bj = ... slice in some way
        rhs = self._bj + self._B.T.applyLocal(j, self._T.applyLocal(j, xj))
        
        # Solve the local volume problem
        #uj = self._LU_j.solve(rhs)
        uj = self._LU.applyLocal(rhs)
        return uj

    def applyGlobal(self, x: np.ndarray) -> np.ndarray:

        rhs = self._b + self._B.T.applyGlobal(self._T.applyGlobal(x))

        # uj = self._LU.solve(rhs)
        uj = self._LU.applyGlobal(rhs)
        return uj