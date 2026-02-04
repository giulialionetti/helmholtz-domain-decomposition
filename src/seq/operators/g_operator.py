import numpy as np
from src.common.operators.operators import GVecOperator
from src.seq.operators.s_operator import FullSFactorization
from src.seq.operators.b_operator import FullBOperator
from src.seq.operators.c_operator import FullCOperator
from src.seq.operators.pi_operator import FullPiOperator
from src.seq.operators.bv_operator import FullBVecOperator


class FullGVecOperator[T](GVecOperator):
    def __init__(self):
        super(FullGVecOperator, self).__init__()

    def applyGlobal(self, s_factorization: FullSFactorization, BVec: FullBVecOperator, B : FullBOperator, 
                    Q: FullCOperator, nx: int, J: int) -> np.ndarray:
        """
        Construct the global right-hand side vector 'g' for the interface linear system.

        In the DDM formulation, 'g' represents the contribution of the physical 
        volume sources (the RHS of the Helmholtz equation) projected onto the 
        subdomain interfaces.

        Mathematical Definition
        -----------------------
        g = Π ( sum_j C_j^T B_j A_j^{-1} b_j )

        Step-by-step construction:
        1. Solve local problems with zero interface input but full volume source:
        u_j^{local} = (A_{local})^{-1} b_j
        2. Restrict solution to the boundary:
        v_j = B_j u_j^{local}
        3. Assemble into global vector:
        v_{global} = sum (C_j^T v_j)
        4. Apply exchange operator:
        g = Π v_{global}

        Parameters
        ----------
        factorizations : list[scipy.sparse.linalg.SuperLU]
            List of pre-computed LU factorizations for each subdomain operator.
        bj_list : list[np.ndarray]
            List of local volume source vectors (b_j) for each subdomain.
        Bj_list : list[scipy.sparse.csr_matrix]
            List of restriction matrices mapping volume DOFs to boundary DOFs.
        Cj_list : list[scipy.sparse.csr_matrix]
            List of boolean matrices mapping local boundary DOFs to the global 
            interface vector.
        nx : int
            Number of grid points in the x-direction (defines interface size).
        J : int
            Total number of subdomains.

        Returns
        -------
        g : np.ndarray
            The global interface source vector (size: 2 * (J-1) * nx).
        """
        # Determine skeleton size: (J-1) interfaces, 2 sides per interface
        n_skeleton = 2 * (J - 1) * nx
        g_temp = np.zeros(n_skeleton, dtype=complex)
        
        for j in range(J):
            # 1. Solve local problem driven ONLY by volume source bj
            #    (Aj - i B_j^T Tj Bj) uj = bj
            uj = s_factorization.applyLocal(j, BVec.getBlock(j))
            
            # 2. Extract trace on the boundary: Bj @ uj
            interface_vals = B.applyLocal(j,uj)
            
            # 3. Map local boundary values to global position
            g_temp += Q.T.applyLocal(j, interface_vals)
        
        # 4. Apply exchange operator to finalize g
        g = FullPiOperator(J, nx).applyGlobal(g_temp)
        
        return -2*1j*g