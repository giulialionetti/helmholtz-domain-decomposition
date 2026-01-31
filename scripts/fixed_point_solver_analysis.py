#!/usr/bin/env python3
import numpy as np
from scipy.sparse import csr_matrix
import logging
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
while not os.path.exists(os.path.join(project_root, 'src')):
    parent = os.path.dirname(project_root)
    if parent == project_root: 
        # Fallback: assume typical structure if loop fails
        project_root = os.path.abspath(os.path.join(current_dir, "../../"))
        break
    project_root = parent

if project_root not in sys.path:
    sys.path.insert(0, project_root)


log_file = "fixed_point_convergence_test.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    filename=log_file,
    filemode='w'
)

logger = logging.getLogger()

try:
    from src.seq.operators.s_operator import FullSFactorization, FullSOperator
    from src.seq.operators.t_operator import FullTOperator
    from src.seq.operators.b_operator import FullBOperator
    from src.seq.operators.a_operator import FullAOperator
    from src.seq.operators.q_operator import FullQOperator
    from src.seq.operators.pi_operator import FullPiOperator
    from src.seq.operators.g_operator import FullGVecOperator
    from src.seq.operators.bv_operator import FullBVecOperator

    from src.seq.mesh.mesh import FullBoundary, FullMesh
    from src.common.mesh import local_boundary, local_mesh
    from src.seq.linear_solver.fixed_point import fixed_point_solver
except ImportError as e:
    logger.error(f"Could not import in file {__file__}: {e}")
    sys.exit(1)

def run_convergence_test():
    Lx, Ly = 1.0, 2.0
    kappa = 16.0
    
    # Mesh: ~10 points per wavelength (lambda ~ 0.4)
    nx_global = 33
    ny_global = 65 
    J = 4
    
    logger.info("="*60)
    logger.info(f" k={kappa}, Grid={nx_global}x{ny_global}")
    logger.info("="*60)

    mesh = FullMesh(J, nx_global, ny_global, Lx, Ly)
    boundary = FullBoundary(J, nx_global, mesh)

    s_factorization = FullSFactorization(J)
    B = FullBOperator[csr_matrix](J, mesh, boundary)
    Q = FullQOperator[csr_matrix](J, mesh)
    T = FullTOperator[csr_matrix](J, mesh, boundary)
    A = FullAOperator[csr_matrix](J, mesh, boundary)
    BVec = FullBVecOperator[np.ndarray](J)
    
    sp = [np.array([0.5, 1.0, 1.0])]

    mesh.build()
    boundary.build()


    for j in range(J):
        vtxj, eltj = mesh.getLocal(j)

        nx_local = nx_global
        ny_local = len(np.unique(vtxj[:, 1]))
        beltj_phys, beltj_artf = boundary.getLocal(j)
        
        B.buildLocal(j, nx_local, ny_local, beltj_artf)
        Q.buildLocal(j, nx_global, ny_global)
        A.buildLocal(j, vtxj, eltj, beltj_phys, kappa)
        T.buildLocal(j, vtxj, beltj_artf, B.getBlock(j), kappa)
        
        # LU = Sj_factorization(Aj, Tj, Bj)
        s_factorization.buildLocal(j, A.getBlock(j), T.getBlock(j), B.getBlock(j))
        BVec.buildLocal(j, vtxj, eltj, sp, kappa)

   
    g = FullGVecOperator().applyGlobal(s_factorization, BVec, B, Q, nx_global, J)
    expected_size = 2 * (J - 1) * nx_global
    if g.shape[0] != expected_size:
        logger.error(f"Size Mismatch! Got {g.shape[0]}, Expected {expected_size}")
        return

   
    # def S_op(x):
    #     return S_operator(x, s_factorization, B, T, Q)
    S = FullSOperator(J, s_factorization, B, T, Q)

    Pi_op = FullPiOperator(J, nx_global)

    omega = 0.1
    max_iter = 400
    tol = 1e-8 # double precision
    
    logger.info("Starting Fixed Point Solver...")
    x_sol, res, converged = fixed_point_solver(-g, S, Pi_op, omega, max_iter, tol)
    
    logger.info("-" * 30)
    logger.info(f"Iterations:     {len(res)}")
    logger.info(f"Final Residual: {res[-1]:.6e}")
    logger.info(f"Converged:      {converged}")
    logger.info("-" * 30)
    
    if converged:
        logger.info("SUCCESS: Solver converged!")
    else:
        logger.warning("WARNING: Solver did not reach tolerance (check k vs h).")

if __name__ == "__main__":
    run_convergence_test()