#!/usr/bin/env python3
import numpy as np
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

# Add console handler so you see output in terminal too
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logger = logging.getLogger()

try:
    from src.helmholtz import (local_mesh, local_boundary, Bj_matrix, Cj_matrix,
                               Aj_matrix, Tj_matrix, Sj_factorization, bj_vector,
                               S_operator, Pi_operator, g_vector, 
                               fixed_point_solver, uj_solution)
except ImportError as e:
    logger.error(f"Could not import src.helmholtz: {e}")
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

    factorizations = []
    Bj_list = []
    Cj_list = []
    Tj_list = []
    bj_list = []
    
    sp = [np.array([0.5, 1.0, 1.0])]

    for j in range(J):
        vtxj, eltj = local_mesh(Lx, Ly, nx_global, ny_global, j, J)
        nx_local = nx_global
        ny_local = len(np.unique(vtxj[:, 1]))
        beltj_phys, beltj_artf = local_boundary(nx_local, ny_local, j, J)
        
        Bj = Bj_matrix(nx_local, ny_local, j, J, beltj_artf)
        Cj = Cj_matrix(nx_global, ny_global, j, J)
        Aj = Aj_matrix(vtxj, eltj, beltj_phys, kappa)
        Tj = Tj_matrix(vtxj, beltj_artf, Bj, kappa)
        
        LU = Sj_factorization(Aj, Tj, Bj)
        bj = bj_vector(vtxj, eltj, sp, kappa)
        
        factorizations.append(LU)
        Bj_list.append(Bj)
        Cj_list.append(Cj)
        Tj_list.append(Tj)
        bj_list.append(bj)

    # --- Generate Global RHS ---
    g = g_vector(factorizations, bj_list, Bj_list, Cj_list, nx_global, J)
    
    # Double check size one last time
    expected_size = 2 * (J - 1) * nx_global
    if g.shape[0] != expected_size:
        logger.error(f"Size Mismatch! Got {g.shape[0]}, Expected {expected_size}")
        return

    # --- Run Solver ---
    def S_op(x):
        return S_operator(x, factorizations, Bj_list, Tj_list, Cj_list)

    def Pi_op(x):
        return Pi_operator(x, nx_global, J)

    omega = 0.1
    max_iter = 200
    tol = 1e-6
    
    logger.info("Starting Fixed Point Solver...")
    x_sol, res, converged = fixed_point_solver(-g, S_op, Pi_op, omega, max_iter, tol)
    
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