#!/usr/bin/env python3
import numpy as np
from scipy.sparse import csr_matrix
import logging
import sys
import os
import matplotlib.pyplot as plt
import imageio
from io import BytesIO

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
    from src.seq.operators.c_operator import FullCOperator
    from src.seq.operators.pi_operator import FullPiOperator
    from src.seq.operators.g_operator import FullGVecOperator
    from src.seq.operators.bv_operator import FullBVecOperator
    from src.seq.operators.u_operator import uj_solution

    from src.common.mesh import plot_mesh
    from src.seq.mesh.mesh import FullBoundary, FullMesh
    from src.common.helmholtz.helmholtz_param import HelmholtzParameters
    from src.seq.linear_solver.fixed_point import fixed_point_solver
except ImportError as e:
    logger.error(f"Could not import in file {__file__}: {e}")
    sys.exit(1)

plots_dir = os.path.join(project_root, 'plots')
os.makedirs(plots_dir, exist_ok=True)

class FixedPointConvergenceCallback:
    def __init__(self, T: FullTOperator, omega: float, deltas: list[float], true_p: np.ndarray):
        self._t_errors = []
        self._bounds = {}
        self._T = T
        self._omega = omega
        self._deltas = deltas
        self._taus = [np.sqrt(1 - omega*(1-omega)/(delta**2)) for delta in deltas]
        self._true_p = true_p

        for tau in self._taus:
            self._bounds[tau] = []

    def __call__(self, n: int, x: np.ndarray, residual: np.ndarray):
        self._t_errors.append(self._T.applyGlobalNorm(x-self._true_p))
        for tau in self._taus:
            self._bounds[tau].append((tau**n) * self._t_errors[0])

    def plot(self):
        plt.figure()
        plt.semilogy(self._t_errors, label="Error")
        for tau, delta in zip(self._taus, self._deltas):
            plt.semilogy(self._bounds[tau], label=f"δ={delta}")
        
        plt.xlabel('Iteration', fontsize=11)
        plt.ylabel('Error norm', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        plt.savefig(os.path.join(plots_dir, f'fixed_point_convergence_w{self._omega}.png'), dpi=150)



def run_convergence_test():
    Lx, Ly = 1.0, 2.0
    kappa = 16.0

    sp = [np.array([0.5, 1.0, 1.0])]
    params = HelmholtzParameters(Lx, Ly, kappa, sp=sp)
    
    # Mesh: ~10 points per wavelength (lambda ~ 0.4)
    nx_global = 33
    ny_global = 65 
    J = 4
    
    logger.info("="*60)
    logger.info(f" k={kappa}, Grid={nx_global}x{ny_global}")
    logger.info("="*60)

    mesh = FullMesh(J, nx_global, ny_global, Lx, Ly)
    boundary = FullBoundary(J, nx_global, mesh)

    B = FullBOperator[csr_matrix](J, mesh, boundary)
    Q = FullCOperator[csr_matrix](J, mesh)
    T = FullTOperator[csr_matrix](J, mesh, boundary, B, params)
    A = FullAOperator[csr_matrix](J, mesh, boundary, params)
    s_factorization = FullSFactorization(J, A, T, B)
    BVec = FullBVecOperator[np.ndarray](J, mesh, params)
    

    mesh.build()
    boundary.build()

    B.build()
    Q.build()
    A.build()
    T.build()
    s_factorization.build()
    BVec.build()
   
    g = FullGVecOperator().applyGlobal(s_factorization, BVec, B, Q, nx_global, J)
    expected_size = 2 * (J - 1) * nx_global
    if g.shape[0] != expected_size:
        logger.error(f"Size Mismatch! Got {g.shape[0]}, Expected {expected_size}")
        return

   
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

def show_theoretical_convergence(omega: float, deltas: list[float]):
    # ============================= PARAMETERS ================================
    Lx, Ly = 1.0, 2.0
    kappa = 16.0
    nx_global = 33
    ny_global = 65 
    J = 4
    logger.info("="*60)
    logger.info(f" k={kappa}, Grid={nx_global}x{ny_global}")
    logger.info("="*60)

    sp = [np.array([0.5, 1.0, 1.0])]
    params = HelmholtzParameters(Lx, Ly, kappa, sp=sp)
    
    # Mesh: ~10 points per wavelength (lambda ~ 0.4)
    
    # ====================== BUILDING MESH and OPERATORS =======================

    mesh = FullMesh(J, nx_global, ny_global, Lx, Ly)
    boundary = FullBoundary(J, nx_global, mesh)

    B = FullBOperator[csr_matrix](J, mesh, boundary)
    Q = FullCOperator[csr_matrix](J, mesh)
    T = FullTOperator[csr_matrix](J, mesh, boundary, B, params)
    A = FullAOperator[csr_matrix](J, mesh, boundary, params)
    s_factorization = FullSFactorization(J, A, T, B)
    BVec = FullBVecOperator[np.ndarray](J, mesh, params)
    

    mesh.build()
    boundary.build()

    B.build()
    Q.build()
    A.build()
    T.build()
    s_factorization.build()
    BVec.build()
   
    g = FullGVecOperator().applyGlobal(s_factorization, BVec, B, Q, nx_global, J)
    expected_size = 2 * (J - 1) * nx_global
    if g.shape[0] != expected_size:
        logger.error(f"Size Mismatch! Got {g.shape[0]}, Expected {expected_size}")
        return

   
    S = FullSOperator(J, s_factorization, B, T, Q)
    Pi_op = FullPiOperator(J, nx_global)

    # ================ APPLYING SUPER PRECISE FIXED POINT =======================
    max_iter = 1000
    tol = 1e-10 # double precision
    
    logger.info("Starting Fixed Point Solver...")
    x_sol_precise, res, converged = fixed_point_solver(-g, S, Pi_op, omega, max_iter, tol)
    
    logger.info("-" * 30)
    logger.info(f"Iterations:     {len(res)}")
    logger.info(f"Final Residual: {res[-1]:.6e}")
    logger.info(f"Converged:      {converged}")
    logger.info("-" * 30)
    
    if converged:
        logger.info("SUCCESS: Solver converged!")
    else:
        logger.warning("WARNING: Solver did not reach tolerance (check k vs h).")

    # ======================= APPLYING FIXED POINT ==============================

    callback = FixedPointConvergenceCallback(T, omega, deltas, x_sol_precise)

    max_iter = 400
    tol = 1e-8 # double precision
    
    logger.info("Starting Fixed Point Solver...")
    x_sol, res, converged = fixed_point_solver(-g, S, Pi_op, omega, max_iter, tol, callback=callback)
    
    callback.plot()

    logger.info("-" * 30)
    logger.info(f"Iterations:     {len(res)}")
    logger.info(f"Final Residual: {res[-1]:.6e}")
    logger.info(f"Converged:      {converged}")
    logger.info("-" * 30)
    
    if converged:
        logger.info("SUCCESS: Solver converged!")
    else:
        logger.warning("WARNING: Solver did not reach tolerance (check k vs h).")

def plot_solution_over_iterations(omega: float):
    # ============================= PARAMETERS ================================
    Lx, Ly = 1.0, 2.0
    kappa = 16.0
    nx_global = 33
    ny_global = 65 
    J = 4
    logger.info("="*60)
    logger.info(f" k={kappa}, Grid={nx_global}x{ny_global}")
    logger.info("="*60)

    sp = [np.array([0.5, 1.0, 1.0])]
    params = HelmholtzParameters(Lx, Ly, kappa, sp=sp)
    
    # Mesh: ~10 points per wavelength (lambda ~ 0.4)
    

    # ====================== BUILDING MESH and OPERATORS =======================

    mesh = FullMesh(J, nx_global, ny_global, Lx, Ly)
    boundary = FullBoundary(J, nx_global, mesh)

    B = FullBOperator[csr_matrix](J, mesh, boundary)
    Q = FullCOperator[csr_matrix](J, mesh)
    T = FullTOperator[csr_matrix](J, mesh, boundary, B, params)
    A = FullAOperator[csr_matrix](J, mesh, boundary, params)
    s_factorization = FullSFactorization(J, A, T, B)
    BVec = FullBVecOperator[np.ndarray](J, mesh, params)
    

    mesh.build()
    boundary.build()

    B.build()
    Q.build()
    A.build()
    T.build()
    s_factorization.build()
    BVec.build()
   
    g = FullGVecOperator().applyGlobal(s_factorization, BVec, B, Q, nx_global, J)
    expected_size = 2 * (J - 1) * nx_global
    if g.shape[0] != expected_size:
        logger.error(f"Size Mismatch! Got {g.shape[0]}, Expected {expected_size}")
        return

   
    S = FullSOperator(J, s_factorization, B, T, Q)
    Pi_op = FullPiOperator(J, nx_global)

    class FixedPointXCallback:
        def __init__(self, iter_limit : int = 20):
            self._iter_limit = iter_limit
            self._x_list = []
        def __call__(self, iter: int, x: np.ndarray, residuals: list):
            if iter % self._iter_limit == 0:
                self._x_list.append(x)

        def get_x_list(self):
            return self._x_list

    callback = FixedPointXCallback(1)

    max_iter = 400
    tol = 1e-8 # double precision
    
    logger.info("Starting Fixed Point Solver...")
    x_sol, res, converged = fixed_point_solver(-g, S, Pi_op, omega, max_iter, tol, callback=callback)

    # print(len(callback.get_x_list()))
    # exit(0)

    x_list = callback.get_x_list()[0:5]
    x_list.append(x_sol)
    uj_list_per_iter = {}
    u_per_iter = {}
    for i, x_iter in enumerate(x_list):
        uj_list_per_iter[i] = []
        u_per_iter[i] = np.zeros(0)
        for j in range(J):
            xj = Q.applyLocal(j, x_iter)
            uj = uj_solution(xj, s_factorization.getBlock(j), B.getBlock(j), T.getBlock(j), BVec.getBlock(j))
            uj_list_per_iter[i].append(uj)


    for i in range(len(x_list)):
        # fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig = plt.figure()
        # axes = axes.flatten()
        total_vtxj, total_eltj = np.zeros(0), np.zeros(0)
        for j in range(J):
            vtxj, eltj = mesh.getLocal(j)
            if total_vtxj.size == 0:
                total_vtxj = vtxj
                total_eltj = eltj
            else:
                total_vtxj = np.concatenate([total_vtxj, vtxj])
                total_eltj = np.concatenate([total_eltj, eltj])
            u_per_iter[i] = np.concatenate([u_per_iter[i],uj_list_per_iter[i][j]])
            
        plot_mesh(total_vtxj, total_eltj)
        tc = plt.tricontourf(total_vtxj[:, 0], total_vtxj[:, 1], 
                            np.abs(u_per_iter[i]), levels=20, cmap='viridis')
        plt.title(f'Full domain - Absolute value', fontsize=12, fontweight='bold')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.setaspect('equal')
        plt.colorbar(tc)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'fixed_point_iteration{i}_sp{sp[0][0]}-{sp[0][1]}.png'), dpi=150)

def plot_solution_over_iterations_gif(omega: float):
    # ============================= PARAMETERS ================================
    Lx, Ly = 1.0, 2.0
    kappa = 16.0
    nx_global = 33
    ny_global = 65 
    J = 4
    logger.info("="*60)
    logger.info(f" k={kappa}, Grid={nx_global}x{ny_global}")
    logger.info("="*60)

    sp = [np.array([0.5, 0.75 , 1.0])]
    params = HelmholtzParameters(Lx, Ly, kappa, sp=sp)
    
    # Mesh: ~10 points per wavelength (lambda ~ 0.4)
    

    # ====================== BUILDING MESH and OPERATORS =======================

    mesh = FullMesh(J, nx_global, ny_global, Lx, Ly)
    boundary = FullBoundary(J, nx_global, mesh)

    B = FullBOperator[csr_matrix](J, mesh, boundary)
    Q = FullCOperator[csr_matrix](J, mesh)
    T = FullTOperator[csr_matrix](J, mesh, boundary, B, params)
    A = FullAOperator[csr_matrix](J, mesh, boundary, params)
    s_factorization = FullSFactorization(J, A, T, B)
    BVec = FullBVecOperator[np.ndarray](J, mesh, params)
    

    mesh.build()
    boundary.build()

    B.build()
    Q.build()
    A.build()
    T.build()
    s_factorization.build()
    BVec.build()
   
    g = FullGVecOperator().applyGlobal(s_factorization, BVec, B, Q, nx_global, J)
    expected_size = 2 * (J - 1) * nx_global
    if g.shape[0] != expected_size:
        logger.error(f"Size Mismatch! Got {g.shape[0]}, Expected {expected_size}")
        return

   
    S = FullSOperator(J, s_factorization, B, T, Q)
    Pi_op = FullPiOperator(J, nx_global)

    class FixedPointXCallback:
        def __init__(self, iter_limit : int = 20):
            self._iter_limit = iter_limit
            self._x_list = []
        def __call__(self, iter: int, x: np.ndarray, residuals: list):
            if iter % self._iter_limit == 0:
                self._x_list.append(x)

        def get_x_list(self):
            return self._x_list

    callback = FixedPointXCallback(1)

    max_iter = 400
    tol = 1e-8 # double precision
    
    logger.info("Starting Fixed Point Solver...")
    x_sol, res, converged = fixed_point_solver(-g, S, Pi_op, omega, max_iter, tol, callback=callback)

    # print(len(callback.get_x_list()))
    # exit(0)

    iter_sample = 5

    x_list = callback.get_x_list()
    x_list_tmp = [x_list[i] for i in range(0, len(x_list), iter_sample)]
    x_list = x_list_tmp
    x_list.append(x_sol)
    uj_list_per_iter = {}
    u_per_iter = {}

    vmax = 0.0
    vmin = 0.0
    for i, x_iter in enumerate(x_list):
        uj_list_per_iter[i] = []
        u_per_iter[i] = np.zeros(0)
        for j in range(J):
            xj = Q.applyLocal(j, x_iter)
            uj = uj_solution(xj, s_factorization.getBlock(j), B.getBlock(j), T.getBlock(j), BVec.getBlock(j))
            uj_list_per_iter[i].append(uj)
            vmax = max(vmax, np.max(np.abs(uj_list_per_iter[i][j])))



    frames = []

    for i in range(len(x_list)):
        fig = plt.figure(figsize=(6, 10))

        total_vtxj, total_eltj = np.zeros(0), np.zeros(0)
        u_full = np.zeros(0)

        for j in range(J):
            vtxj, eltj = mesh.getLocal(j)

            if total_vtxj.size == 0:
                total_vtxj = vtxj
                total_eltj = eltj
            else:
                total_vtxj = np.concatenate([total_vtxj, vtxj])
                total_eltj = np.concatenate([total_eltj, eltj])

            u_full = np.concatenate([u_full, uj_list_per_iter[i][j]])

        plot_mesh(total_vtxj, total_eltj)
        tc = plt.tricontourf(
            total_vtxj[:, 0],
            total_vtxj[:, 1],
            np.abs(u_full),
            levels=20,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax
        )

        plt.title(f'Iteration {i*iter_sample}', fontsize=12, fontweight='bold')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(tc)
        plt.tight_layout()

        # ---- salva il frame in memoria ----
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        frames.append(imageio.v2.imread(buf))
        buf.close()

        plt.close(fig)

    filename = f'fixed_point_evolution_sp{sp[0][0]}-{sp[0][1]}'
    for p in sp[1:]:
        filename += f'_{p[0]}-{p[1]}'
    filename += '.gif'
    gif_path = os.path.join(
        plots_dir,
        filename
    )

    imageio.mimsave(gif_path, frames, fps=2)



if __name__ == "__main__":
    run_convergence_test()
    show_theoretical_convergence(0.5, [0.55, 0.65] )
    show_theoretical_convergence(0.1, [0.55, 0.65])
    show_theoretical_convergence(0.05, [0.55, 0.65] )

    # plot_solution_over_iterations(0.1)
    plot_solution_over_iterations_gif(0.1)