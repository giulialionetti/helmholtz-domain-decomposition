#!/usr/bin/env python3
"""
All tasks: Fixed point, GMRES, convergence studies, solution plots, runtime comparison
"""

import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
import os
import logging
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_path = current_dir
while not os.path.exists(os.path.join(project_root, 'src')):
    parent = os.path.dirname(project_root)
    if parent == project_root:
        project_root = os.path.abspath(os.path.join(current_dir, "../"))
        break
    project_root = parent

if project_root not in sys.path:
    sys.path.insert(0, project_root)


# from src.seq.linear_solver.GMRES import solve_ddm_gmres
from src.seq.helmholtz_solver import HelmholtzSolver
from src.common.helmholtz.helmholtz_param import HelmholtzParameters

from src.seq.operators.s_operator import FullSOperator
from src.seq.operators.pi_operator import FullPiOperator
from src.seq.operators.u_operator import uj_solution
from src.seq.linear_solver.fixed_point import fixed_point_solver
from src.common.mesh import plot_mesh
from src.common.mesh import mesh as mesh_fnc
from src.common.mesh import boundary as boundary_fnc
from src.common.helmholtz.system_assembly import mass, stiffness, point_source

# ============================== LOGGING & PLOTTING SETUP ===============================
# Ensure imports work from project root (Path fix)
log_file = "GMRES_results.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    filename=log_file,  # Log to this file
    filemode='w'        # 'w' overwrites, 'a' appends
)


logger = logging.getLogger(__name__)

plots_dir = os.path.join(project_root, 'plots')
os.makedirs(plots_dir, exist_ok=True)

logger.info(f"Logging initialized. Output will be saved to {log_file}")


# ================================= PROBLEM PARAMETERS =================================
params = HelmholtzParameters() 


logger.info("="*70)
logger.info("Section 2.5 - Complete Implementation")
logger.info("="*70)

# ==============================================================================
# Task 1 & 2: Fixed Point and GMRES with convergence plots
# ==============================================================================
def task1_2():
    logger.info("")
    logger.info("Tasks 1-3: Fixed point, GMRES, and convergence comparison")
    logger.info("-"*70)

    nx, ny, J = 33, 65, 4
    params.nx = nx
    params.ny = ny
    logger.info(f"Configuration: {nx}x{ny} mesh, {J} subdomains")

    solver12 = HelmholtzSolver(params, J)

    solver12.assembly()
    components = solver12.getComponents()
    # factorizations, Bj_list, Cj_list, Tj_list, bj_list, vtxj_list, eltj_list, g = components
    s_factorization, B, Q, T, b, mesh, g, S, Pi = components



    # Fixed point solver
    logger.info("Running fixed-point solver")
    omega = 0.1
    # Note: fixed_point_solver returns 3 values: x, residuals, converged
    x_fp, residuals_fp, convergence = fixed_point_solver(
        -g, # Note the minus g
        S, Pi,
        omega, max_iter=400, tol=1e-10)
    logger.info(f"  Iterations: {len(residuals_fp)}, Final residual: {residuals_fp[-1]:.6e}, Converged: {convergence}")

    # GMRES solver
    logger.info("Running GMRES solver")
    x_gmres, residuals_gmres, info, _, _ = solver12.solve()
    logger.info(f"  Iterations: {len(residuals_gmres)}, Final residual: {residuals_gmres[-1]:.6e}")

    # Plot comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.semilogy(residuals_gmres, 'b-', linewidth=2, label='GMRES')
    ax.semilogy(residuals_fp, 'r--', linewidth=2, label=f'Fixed Point (ω={omega})')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Residual norm', fontsize=12)
    ax.set_title('Task 3: Convergence Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'task3_convergence_comparison.png'), dpi=150)

    return residuals_fp, residuals_gmres

# ==============================================================================
# Task 4: Mesh refinement study
# ==============================================================================
def task4():
    logger.info("")
    logger.info("Task 4: Mesh refinement study")
    logger.info("-"*70)

    mesh_sizes = [(17, 33), (25, 49), (33, 65), (101, 101)]
    J = 4
    refinement_results = []

    for nx, ny in mesh_sizes:
        logger.info(f"  Mesh {nx}x{ny}")
        
        params.nx = nx
        params.ny = ny
        solver4 = HelmholtzSolver(params,J)

        solver4.assembly()
        components = solver4.getComponents()
        # factorizations, Bj_list, Cj_list, Tj_list, bj_list, vtxj_list, eltj_list, g = components
        s_factorization, B, Q, T, b, mesh, g, S, Pi = components

        start = time.time() 
        x, residuals, info, _, _ = solver4.solve()
        end = time.time()
        logger.info(f"    DOFs: {nx * ny}, Iterations: {len(residuals)}, Time: {end-start}")
        
        refinement_results.append({
            'nx': nx, 'ny': ny, 'ndof': nx * ny,
            'iterations': len(residuals), 'residuals': residuals
        })

    return refinement_results

def task4_2():
    logger.info("")
    logger.info("Task 4.2: Mesh refinement study (advanced)")
    logger.info("-"*70)

    local_params = HelmholtzParameters(Lx=100, Ly=200)

    mesh_sizes = [(17, 33), (25, 49), (33, 65), (101, 101)]
    J = 4
    refinement_results = []

    for nx, ny in mesh_sizes:
        logger.info(f"  Mesh {nx}x{ny}")
        
        params.nx = nx
        params.ny = ny
        solver4 = HelmholtzSolver(params,J)

        solver4.assembly()
        components = solver4.getComponents()
        # factorizations, Bj_list, Cj_list, Tj_list, bj_list, vtxj_list, eltj_list, g = components
        s_factorization, B, Q, T, b, mesh, g, S, Pi = components

        start = time.time() 
        x, residuals, info, _, _ = solver4.solve()
        end = time.time()
        A_op = solver4.getIterationMatrix()
        # sigma_max = spla.eigsh(A_op, k=1, which='LM', return_eigenvectors=False)[0]
        # sigma_min = spla.eigsh(A_op, k=1, which='SM', return_eigenvectors=False)[0]
        logger.info(f"    DOFs: {nx * ny}, Iterations: {len(residuals)}, Time: {end-start}")
        
        refinement_results.append({
            'nx': nx, 'ny': ny, 'ndof': nx * ny,
            'iterations': len(residuals), 'residuals': residuals
        })

    return refinement_results

# ==============================================================================
# Task 5: Subdomain scaling studies
# ==============================================================================
def task_5():
    logger.info("")
    logger.info("Task 5a: Subdomain scaling (fixed domain size)")
    logger.info("-"*70)

    nx, ny = 33, 65
    J_values = [2, 4, 8]
    fixed_domain_results = []

    for J in J_values:
        if (ny - 1) % J != 0:
            continue
        
        logger.info(f"  J={J} subdomains")
        
        params.nx = nx
        params.ny = ny
        solver5_1 = HelmholtzSolver(params, J)

        solver5_1.assembly()
        components = solver5_1.getComponents()
        # factorizations, Bj_list, Cj_list, Tj_list, bj_list, vtxj_list, eltj_list, g = components
        s_factorization, B, Q, T, b, mesh, g, S, Pi = components

        # components = build_ddm_solver(nx, ny, J, params)
        # factorizations, Bj_list, Cj_list, Tj_list, bj_list, vtxj_list, eltj_list, g = components
        
        x, residuals, info, _, _ = solver5_1.solve()
        
        logger.info(f"    DOFs/subdomain: {nx * ((ny-1)//J + 1)}, Iterations: {len(residuals)}")
        
        fixed_domain_results.append({
            'J': J, 'iterations': len(residuals), 'residuals': residuals
        })

    logger.info("")
    logger.info("Task 5b: Subdomain scaling (fixed DOFs per subdomain)")
    logger.info("-"*70)

    configs = [(17, 17, 2), (17, 33, 4), (17, 65, 8)]
    fixed_dofs_results = []

    for nx, ny, J in configs:
        dofs_per_sub = nx * ((ny-1)//J + 1)
        logger.info(f"  J={J}, mesh {nx}x{ny}, ~{dofs_per_sub} DOFs/subdomain")
        
        params.nx = nx
        params.ny = ny
        solver5_2 = HelmholtzSolver(params, J)

        solver5_2.assembly()
        components = solver5_2.getComponents()
        # factorizations, Bj_list, Cj_list, Tj_list, bj_list, vtxj_list, eltj_list, g = components
        s_factorization, B, Q, T, b, mesh, g, S, Pi = components

        # components = build_ddm_solver(nx, ny, J, params)
        # factorizations, Bj_list, Cj_list, Tj_list, bj_list, vtxj_list, eltj_list, g = components
        
        x, residuals, info, _, _ = solver5_2.solve()
        
        logger.info(f"    Iterations: {len(residuals)}")
        
        fixed_dofs_results.append({
            'J': J, 'dofs_per_sub': dofs_per_sub,
            'iterations': len(residuals), 'residuals': residuals
        })

    return fixed_domain_results, fixed_dofs_results

def plot_4_5(refinement_results, fixed_domain_results, fixed_dofs_results):
    # Plot Tasks 4 and 5
    fig = plt.figure(figsize=(18, 5))

    ax1 = plt.subplot(131)
    for r in refinement_results:
        ax1.semilogy(r['residuals'], label=f"{r['nx']}×{r['ny']} ({r['ndof']} DOFs)", linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Residual norm', fontsize=11)
    ax1.set_title('Task 4: Mesh Refinement', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2 = plt.subplot(132)
    for r in fixed_domain_results:
        ax2.semilogy(r['residuals'], label=f"J={r['J']}", linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Residual norm', fontsize=11)
    ax2.set_title('Task 5a: Fixed Domain Size', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    ax3 = plt.subplot(133)
    for r in fixed_dofs_results:
        ax3.semilogy(r['residuals'], label=f"J={r['J']}", linewidth=2)
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Residual norm', fontsize=11)
    ax3.set_title('Task 5b: Fixed DOFs/Subdomain', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'tasks_4_5_convergence_studies.png'), dpi=150)


# ==============================================================================
# Task 6: Plot local solutions
# ==============================================================================
def task_6():
    logger.info("")
    logger.info("Task 6: Plot local solutions")
    logger.info("-"*70)

    nx, ny, J = 33, 65, 4
    logger.info(f"Configuration: {nx}x{ny} mesh, {J} subdomains")

    params.nx = nx
    params.ny = ny
    solver6 = HelmholtzSolver(params, J)

    solver6.assembly()
    components = solver6.getComponents()
    # factorizations, Bj_list, Cj_list, Tj_list, bj_list, vtxj_list, eltj_list, g = components
    s_factorization, B, Q, T, b, mesh, g, S, Pi = components

    # components = build_ddm_solver(nx, ny, J, params)
    # factorizations, Bj_list, Cj_list, Tj_list, bj_list, vtxj_list, eltj_list, g = components

    # x_solution, residuals, info, _, _ = solver6.solve()

    x_solution, res, converged = fixed_point_solver(-g, S, Pi, 0.1, 400, 10e-10)

    # Compute local solutions
    uj_list = []
    for j in range(J):
        xj = Q.applyLocal(j, x_solution)
        uj = uj_solution(xj, s_factorization.getBlock(j), B.getBlock(j), T.getBlock(j), b.getBlock(j))
        uj_list.append(uj)

    # Plot real parts
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for j in range(J):
        ax = axes[j]
        plt.sca(ax)
        vtxj, eltj = mesh.getLocal(j)
        plot_mesh(vtxj, eltj)
        tc = ax.tricontourf(vtxj[:, 0], vtxj[:, 1], 
                            uj_list[j].real, levels=20, cmap='RdBu_r')
        ax.set_title(f'Subdomain {j} - Real part', fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(tc, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'task6_local_solutions_real.png'), dpi=150)

    # Plot modulus
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for j in range(J):
        ax = axes[j]
        plt.sca(ax)
        vtxj, eltj = mesh.getLocal(j)
        plot_mesh(vtxj, eltj)
        tc = ax.tricontourf(vtxj[:, 0], vtxj[:, 1], 
                            np.abs(uj_list[j]), levels=20, cmap='viridis')
        ax.set_title(f'Subdomain {j} - Modulus', fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(tc, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'task6_local_solutions_modulus.png'), dpi=150)
    
    return solver6, nx, ny

# ==============================================================================
# Task 7: Runtime comparison
# ==============================================================================
def task_7(solver6, nx, ny):
    logger.info("")
    logger.info("Task 7: Runtime comparison with full GMRES")
    logger.info("-"*70)

    # DDM-GMRES timing
    logger.info("Timing DDM-GMRES")
    start_ddm = time.time()
    # x_ddm, residuals_ddm, info_ddm, _, _ = solve_ddm_gmres(factorizations, Bj_list, Cj_list, Tj_list, g, nx, J)
    x_ddm, residuals_ddm, info_ddm, _, _ = solver6.solve()
    time_ddm = time.time() - start_ddm
    logger.info(f"  Time: {time_ddm:.3f}s, Iterations: {len(residuals_ddm)}")

    # Full GMRES
    logger.info("Building and solving full problem with GMRES")
    vtx, elt = mesh_fnc(nx, ny, params.Lx, params.Ly)

    # --- CORRECT FIX: Stack edges for mass matrix ---
    boundary_dict = boundary_fnc(nx, ny)
    belt = np.vstack(list(boundary_dict.values()))
    # -----------------------------------------------

    M = mass(vtx, elt)
    Mb = mass(vtx, belt)
    K = stiffness(vtx, elt)
    A_full = K - params.kappa**2 * M - 1j * params.kappa * Mb
    b_full = M @ point_source(params.sp, params.kappa)(vtx)

    start_full = time.time()
    residuals_full = []
    def callback_full(rk):
        residuals_full.append(rk)

    x_full, info_full = spla.gmres(A_full, b_full, rtol=1e-10, callback=callback_full,
                                    callback_type='pr_norm', maxiter=5000)
    time_full = time.time() - start_full
    logger.info(f"  Time: {time_full:.3f}s, Iterations: {len(residuals_full)}")

    logger.info("")
    logger.info(f"Speedup: {time_full/time_ddm:.2f}x")

    # Plot runtime comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Convergence
    ax1.semilogy(residuals_ddm, 'b-', linewidth=2, label='DDM-GMRES')
    ax1.semilogy(residuals_full, 'r--', linewidth=2, label='Full GMRES')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Residual norm', fontsize=11)
    ax1.set_title('Convergence Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Runtime bars
    methods = ['DDM-GMRES', 'Full GMRES']
    times = [time_ddm, time_full]
    iterations = [len(residuals_ddm), len(residuals_full)]

    x_pos = np.arange(len(methods))
    ax2_twin = ax2.twinx()

    bars1 = ax2.bar(x_pos - 0.2, times, 0.4, label='Time (s)', color='steelblue')
    bars2 = ax2_twin.bar(x_pos + 0.2, iterations, 0.4, label='Iterations', color='coral')

    ax2.set_xlabel('Method', fontsize=11)
    ax2.set_ylabel('Time (s)', color='steelblue', fontsize=11)
    ax2_twin.set_ylabel('Iterations', color='coral', fontsize=11)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods)
    ax2.set_title('Runtime Comparison', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2_twin.tick_params(axis='y', labelcolor='coral')

    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'task7_runtime_comparison.png'), dpi=150)

    return time_full, time_ddm


    

if __name__ == "__main__":

    residuals_fp, residuals_gmres = task1_2()
    refinement_results = task4()
    task4_2()
    fixed_domain_results, fixed_dofs_results = task_5()
    plot_4_5(refinement_results, fixed_domain_results, fixed_dofs_results)
    solver6, nx, ny = task_6()
    time_full, time_ddm = task_7(solver6, nx, ny)

    logger.info(f"Task 3: Convergence comparison - GMRES converges {len(residuals_fp)//len(residuals_gmres)}x faster")
    logger.info(f"Task 4: Mesh refinement - iterations stable ({refinement_results[0]['iterations']} to {refinement_results[-1]['iterations']})")
    logger.info(f"Task 7: Runtime comparison - {time_full/time_ddm:.2f}x speedup with DDM")
