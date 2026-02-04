#!/usr/bin/env python3
"""
All tasks: Fixed point, GMRES, convergence studies, solution plots, runtime comparison
"""

import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time
import os
import logging
import sys
from typing import Tuple, List, Optional

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

params = HelmholtzParameters() 


logger.info("="*70)
logger.info("Section 2.5 - Complete Implementation")
logger.info("="*70)


def _merge_duplicate_vertices(vertices: np.ndarray, 
                               solution: np.ndarray, 
                               tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
   
    n_vertices = len(vertices)
    
   
    rounded = np.round(vertices / tol) * tol
    
   
    scale = 1.0 / tol
    keys = rounded[:, 0] * scale + rounded[:, 1] * scale * 1e10
    
    
    sort_idx = np.argsort(keys)
    sorted_keys = keys[sort_idx]
    
   
    unique_mask = np.concatenate([[True], np.abs(np.diff(sorted_keys)) > 0.5])
    
    
    inverse_map = np.zeros(n_vertices, dtype=int)
    unique_indices = np.cumsum(unique_mask) - 1
    
  
    inverse_map[sort_idx] = unique_indices
    
   
    unique_sorted_idx = sort_idx[unique_mask]
    unique_vtx = vertices[unique_sorted_idx]
    
   
    n_unique = len(unique_vtx)
    merged_solution = np.zeros(n_unique, dtype=solution.dtype)
    counts = np.zeros(n_unique, dtype=int)
    
    for i, unique_idx in enumerate(inverse_map):
        merged_solution[unique_idx] += solution[i]
        counts[unique_idx] += 1
    
    merged_solution /= counts
    
    return unique_vtx, inverse_map, merged_solution


def merge_local_solutions(mesh, uj_list: List[np.ndarray], 
                          tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
   
    J = len(uj_list)
    
    
    all_vertices = []
    all_elements = []
    all_solutions = []
    vertex_offset = 0
    
    
    for j in range(J):
        vtxj, eltj = mesh.getLocal(j)
        uj = uj_list[j]
        
        all_vertices.append(vtxj)
        all_elements.append(eltj + vertex_offset)
        all_solutions.append(uj)
        
        vertex_offset += len(vtxj)
    
   
    raw_vtx = np.vstack(all_vertices)
    raw_elt = np.vstack(all_elements)
    raw_u = np.concatenate(all_solutions)
    
   
    global_vtx, inverse_map, global_u = _merge_duplicate_vertices(raw_vtx, raw_u, tol)
    
    global_elt = inverse_map[raw_elt]
    
    return global_vtx, global_elt, global_u


def compute_interface_jump(mesh, uj_list: List[np.ndarray], 
                           nx: int) -> Tuple[np.ndarray, float]:
   
    J = len(uj_list)
    jumps = []
    
    for j in range(J - 1):
        # Soluzione sul bordo superiore del subdominio j
        # (ultimi nx vertici nella numerazione locale)
        vtxj, _ = mesh.getLocal(j)
        ny_local_j = len(np.unique(vtxj[:, 1]))
        top_indices_j = np.arange((ny_local_j - 1) * nx, ny_local_j * nx)
        u_top_j = uj_list[j][top_indices_j]
        
        # Soluzione sul bordo inferiore del subdominio j+1
        # (primi nx vertici nella numerazione locale)
        bottom_indices_jp1 = np.arange(nx)
        u_bottom_jp1 = uj_list[j + 1][bottom_indices_jp1]
        
        # Calcola il salto
        jump = np.abs(u_top_j - u_bottom_jp1)
        jumps.append(jump)
    
    jumps = np.array(jumps)
    
    # Normalizza per il valore massimo della soluzione
    max_u = max(np.abs(uj).max() for uj in uj_list)
    max_jump = jumps.max() / max_u if max_u > 0 else 0.0
    
    return jumps, max_jump


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
    x_fp, residuals_fp, convergence = fixed_point_solver(-g, S, Pi, omega, max_iter=400, tol=1e-10)
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

    # IMPORTANT: Get parameters from solver6 to ensure same sources are used
    solver_params = solver6._params
    logger.info(f"  Using {len(solver_params.sp)} point sources from DDM solver")

    # DDM-GMRES timing
    logger.info("Timing DDM-GMRES")
    start_ddm = time.time()
    # x_ddm, residuals_ddm, info_ddm, _, _ = solve_ddm_gmres(factorizations, Bj_list, Cj_list, Tj_list, g, nx, J)
    x_ddm, residuals_ddm, info_ddm, _, _ = solver6.solve()
    time_ddm = time.time() - start_ddm
    logger.info(f"  Time: {time_ddm:.3f}s, Iterations: {len(residuals_ddm)}")

   
    logger.info("Building and solving full problem with GMRES")
    vtx, elt = mesh_fnc(nx, ny, solver_params.Lx, solver_params.Ly)

  
    boundary_dict = boundary_fnc(nx, ny)
    belt = np.vstack(list(boundary_dict.values()))
   

    M = mass(vtx, elt)
    Mb = mass(vtx, belt)
    K = stiffness(vtx, elt)
   
    A_full = K - solver_params.kappa**2 * M - 1j * solver_params.kappa * Mb
    b_full = M @ point_source(solver_params.sp, solver_params.kappa)(vtx)

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

    # Plot full reference solution
    plt.figure(figsize=(8, 10))
    triang = mtri.Triangulation(vtx[:, 0], vtx[:, 1], elt)
    tc_full = plt.tricontourf(triang, np.abs(x_full), levels=100, cmap='viridis')
    plt.colorbar(tc_full)
    plt.title("Full Reference Solution (GMRES)")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.savefig(os.path.join(plots_dir, "full_reference_solution.png"), dpi=150) 

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

    # =========================================================================
    # Quantitative comparison: DDM vs Full GMRES solutions
    # =========================================================================
    logger.info("")
    logger.info("Comparing DDM and Full GMRES solutions quantitatively")
    
    # Reconstruct DDM global solution
    components = solver6.getComponents()
    s_fact, B, Q, T, BVec, mesh_ddm = components[0:6]
    J = solver6._J
    
    uj_list = []
    for j in range(J):
        xj = Q.applyLocal(j, x_ddm)
        uj = uj_solution(xj, s_fact.getBlock(j), B.getBlock(j), T.getBlock(j), BVec.getBlock(j))
        uj_list.append(uj)
    
    # Merge DDM solution
    global_vtx_ddm, global_elt_ddm, global_u_ddm = merge_local_solutions(mesh_ddm, uj_list)
    
    # =========================================================================
    # DIAGNOSTIC: Check if vertices match directly
    # =========================================================================
    logger.info("")
    logger.info("  Diagnostic info:")
    logger.info(f"    Full mesh vertices: {len(vtx)}")
    logger.info(f"    DDM merged vertices: {len(global_vtx_ddm)}")
    logger.info(f"    Full solution shape: {x_full.shape}, dtype: {x_full.dtype}")
    logger.info(f"    DDM solution shape: {global_u_ddm.shape}, dtype: {global_u_ddm.dtype}")
    
    # Check if vertices are the same (they should be for the same nx, ny)
    # Sort both by coordinates to align them
    def sort_vertices_with_values(vtx, values):
        """Sort vertices lexicographically and reorder values accordingly."""
        # Create composite key for sorting
        keys = np.round(vtx[:, 0], 10) + np.round(vtx[:, 1], 10) * 1e6
        sort_idx = np.argsort(keys)
        return vtx[sort_idx], values[sort_idx], sort_idx
    
    vtx_full_sorted, u_full_sorted, idx_full = sort_vertices_with_values(vtx, x_full)
    vtx_ddm_sorted, u_ddm_sorted, idx_ddm = sort_vertices_with_values(global_vtx_ddm, global_u_ddm)
    
    # Check if vertices match
    if len(vtx_full_sorted) == len(vtx_ddm_sorted):
        vtx_diff = np.max(np.abs(vtx_full_sorted - vtx_ddm_sorted))
        logger.info(f"    Max vertex coordinate difference: {vtx_diff:.2e}")
        
        if vtx_diff < 1e-10:
            # Vertices match! Direct comparison possible
            error = u_ddm_sorted - u_full_sorted
            rel_error_l2 = np.linalg.norm(error) / np.linalg.norm(u_full_sorted)
            rel_error_linf = np.max(np.abs(error)) / np.max(np.abs(u_full_sorted))
            
            logger.info(f"  Direct comparison (sorted vertices):")
            logger.info(f"    Relative L2 error: {rel_error_l2:.2e}")
            logger.info(f"    Relative Linf error: {rel_error_linf:.2e}")
            
            # Check for phase/sign issues
            # Compute correlation
            correlation = np.abs(np.vdot(u_ddm_sorted, u_full_sorted)) / (np.linalg.norm(u_ddm_sorted) * np.linalg.norm(u_full_sorted))
            logger.info(f"    Solution correlation: {correlation:.6f}")
            
            # Check if there's a global phase difference
            phase_ratio = u_ddm_sorted / (u_full_sorted + 1e-15)
            phase_at_max = phase_ratio[np.argmax(np.abs(u_full_sorted))]
            logger.info(f"    Phase ratio at max: {phase_at_max:.4f} (magnitude: {np.abs(phase_at_max):.4f})")
            
            # Check real and imaginary parts separately
            rel_error_real = np.linalg.norm(u_ddm_sorted.real - u_full_sorted.real) / (np.linalg.norm(u_full_sorted.real) + 1e-15)
            rel_error_imag = np.linalg.norm(u_ddm_sorted.imag - u_full_sorted.imag) / (np.linalg.norm(u_full_sorted.imag) + 1e-15)
            logger.info(f"    Relative error (real part): {rel_error_real:.2e}")
            logger.info(f"    Relative error (imag part): {rel_error_imag:.2e}")
            
            # Use sorted values for plotting
            error_for_plot = np.abs(error)
            # Map back to DDM ordering for plot
            error_plot_ddm = np.zeros_like(global_u_ddm, dtype=float)
            error_plot_ddm[idx_ddm] = error_for_plot[np.argsort(idx_ddm)]
            
        else:
            logger.warning("    Vertices don't match exactly - using interpolation")
            # Fall back to interpolation
            from scipy.interpolate import griddata
            u_full_at_ddm = griddata(vtx, x_full.real, global_vtx_ddm, method='linear') + \
                            1j * griddata(vtx, x_full.imag, global_vtx_ddm, method='linear')
            valid_mask = ~np.isnan(u_full_at_ddm)
            error = global_u_ddm[valid_mask] - u_full_at_ddm[valid_mask]
            rel_error_l2 = np.linalg.norm(error) / np.linalg.norm(u_full_at_ddm[valid_mask])
            error_plot_ddm = np.abs(global_u_ddm - u_full_at_ddm)
            error_plot_ddm[~valid_mask] = 0
    else:
        logger.warning(f"    Different number of vertices! Full: {len(vtx)}, DDM: {len(global_vtx_ddm)}")
        # Use interpolation
        from scipy.interpolate import griddata
        u_full_at_ddm = griddata(vtx, x_full.real, global_vtx_ddm, method='linear') + \
                        1j * griddata(vtx, x_full.imag, global_vtx_ddm, method='linear')
        valid_mask = ~np.isnan(u_full_at_ddm)
        error = global_u_ddm[valid_mask] - u_full_at_ddm[valid_mask]
        rel_error_l2 = np.linalg.norm(error) / np.linalg.norm(u_full_at_ddm[valid_mask])
        error_plot_ddm = np.abs(global_u_ddm - u_full_at_ddm)
        error_plot_ddm[~valid_mask] = 0
    
    # Side-by-side comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    # Full GMRES solution
    ax = axes[0]
    triang_full = mtri.Triangulation(vtx[:, 0], vtx[:, 1], elt)
    tc = ax.tricontourf(triang_full, np.abs(x_full), levels=50, cmap='viridis')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Full GMRES Solution')
    plt.colorbar(tc, ax=ax)
    
    # DDM solution
    ax = axes[1]
    triang_ddm = mtri.Triangulation(global_vtx_ddm[:, 0], global_vtx_ddm[:, 1], global_elt_ddm)
    tc = ax.tricontourf(triang_ddm, np.abs(global_u_ddm), levels=50, cmap='viridis')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('DDM Solution')
    plt.colorbar(tc, ax=ax)
    
    # Error
    ax = axes[2]
    tc = ax.tricontourf(triang_ddm, error_plot_ddm, levels=50, cmap='hot')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'|Error| (rel. L2: {rel_error_l2:.2e})')
    plt.colorbar(tc, ax=ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'task7_solution_comparison.png'), dpi=150)
    logger.info(f"  Saved: task7_solution_comparison.png")

    return time_full, time_ddm


def plot_global_solution(solver, mesh, J, nx, 
                         title: str = "Global DDM Solution",
                         plot_type: str = "both",
                         save_prefix: str = "global_ddm_solution"):
    """
    Ricostruisce e plotta la soluzione globale dai subdomini, 
    gestendo correttamente i vertici duplicati alle interfacce.
    
    Parameters
    ----------
    solver : HelmholtzSolver
        Il solver DDM già assemblato
    mesh : FullMesh
        L'oggetto mesh
    J : int
        Numero di subdomini
    nx : int
        Numero di punti in direzione x (per calcolo salti)
    title : str
        Titolo del grafico
    plot_type : str
        "real", "abs", "both"
    save_prefix : str
        Prefisso per i file salvati
    """
    logger.info("")
    logger.info("Plotting global solution with vertex merging")
    logger.info("-"*70)
    
   
    x_solution, _, _, _, _ = solver.solve()
    components = solver.getComponents()
    s_fact, B, Q, T, BVec = components[0:5]

   
    uj_list = []
    for j in range(J):
        xj = Q.applyLocal(j, x_solution)
        uj = uj_solution(xj, s_fact.getBlock(j), B.getBlock(j), T.getBlock(j), BVec.getBlock(j))
        uj_list.append(uj)
        logger.info(f"  Subdomain {j}: {len(uj)} DOFs, |u|_max = {np.abs(uj).max():.4f}")

   
    jumps, max_jump = compute_interface_jump(mesh, uj_list, nx)
    logger.info(f"  Max normalized interface jump: {max_jump:.2e}")
    
   
    global_vtx, global_elt, global_u = merge_local_solutions(mesh, uj_list)
    
   
    total_local_vtx = sum(mesh.getLocal(j)[0].shape[0] for j in range(J))
    logger.info(f"  Vertices before merge: {total_local_vtx}")
    logger.info(f"  Vertices after merge:  {len(global_vtx)}")
    logger.info(f"  Duplicates removed:    {total_local_vtx - len(global_vtx)}")

    # Crea triangolazione matplotlib
    triang = mtri.Triangulation(global_vtx[:, 0], global_vtx[:, 1], global_elt)

    if plot_type == "both":
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        # Real part
        ax = axes[0]
        tc = ax.tricontourf(triang, global_u.real, levels=50, cmap='RdBu_r')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{title} - Real Part')
        plt.colorbar(tc, ax=ax)
        
        # Absolute value
        ax = axes[1]
        tc = ax.tricontourf(triang, np.abs(global_u), levels=50, cmap='viridis')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{title} - Absolute Value')
        plt.colorbar(tc, ax=ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{save_prefix}_both.png'), dpi=150)
        
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        
        if plot_type == "real":
            data = global_u.real
            cmap = 'RdBu_r'
        else:  # abs
            data = np.abs(global_u)
            cmap = 'viridis'
            
        tc = ax.tricontourf(triang, data, levels=50, cmap=cmap)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        plt.colorbar(tc, ax=ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{save_prefix}_{plot_type}.png'), dpi=150)
    fig_jump, ax_jump = plt.subplots(figsize=(10, 4))
    for k, jump in enumerate(jumps):
        ax_jump.semilogy(jump, label=f'Interface {k}-{k+1}')
    ax_jump.set_xlabel('Interface node index')
    ax_jump.set_ylabel('|jump|')
    ax_jump.set_title(f'Solution jump at interfaces (max normalized: {max_jump:.2e})')
    ax_jump.legend()
    ax_jump.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{save_prefix}_interface_jumps.png'), dpi=150)
    
    logger.info(f"  Saved: {save_prefix}_*.png")
    
    return global_vtx, global_elt, global_u


if __name__ == "__main__":

    residuals_fp, residuals_gmres = task1_2()
    refinement_results = task4()
    task4_2()
    fixed_domain_results, fixed_dofs_results = task_5()
    plot_4_5(refinement_results, fixed_domain_results, fixed_dofs_results)
    solver6, nx, ny = task_6()
    time_full, time_ddm = task_7(solver6, nx, ny)
    
   
    components = solver6.getComponents()
    mesh = components[5]  
    J = solver6._J       
    print("\n" + "="*70)
    print("Generazione grafico globale (con merge dei vertici duplicati)...")
    print("="*70)
    plot_global_solution(solver6, mesh, J, nx, 
                         title="Helmholtz DDM Solution",
                         plot_type="both",
                         save_prefix="global_ddm_solution")

    logger.info("")
    logger.info("="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Task 3: Convergence comparison - GMRES converges {len(residuals_fp)//len(residuals_gmres)}x faster")
    logger.info(f"Task 4: Mesh refinement - iterations stable ({refinement_results[0]['iterations']} to {refinement_results[-1]['iterations']})")
    logger.info(f"Task 7: Runtime comparison - {time_full/time_ddm:.2f}x speedup with DDM")