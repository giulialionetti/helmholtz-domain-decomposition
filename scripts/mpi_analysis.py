import os
import sys
import imageio
from io import BytesIO
import matplotlib.pyplot as plt

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

import numpy as np
from mpi4py import MPI
from src.common.helmholtz.helmholtz_param import HelmholtzParameters
from src.par.helmholtz_solver import HelmholtzSolverMPI
from src.seq.mesh.mesh import FullMesh
from src.common.mesh import plot_mesh

plots_dir = os.path.join(project_root, 'plots')
os.makedirs(plots_dir, exist_ok=True)

def run_helmholtz_mpi(params, J: int, solver_type: str = 'gmres'):
    """
    Run the MPI-parallelized Helmholtz DDM solver.
    
    This function should be called from an MPI script:
    
    $ mpiexec -n 4 python helmholtz_mpi_script.py
    
    Parameters
    ----------
    params : HelmholtzParameters
        Problem parameters
    J : int
        Number of subdomains
    solver_type : str
        'gmres' or 'fixed_point'
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Create solver (one instance per process)
    solver = HelmholtzSolverMPI(comm, params, J)
    
    # Assemble local components
    solver.assembly()
    

    # Solve interface problem
    if solver_type == 'gmres':
        x_local, residuals, info = solver.solve_gmres(tol=1e-10, maxiter=500)
    elif solver_type == 'fixed_point':
        x_local, residuals, converged = solver.solve_fixed_point(
            omega=0.8, tol=1e-8, maxiter=500
        )
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
    

    # Reconstruct local solution
    u_j = solver.reconstruct_local_solution()
    
    if rank == 0:
        print(f"\nSolution reconstruction complete on all {J} processes.")
    
    return solver, u_j


if __name__ == "__main__":
    
    # Define problem parameters
    Lx, Ly = 1.0, 2.0
    kappa = 16.0
    nx_global = 33
    ny_global = 65
    J = 4  # Must match number of MPI processes
    
    mesh = FullMesh(J, nx_global, ny_global, Lx, Ly)
    mesh.build()

    sp = [np.array([0.5, 0.75, 1.0])]
    params = HelmholtzParameters(Lx, Ly, kappa, sp=sp)
    params.nx = nx_global
    params.ny = ny_global
    
    # Run MPI solver
    solver, u_j = run_helmholtz_mpi(params, J, solver_type='gmres')
    
    comm = MPI.COMM_WORLD

    print(f"Rank {comm.Get_rank()}: len(u_j)={len(u_j)}")
    exit(0)

    u_full = np.zeros(u_j.shape[0]*J)
    comm.Gather(u_j, u_full, root = 0)

    if comm.Get_rank() == 0:
        frames = []
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

        plot_mesh(total_vtxj, total_eltj)
        tc = plt.tricontourf(
            total_vtxj[:, 0],
            total_vtxj[:, 1],
            np.abs(u_full),
            levels=20,
            cmap='viridis'
        )

        plt.title(f'MPI solution', fontsize=12, fontweight='bold')
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
        plt.close()

        filename = f'fixed_point_evolution_sp{params.sp[0][0]}-{params.sp[0][1]}'
        # for p in params.sp[1:]:
        #     filename += f'_{p[0]}-{p[1]}'
        filename += '.gif'
        gif_path = os.path.join(
            plots_dir,
            filename
        )

        imageio.mimsave(gif_path, frames, fps=2)
