import numpy as np
from mpi4py import MPI
from src.par.operators.s_operator import SparseSOperator
from src.par.operators.pi_operator import SparsePiOperator

def fixed_point_solver_mpi(comm: MPI.Comm, g_local: np.ndarray, 
                           S_op: SparseSOperator, Pi_op: SparsePiOperator,
                           omega: float, max_iter: int = 500, 
                           tol: float = 1e-8) -> tuple:
    """
    Solve (I + Π S) x = -g using fixed-point iteration in MPI-distributed setting.
    
    Fixed-point iteration: x^{k+1} = ω(−g − Π S x^k) + (1−ω) x^k
    
    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator
    g_local : np.ndarray
        Local portion of RHS vector
    S_op : SparseSOperator
        Schur complement operator
    Pi_op : SparsePiOperator
        Exchange operator
    omega : float
        Relaxation parameter
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
        
    Returns
    -------
    x_local : np.ndarray
        Local solution vector
    residuals : list
        Residual history (computed on rank 0)
    converged : bool
        Whether solver converged
    """
    rank = comm.Get_rank()
    
    x = np.zeros_like(g_local)
    residuals = []
    
    for k in range(max_iter):
        # Compute Π S x
        Sx = S_op.applyGlobal(x)
        Pi_Sx = Pi_op.applyGlobal(Sx)
        
        # Fixed-point update
        x_new = omega * (-g_local - Pi_Sx) + (1 - omega) * x
        
        # Compute residual: r = (I + Π S) x + g
        residual_local = x_new + Pi_Sx + g_local
        
        # Global residual norm (need MPI reduction)
        local_norm_sq = np.vdot(residual_local, residual_local).real
        global_norm_sq = comm.allreduce(local_norm_sq, op=MPI.SUM)
        res_norm = np.sqrt(global_norm_sq)
        
        residuals.append(res_norm)
        
        if rank == 0 and k % 10 == 0:
            print(f"Iteration {k:4d}: residual = {res_norm:.6e}")
        
        if res_norm < tol:
            if rank == 0:
                print(f"Converged in {k+1} iterations!")
            return x_new, residuals, True
        
        x = x_new
    
    if rank == 0:
        print(f"Did not converge in {max_iter} iterations. Final residual: {residuals[-1]:.6e}")
    
    return x, residuals, False