import numpy as np

from src.seq.operators.pi_operator import PiOperator

def fixed_point_solver(g: np.ndarray, S_op, Pi_op : PiOperator, omega: float, 
                       max_iter: int = 1000, tol: float = 1e-10) -> tuple[np.ndarray, list, bool]:
    """
    Solve the DDM interface linear system using Richardson (Fixed-Point) iteration.

    This function solves the global interface problem:
        (I + Π S) x = -g
    
    It uses a damped fixed-point iteration scheme (Richardson iteration) to find 
    the equilibrium state of the interface variables.

    Iteration Scheme
    ----------------
    Calculates the residual r_k and updates solution x_k:
        r_k = (I + Π S) x_k + g
        x_{k+1} = x_k - ω * r_k

    Parameters
    ----------
    g : np.ndarray
        The global interface source vector (right-hand side).
    S_op : callable
        Operator function S(x) that computes the subdomain response. 
        Mathematically: S = diag(S_1, ..., S_J).
    Pi_op : callable
        Operator function Pi(x) that handles the exchange of data between 
        neighboring subdomains (permutation/communication).
    omega : float
        Relaxation parameter (damping factor). Controls convergence speed 
        and stability. 0 < omega <= 1 is typical.
    max_iter : int, optional
        Maximum number of iterations allowed (default: 1000).
    tol : float, optional
        Convergence tolerance for the residual norm (default: 1e-10).

    Returns
    -------
    x : np.ndarray
        The converged solution vector for the interface variables.
    residuals : list
        History of the L2 norm of the residual at each iteration.
    converged : bool
        True if the residual norm dropped below 'tol', False otherwise.
    """
    x = np.zeros_like(g)
    residuals = []
    converged = False
    
    for _ in range(max_iter):
        # Apply the linear operator A = (I + Pi S)
        Sx = S_op(x)
        PSx = Pi_op.apply(Sx)
        
        # Calculate residual: r = Ax - b = (I + Pi S)x + g
        # Note: We are solving Ax = -g, so Ax + g = 0
        residual = x + PSx + g
        
        res_norm = np.linalg.norm(residual)
        residuals.append(res_norm)
        
        if res_norm < tol:
            converged = True 
            break
        
        # Richardson update step
        x = x - omega * residual
    
    return x, residuals, converged