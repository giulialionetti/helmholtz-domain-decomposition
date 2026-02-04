import os
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

import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix
import numpy as np
from mpi4py import MPI

from src.seq.mesh.mesh import FullBoundary, FullMesh
from src.common.helmholtz.helmholtz_param import HelmholtzParameters
from src.seq.operators.u_operator import uj_solution
from src.seq.operators.b_operator import FullBOperator
from src.seq.operators.a_operator import FullAOperator
from src.seq.operators.t_operator import FullTOperator
from src.seq.operators.c_operator import FullCOperator
from src.seq.operators.s_operator import FullSFactorization
from src.par.operators.s_operator import SparseSOperator
from src.par.operators.pi_operator import SparsePiOperator
from src.seq.operators.bv_operator import FullBVecOperator
from src.par.operators.g_operator import SparseGVecOperator
from src.par.operators.mpi_linear_operator import MPILinearOperator
from src.common.helmholtz.helmholtz_param import HelmholtzParameters
from src.par.linear_solver.fixed_point import fixed_point_solver_mpi


class HelmholtzSolverMPI:
    """
    MPI-distributed Helmholtz DDM solver.
    
    Each MPI process handles exactly ONE subdomain j = rank.
    Local operators (A, B, T) are built independently.
    Global operators (S, Π, g) coordinate via MPI communication.
    """
    
    def __init__(self, comm: MPI.Comm, params, J: int):
        """
        Parameters
        ----------
        comm : MPI.Comm
            MPI communicator
        params : HelmholtzParameters
            Physical parameters (kappa, Lx, Ly, source points, etc.)
        J : int
            Total number of subdomains
        """
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()
        self._params = params
        self._J = J
        
        # Verify MPI setup
        assert self._size == J, \
            f"Number of MPI processes ({self._size}) must equal subdomains ({J})"
        
        # Solution storage
        self._x_local = None
        self._residuals = []
        
        # Log only from rank 0
        self._verbose = (self._rank == 0)
    
    def assembly(self):
        """
        Build all DDM components for the LOCAL subdomain j = rank.
        
        Each process constructs:
        - Local mesh and boundary info
        - Local operators: A_j, B_j, T_j, b_j
        - Local S-factorization
        - Global operators (MPI-aware): S, Π, g
        """
        j = self._rank  # This process handles subdomain j
        
        if self._verbose:
            print("="*60)
            print(f"Assembling DDM components for {self._J} subdomains...")
            print("="*60)
        
        # Import local operator classes (your existing Full* classes)
        
        # ============================
        # 1. Build LOCAL mesh/boundary
        # ============================
        self._mesh = FullMesh(
            self._J, 
            self._params.nx, 
            self._params.ny, 
            self._params.Lx, 
            self._params.Ly
        )
        self._boundary = FullBoundary(self._J, self._params.nx, self._mesh)
        
        self._mesh.build()
        self._boundary.build()
        
        # ============================
        # 2. Build LOCAL operators
        # ============================
        # These are standard - each process builds only for j = rank
        self._B = FullBOperator(self._J, self._mesh, self._boundary)
        self._C = FullCOperator(self._J, self._mesh)
        self._T = FullTOperator(
            self._J, self._mesh, self._boundary, 
            self._B, self._params
        )
        self._A = FullAOperator(
            self._J, self._mesh, self._boundary, self._params
        )
        self._s_factorization = FullSFactorization(
            self._J, self._A, self._T, self._B
        )
        self._BVec = FullBVecOperator(self._J, self._mesh, self._params)
        
        # Build only local block (j = rank)
        if self._verbose:
            print(f"Building local operators for subdomain {j}...")
        
        self._B.buildLocal(j)
        self._C.buildLocal(j)
        self._A.buildLocal(j)
        self._T.buildLocal(j)
        self._s_factorization.buildLocal(j)
        self._BVec.buildLocal(j)
        
        # ============================
        # 3. Build MPI-aware GLOBAL operators
        # ============================
        if self._verbose:
            print("Building global MPI operators (S, Π, g)...")
        
        # G vector operator (constructs local RHS)
        g_vec_op = SparseGVecOperator(self._comm)
        self._g_local = g_vec_op.applyGlobal(
            self._s_factorization, 
            self._BVec, 
            self._B, 
            self._C, 
            self._params.nx, 
            self._J
        )
        
        # Schur complement operator (MPI-distributed)
        self._S = SparseSOperator(
            self._comm, 
            self._J, 
            self._s_factorization, 
            self._B, 
            self._T, 
            self._C
        )
        
        # Exchange operator Π (communicates between neighbors)
        self._Pi = SparsePiOperator(self._comm, self._J, self._params.nx)
        
        if self._verbose:
            print("Assembly complete!")
            print("="*60)
    
    def solve_gmres(self, tol: float = 1e-10, maxiter: int = 500):
        """
        Solve the interface problem using GMRES.
        
        Solves: (I + Π S) x = -g
        
        Uses scipy's GMRES with MPI-aware linear operator.
        
        Parameters
        ----------
        tol : float
            Convergence tolerance
        maxiter : int
            Maximum GMRES iterations
            
        Returns
        -------
        x_local : np.ndarray
            Local solution on interface
        residuals : list
            Residual history
        info : int
            GMRES convergence info (0 = success)
        """
        if self._verbose:
            print("\n" + "="*60)
            print("Solving interface system with GMRES...")
            print("="*60)
        
        # Create MPI-aware linear operator
        local_size = len(self._g_local)
        A_op = MPILinearOperator(
            self._comm, 
            self._S, 
            self._Pi, 
            local_size
        )
        
        # Callback to track residuals
        self._residuals = []
        def callback(rk):
            self._residuals.append(rk)
            if self._verbose and len(self._residuals) % 10 == 0:
                # print(f"  Iteration {len(self._residuals):4d}: residual = {rk:.6e}")
                pass
        
        # Solve with GMRES
        x_local, info = spla.gmres(
            A_op, 
            -self._g_local, 
            rtol=tol,
            atol=0,
            maxiter=maxiter,
            callback=callback,
            callback_type='pr_norm'
        )
        print(f"Rank {self._rank}: here")
        
        self._x_local = x_local
        
        if self._verbose:
            if info == 0:
                print(f"\n✓ GMRES converged in {len(self._residuals)} iterations")
            else:
                print(f"\n✗ GMRES did not converge (info={info})")
            print(f"  Final residual: {self._residuals[-1]:.6e}")
            print("="*60)
        
        return x_local, self._residuals, info
    
    def solve_fixed_point(self, omega: float = 0.8, 
                         tol: float = 1e-8, 
                         maxiter: int = 500):
        """
        Solve the interface problem using fixed-point iteration.
        
        Iteration: x^{k+1} = ω(-g - Π S x^k) + (1-ω) x^k
        
        Parameters
        ----------
        omega : float
            Relaxation parameter (0 < ω ≤ 1)
        tol : float
            Convergence tolerance
        maxiter : int
            Maximum iterations
            
        Returns
        -------
        x_local : np.ndarray
            Local solution on interface
        residuals : list
            Residual history
        converged : bool
            Whether iteration converged
        """
        if self._verbose:
            print("\n" + "="*60)
            print(f"Solving with fixed-point (ω={omega})...")
            print("="*60)
        
        x_local, residuals, converged = fixed_point_solver_mpi(
            self._comm,
            self._g_local,
            self._S,
            self._Pi,
            omega,
            maxiter,
            tol
        )
        
        self._x_local = x_local
        self._residuals = residuals
        
        return x_local, residuals, converged
    
    def reconstruct_local_solution(self) -> np.ndarray:
        """
        Reconstruct the full local solution u_j from interface values x.
        
        Uses: u_j = S_j^{-1} (b_j - i B_j^T T_j x_j)
        
        Returns
        -------
        u_j : np.ndarray
            Local solution on subdomain Ω_j
        """
        if self._x_local is None:
            raise RuntimeError("Must call solve() before reconstructing solution")
        
        j = self._rank
        
        # Extract local interface values
        xj = self._C.applyLocal(j, self._x_local)
        
        # Reconstruct full local solution
        u_j = uj_solution(
            xj,
            self._s_factorization.getBlock(j),
            self._B.getBlock(j),
            self._T.getBlock(j),
            self._BVec.getBlock(j)
        )
        
        return u_j
    
    def get_local_mesh(self):
        """Get local mesh vertices and elements for plotting."""
        return self._mesh.getLocal(self._rank)
    
    def get_components(self):
        """Return all solver components (for debugging/analysis)."""
        return {
            's_factorization': self._s_factorization,
            'B': self._B,
            'C': self._C,
            'T': self._T,
            'BVec': self._BVec,
            'mesh': self._mesh,
            'boundary': self._boundary,
            'g_local': self._g_local,
            'S': self._S,
            'Pi': self._Pi,
            'x_local': self._x_local
        }
