#!/usr/bin/env python3
"""
Domain Decomposition Method for 2D Helmholtz Equation
Project 2: Sequential Implementation
"""

from math import pi
import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, csc_matrix, eye
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from src.helmholtz_base import mesh, boundary, mass, stiffness, point_source, plot_mesh



def local_mesh(Lx: float, Ly: float, 
               nx: int, ny: int, 
               j: int, J: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the local mesh for subdomain j.
    
    Parameters:
    -----------
    Lx, Ly : float
        Domain dimensions
    nx, ny : int
        Number of points in x and y directions for the GLOBAL mesh
    j : int
        Subdomain index (0 <= j < J)
    J : int
        Total number of subdomains
    
    Returns:
    --------
    vtxj : ndarray
        Local vertex coordinates
    eltj : ndarray
        Local triangle connectivity
    
    Notes:
    ------
    - The domain is split in the y-direction into J horizontal slabs
    - Each subdomain has size (0, Lx) X (j*Ly/J, (j+1)*Ly/J)
    - We assume (ny - 1) is divisible by J
    """
    # Number of y-intervals per subdomain
    ny_per_subdomain = (ny - 1) // J
    
    # Local number of points in y direction
    ny_local = ny_per_subdomain + 1
    
    # Local domain dimensions
    Ly_local = Ly / J
    
    # Y-offset for subdomain j
    y_offset = j * Ly_local
    
    # Generate local mesh
    vtxj, eltj = mesh(nx, ny_local, Lx, Ly_local)
    
    # Shift vertices to correct y-position
    vtxj[:, 1] += y_offset
    
    return vtxj, eltj


def local_boundary(nx: int, ny: int,
                   j: int, J: int) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Construct boundary edge arrays for subdomain j.
    
    Parameters:
    -----------
    nx, ny : int
        Number of points in x and y directions for LOCAL mesh
    j : int
        Subdomain index (0 <= j < J)
    J : int
        Total number of subdomains
    
    Returns:
    --------
    beltj_phys : ndarray
        Physical boundary edges (∂Ωj ∩ ∂Ω)
    beltj_artf : ndarray
        Artificial interface edges (∂Ωj \ ∂Ω)
    
    Notes:
    ------
    - Physical boundaries are: bottom (j=0), top (j=J-1), left, right (all j)
    - Artificial interfaces are: bottom (j>0), top (j<J-1)
    """
    na = np.newaxis
    
    # Bottom boundary (y=0 for local mesh)
    bottom = np.hstack((np.arange(0, nx-1, 1)[:, na],
                        np.arange(1, nx, 1)[:, na]))
    
    # Top boundary (y=Ly_local for local mesh)
    top = np.hstack((np.arange(nx*(ny-1), nx*ny-1, 1)[:, na],
                     np.arange(nx*(ny-1)+1, nx*ny, 1)[:, na]))
    
    # Left boundary (x=0)
    left = np.hstack((np.arange(0, nx*(ny-1), nx)[:, na],
                      np.arange(nx, nx*ny, nx)[:, na]))
    
    # Right boundary (x=Lx)
    right = np.hstack((np.arange(nx-1, nx*(ny-1), nx)[:, na],
                       np.arange(2*nx-1, nx*ny, nx)[:, na]))
    
    # Determine physical vs artificial boundaries
    beltj_phys_list = []
    beltj_artf_list = []
    
    # Bottom: physical if j==0, artificial otherwise
    if j == 0:
        beltj_phys_list.append(bottom)
    else:
        beltj_artf_list.append(bottom)
    
    # Top: physical if j==J-1, artificial otherwise
    if j == J - 1:
        beltj_phys_list.append(top)
    else:
        beltj_artf_list.append(top)
    
    # Left and right are always physical
    beltj_phys_list.extend([left, right])
    
    # Concatenate arrays
    beltj_phys = np.vstack(beltj_phys_list) if beltj_phys_list else np.array([]).reshape(0, 2)
    beltj_artf = np.vstack(beltj_artf_list) if beltj_artf_list else np.array([]).reshape(0, 2)
    
    
    
    return beltj_phys, beltj_artf

def Rj_matrix(nx: int, ny: int, j: int, J: int) -> csr_matrix:
    r"""
    Construct local restriction matrix Rj.
    
    Rj maps global degrees of freedom V(Ω) to local degrees of freedom V(Ωj).
    
    Parameters:
    -----------
    nx, ny : int
        Number of points for GLOBAL mesh
    j : int
        Subdomain index
    J : int
        Total number of subdomains
    
    Returns:
    --------
    Rj : sparse matrix
        Restriction matrix of size |V(Ωj)| X |V(Ω)|
    
    Notes:
    ------
    - This is a boolean matrix (0s and 1s)
    - Each row has exactly one 1, selecting one global DOF
    - The ordering follows the local mesh numbering
    """
    # Number of y-intervals per subdomain
    ny_per_subdomain = (ny - 1) // J
    
    # Local number of points
    ny_local = ny_per_subdomain + 1
    nv_local = nx * ny_local
    nv_global = nx * ny
    
    # Starting y-index for subdomain j in global mesh
    y_start = j * ny_per_subdomain
    
    # Build mapping from local to global indices
    row_indices = []
    col_indices = []
    
    for iy_local in range(ny_local):
        iy_global = y_start + iy_local
        for ix in range(nx):
            local_idx = ix + iy_local * nx
            global_idx = ix + iy_global * nx
            row_indices.append(local_idx)
            col_indices.append(global_idx)
    
    # Create sparse restriction matrix
    data = np.ones(len(row_indices))
    Rj = csr_matrix((data, (row_indices, col_indices)), shape=(nv_local, nv_global))
    
    return Rj
    


def Bj_matrix(nx: int, ny: int, j: int, J: int, beltj_artf: np.ndarray) -> csr_matrix:
    r"""
    Construct interface restriction matrix Bj.
    
    Bj maps local degrees of freedom V(Ωj) to interface DOFs V(Σj).
    
    Parameters:
    -----------
    nx, ny : int
        Number of points for LOCAL mesh
    j : int
        Subdomain index
    J : int
        Total number of subdomains
    beltj_artf : ndarray
        Artificial boundary edges
    
    Returns:
    --------
    Bj : sparse matrix
        Restriction matrix of size |V(Σj)| X |V(Ωj)|
    
    Notes:
    ------
    - Σj = ∂Ωj \ ∂Ω (artificial interfaces only)
    - This extracts interface DOFs from the local solution
    """
    nv_local = nx * ny
    
    if len(beltj_artf) == 0:
        return csr_matrix((0, nv_local))
    
    # Extract unique vertex indices on artificial interfaces
    interface_vertices = np.unique(beltj_artf.flatten())
    n_interface = len(interface_vertices)
    
    # Build restriction matrix
    row_indices = np.arange(n_interface)
    col_indices = interface_vertices
    data = np.ones(n_interface)
    
    Bj = csr_matrix((data, (row_indices, col_indices)), shape=(n_interface, nv_local))
    
    return Bj

def Cj_matrix(nx: int, ny: int, j: int, J: int) -> csr_matrix:
    r"""
    Construct global interface restriction matrix Cj.
    
    Cj maps global interface vector x = (x1, x2, ..., xJ) to local part xj.
    
    Parameters:
    -----------
    nx, ny : int
        Number of points for GLOBAL mesh
    j : int
        Subdomain index
    J : int
        Total number of subdomains
    
    Returns:
    --------
    Cj : sparse matrix
        Restriction matrix of size |V(Σj)| X |V(S)|
        where S = Uj Σj is the skeleton (all interfaces)
    
    Notes:
    ------
    - This selects the portion of the global interface vector belonging to subdomain j
    - The global skeleton S consists of all interface vertices
    """
    # Total number of interface DOFs globally (J-1 interfaces, each with nx vertices)
    n_interface_total = (J - 1) * nx
    
    # Subdomain j has interfaces on:
    # - bottom (if j > 0): interface j-1
    # - top (if j < J-1): interface j
    
    row_indices = []
    col_indices = []
    current_row = 0
    
    # Bottom interface (if exists)
    if j > 0:
        interface_idx = j - 1
        for i in range(nx):
            row_indices.append(current_row)
            col_indices.append(interface_idx * nx + i)
            current_row += 1
    
    # Top interface (if exists)
    if j < J - 1:
        interface_idx = j
        for i in range(nx):
            row_indices.append(current_row)
            col_indices.append(interface_idx * nx + i)
            current_row += 1
    
    n_local_interface = current_row
    
    # Create sparse restriction matrix
    data = np.ones(len(row_indices))
    Cj = csr_matrix((data, (row_indices, col_indices)), 
                    shape=(n_local_interface, n_interface_total))
    
    return Cj


def Aj_matrix(vtxj, eltj, beltj_phys, kappa):
    """
    Construct local problem matrix Aj for subdomain j.
    
    Parameters:
    -----------
    vtxj : ndarray
        Local vertex coordinates
    eltj : ndarray
        Local triangle connectivity
    beltj_phys : ndarray
        Physical boundary edges
    kappa : float
        Wavenumber k
    
    Returns:
    --------
    Aj : sparse matrix
        Local Helmholtz operator K - k²M - ikMb
    """
    # TODO
    
    pass


def Tj_matrix(vtxj, beltj_artf, Bj, kappa):
    """
    Construct local transmission matrix Tj.
    
    Tj = κ * (mass matrix on artificial interfaces)
    
    Parameters:
    -----------
    vtxj : ndarray
        Local vertex coordinates
    beltj_artf : ndarray
        Artificial boundary edges
    Bj : sparse matrix
        Interface restriction matrix
    kappa : float
        Wavenumber k
    
    Returns:
    --------
    Tj : sparse matrix
        Transmission matrix on Σj
    """
    # TODO
    
    pass


def Sj_factorization(Aj, Tj, Bj):
    """
    Factorize the local problem matrix Aj - iB*j Tj Bj.
    
    Parameters:
    -----------
    Aj : sparse matrix
        Local problem matrix
    Tj : sparse matrix
        Transmission matrix
    Bj : sparse matrix
        Interface restriction matrix
    
    Returns:
    --------
    LU : SuperLU object
        LU factorization for efficient solves
    """
    # TODO
    
    pass


def bj_vector(vtxj, eltj, sp, kappa):
    """
    Construct local right-hand side vector bj.
    
    Parameters:
    -----------
    vtxj : ndarray
        Local vertex coordinates
    eltj : ndarray
        Local triangle connectivity
    sp : list
        Point source specifications
    kappa : float
        Wavenumber k
    
    Returns:
    --------
    bj : ndarray
        Local RHS vector
    """
    # TODO
    
    pass



def S_operator(x, factorizations, Bj_list, Tj_list, Cj_list):
    """
    Apply the global Schur complement operator S to vector x.
    
    S is block diagonal: S = diag(S1, S2, ..., SJ)
    where Sj xj = Bj (Aj - iB*j Tj Bj)^(-1) B*j Tj xj
    
    Parameters:
    -----------
    x : ndarray
        Global interface vector
    factorizations : list
        List of LU factorizations for each subdomain
    Bj_list : list
        List of Bj matrices
    Tj_list : list
        List of Tj matrices
    Cj_list : list
        List of Cj matrices
    
    Returns:
    --------
    Sx : ndarray
        Result of S @ x
    """
    # TODO
    
    pass


def Pi_operator(x, nx, J):
    """
    Apply the exchange operator Π to vector x.
    
    Π swaps interface values between neighboring subdomains.
    
    Parameters:
    -----------
    x : ndarray
        Global interface vector
    nx : int
        Number of points in x direction
    J : int
        Number of subdomains
    
    Returns:
    --------
    Px : ndarray
        Result of Π @ x
    
    Notes:
    ------
    - Due to the horizontal slab decomposition, exchange is simple
    - Each interior interface is shared by exactly 2 subdomains
    - We swap the values at these shared vertices
    """
    # TODO
    
    pass


def g_vector(factorizations, bj_list, Bj_list, Cj_list, nx, J):
    """
    Construct global right-hand side g for the interface problem.
    
    g = ΠS b where b = (b1, b2, ..., bJ) with
    bj = Bj (Aj - iB*j Tj Bj)^(-1) bj
    
    Parameters:
    -----------
    factorizations : list
        List of LU factorizations
    bj_list : list
        List of local RHS vectors
    Bj_list : list
        List of Bj matrices
    Cj_list : list
        List of Cj matrices
    nx : int
        Number of points in x direction
    J : int
        Number of subdomains
    
    Returns:
    --------
    g : ndarray
        Global interface RHS
    """
    # TODO
    
    pass




def fixed_point_solver(g, S_op, Pi_op, omega, max_iter=1000, tol=1e-10):
    """
    Solve interface problem using fixed-point iteration.
    
    Iteration: x^(n+1) = x^n + ω(Π S x^n + g)
    
    Parameters:
    -----------
    g : ndarray
        RHS vector
    S_op : callable
        Function implementing S @ x
    Pi_op : callable
        Function implementing Π @ x
    omega : float
        Relaxation parameter (0 < ω < 1)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    
    Returns:
    --------
    x : ndarray
        Solution
    residuals : list
        Residual history
    """
    # TODO
    pass


def uj_solution(xj, LU_j, Bj, Tj, bj):
    """
    Compute local solution uj from interface solution xj.
    
    uj = (Aj - iB*j Tj Bj)^(-1) (bj - B*j Tj xj)
    
    Parameters:
    -----------
    xj : ndarray
        Local interface solution
    LU_j : SuperLU object
        Factorized local matrix
    Bj : sparse matrix
        Interface restriction matrix
    Tj : sparse matrix
        Transmission matrix
    bj : ndarray
        Local RHS
    
    Returns:
    --------
    uj : ndarray
        Local solution vector
    """
    # TODO: Implement this function
    pass


