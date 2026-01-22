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
    if Lx <= 0 or Ly <= 0:
        raise ValueError(f"Domain dimensions must be positive: Lx={Lx}, Ly={Ly}")
    
    if nx < 2 or ny < 2:
        raise ValueError(f"Mesh must have at least 2 points: nx={nx}, ny={ny}")
    
    if j < 0 or j >= J:
        raise ValueError(f"Subdomain index j={j} out of range [0, {J-1}]")
    
    if (ny - 1) % J != 0:
        raise ValueError(
            f"Cannot evenly divide mesh: ny-1={ny-1} not divisible by J={J}\n"
            f"Try ny={J * ((ny-1)//J + 1) + 1} or J={J-1}"
        )
    
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
    edges = boundary(nx, ny)
    
    beltj_phys_list = []
    beltj_artf_list = []
    
    # Bottom: physical if j==0, artificial otherwise
    if j == 0:
        beltj_phys_list.append(edges['bottom'])
    else:
        beltj_artf_list.append(edges['bottom'])
    
    # Top: physical if j==J-1, artificial otherwise
    if j == J - 1:
        beltj_phys_list.append(edges['top'])
    else:
        beltj_artf_list.append(edges['top'])
    
    # Left and right are always physical
    beltj_phys_list.extend([edges['left'], edges['right']])
    
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
    """
    Construct global interface restriction matrix Cj with 2-sided interfaces.
    
    Mapping logic:
    - If Subdomain j has a BOTTOM interface (connecting to j-1):
      It maps to Interface j-1, Side 1 (the 'Down' side belonging to j).
    - If Subdomain j has a TOP interface (connecting to j+1):
      It maps to Interface j, Side 0 (the 'Up' side belonging to j).
    """
    # Total global interface DOFs: (J-1) interfaces * 2 sides * nx points
    n_interface_total = 2 * (J - 1) * nx
    
    row_indices = []
    col_indices = []
    current_row = 0
    
    # --- 1. Bottom Interface (if exists) ---
    # This is Global Interface (j-1). We are on the top side of it.
    if j > 0:
        interface_idx = j - 1
        # Block index for "Side 1" of interface_idx is: 2 * interface_idx + 1
        global_start_idx = (2 * interface_idx + 1) * nx
        
        for i in range(nx):
            row_indices.append(current_row)
            col_indices.append(global_start_idx + i)
            current_row += 1
    
    # --- 2. Top Interface (if exists) ---
    # This is Global Interface (j). We are on the bottom side of it.
    if j < J - 1:
        interface_idx = j
        # Block index for "Side 0" of interface_idx is: 2 * interface_idx
        global_start_idx = (2 * interface_idx) * nx
        
        for i in range(nx):
            row_indices.append(current_row)
            col_indices.append(global_start_idx + i)
            current_row += 1
            
    n_local_interface = current_row
    
    # Create sparse restriction matrix
    # Note: If a subdomain has no interfaces (J=1), this returns empty
    if n_local_interface > 0:
        data = np.ones(len(row_indices))
        Cj = csr_matrix((data, (row_indices, col_indices)), 
                        shape=(n_local_interface, n_interface_total))
    else:
        Cj = csr_matrix((0, n_interface_total))
        
    return Cj


def Aj_matrix(vtxj: np.ndarray, eltj: np.ndarray, 
              beltj_phys: np.ndarray, kappa: float) -> csr_matrix:
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
    # Build local mass and stiffness matrices
    M = mass(vtxj, eltj)
    K = stiffness(vtxj, eltj)
    
    # Build boundary mass matrix (only on physical boundaries)
    if len(beltj_phys) > 0:
        Mb = mass(vtxj, beltj_phys)
    else:
        Mb = csr_matrix((len(vtxj), len(vtxj)))
    
    # Construct Helmholtz operator: A = K - k²M - ikMb
    Aj = K - kappa**2 * M - 1j * kappa * Mb
    
    return csr_matrix(Aj)


def Tj_matrix(vtxj: np.ndarray, beltj_artf: np.ndarray, 
              Bj: csr_matrix, kappa: float) -> csr_matrix:
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
    if len(beltj_artf) == 0:
        # No artificial interfaces
        return csr_matrix((0, 0))
    
    # Build mass matrix on artificial interfaces
    M_interface = mass(vtxj, beltj_artf)
    
    # Restrict to interface DOFs: Tj = κ * Bj @ M_interface @ Bj^T
    Tj = kappa * (Bj @ M_interface @ Bj.T)
    
    return csr_matrix(Tj)


def Sj_factorization(Aj: csr_matrix, Tj: csr_matrix, Bj: csr_matrix):
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
    # Construct modified local matrix: Aj - i * Bj^T @ Tj @ Bj
    if Bj.shape[0] > 0: # If there are artificial interfaces
        modified_Aj = Aj - 1j * (Bj.T @ Tj @ Bj) 
    else:
        modified_Aj = Aj # No modification needed
    
    # LU factorization
    LU = spla.splu(csc_matrix(modified_Aj))
    
    return LU


def bj_vector(vtxj: np.ndarray, eltj: np.ndarray, 
              sp: list, kappa: float) -> np.ndarray:
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
    # Build mass matrix
    M = mass(vtxj, eltj)
    
    # Evaluate point sources at local vertices
    f = point_source(sp, kappa)(vtxj)
    
    # RHS: bj = M @ f
    bj = M @ f
    
    return bj



def S_operator(x: np.ndarray, factorizations: list, Bj_list: list, 
               Tj_list: list, Cj_list: list) -> np.ndarray:
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
    J = len(factorizations)
    Sx = np.zeros_like(x)
    
    for j in range(J):
        # Extract local interface portion: xj = Cj @ x
        xj = Cj_list[j] @ x
        
        if len(xj) == 0:
            continue
        
        # Compute: Tj @ xj
        rhs = Tj_list[j] @ xj
        
        # Solve: (Aj - iB*j Tj Bj)^(-1) @ (B*j @ Tj @ xj)
        local_sol = factorizations[j].solve(Bj_list[j].T @ rhs)
        
        # Apply: Bj @ local_sol
        Sj_xj = Bj_list[j] @ local_sol
        
        # Assemble back to global skeleton: C*j @ Sj_xj
        Sx += Cj_list[j].T @ Sj_xj
    
    return Sx


def Pi_operator(x: np.ndarray, nx: int, J: int) -> np.ndarray:
    """
    Apply the exchange operator Π to vector x.
    
    For every interface k:
    - Swaps Block 2k (Side 0) with Block 2k+1 (Side 1).
    """
    Px = np.zeros_like(x)
    n_interfaces = J - 1
    
    for k in range(n_interfaces):
        # Indices for Side 0 (belonging to subdomain below)
        idx_side0 = slice((2 * k) * nx, (2 * k + 1) * nx)
        
        # Indices for Side 1 (belonging to subdomain above)
        idx_side1 = slice((2 * k + 1) * nx, (2 * k + 2) * nx)
        
        # Perform Swap
        Px[idx_side0] = x[idx_side1]
        Px[idx_side1] = x[idx_side0]
        
    return Px


def g_vector(factorizations: list, bj_list: list, Bj_list: list, 
             Cj_list: list, nx: int, J: int) -> np.ndarray:
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
    # Determine skeleton size
    n_skeleton = 2 * (J - 1) * nx
    g = np.zeros(n_skeleton, dtype=complex)
    
    for j in range(J):
        # Solve local problem: (Aj - iB*j Tj Bj)^(-1) @ bj
        uj = factorizations[j].solve(bj_list[j])
        
        # Extract interface values: Bj @ uj
        interface_vals = Bj_list[j] @ uj
        
        # Assemble to global skeleton: C*j @ interface_vals
        g += Cj_list[j].T @ interface_vals
    
    # Apply exchange operator Π
    g = Pi_operator(g, nx, J)
    
    return g



def fixed_point_solver(g: np.ndarray, S_op, Pi_op, omega: float, 
                       max_iter: int = 1000, tol: float = 1e-10) -> tuple[np.ndarray, list, bool]:
    """
    Solve interface problem using fixed-point iteration.
    
    Iteration: x^(n+1) = x^n - ω((I + ΠS)x^n + g)
    
    Returns:
    --------
    x : ndarray
        Solution
    residuals : list
        Residual history
    converged : bool
        Whether the solver converged within max_iter or not
    """
    x = np.zeros_like(g)
    residuals = []
    converged = False
    
    for _ in range(max_iter): # _ for iteration count (not used)
        Sx = S_op(x)
        PSx = Pi_op(Sx)
        residual = x + PSx + g
        res_norm = np.linalg.norm(residual)
        residuals.append(res_norm)
        
        if res_norm < tol:
            converged = True 
            break
        
        x = x - omega * residual
    
    return x, residuals, converged 


def uj_solution(xj: np.ndarray, LU_j, Bj: csr_matrix, 
                Tj: csr_matrix, bj: np.ndarray) -> np.ndarray:
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
   
    rhs = bj + Bj.T @ (Tj @ xj)
    
    # Solve: (Aj - iB*j Tj Bj) uj = rhs
    uj = LU_j.solve(rhs)
    
    return uj