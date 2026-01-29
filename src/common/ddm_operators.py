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
from src.common.system_assembly import mass, stiffness, point_source



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
    Construct the global right-hand side vector 'g' for the interface linear system.

    In the DDM formulation, 'g' represents the contribution of the physical 
    volume sources (the RHS of the Helmholtz equation) projected onto the 
    subdomain interfaces.

    Mathematical Definition
    -----------------------
    g = Π ( sum_j C_j^T B_j A_j^{-1} b_j )

    Step-by-step construction:
    1. Solve local problems with zero interface input but full volume source:
       u_j^{local} = (A_{local})^{-1} b_j
    2. Restrict solution to the boundary:
       v_j = B_j u_j^{local}
    3. Assemble into global vector:
       v_{global} = sum (C_j^T v_j)
    4. Apply exchange operator:
       g = Π v_{global}

    Parameters
    ----------
    factorizations : list[scipy.sparse.linalg.SuperLU]
        List of pre-computed LU factorizations for each subdomain operator.
    bj_list : list[np.ndarray]
        List of local volume source vectors (b_j) for each subdomain.
    Bj_list : list[scipy.sparse.csr_matrix]
        List of restriction matrices mapping volume DOFs to boundary DOFs.
    Cj_list : list[scipy.sparse.csr_matrix]
        List of boolean matrices mapping local boundary DOFs to the global 
        interface vector.
    nx : int
        Number of grid points in the x-direction (defines interface size).
    J : int
        Total number of subdomains.

    Returns
    -------
    g : np.ndarray
        The global interface source vector (size: 2 * (J-1) * nx).
    """
    # Determine skeleton size: (J-1) interfaces, 2 sides per interface
    n_skeleton = 2 * (J - 1) * nx
    g_temp = np.zeros(n_skeleton, dtype=complex)
    
    for j in range(J):
        # 1. Solve local problem driven ONLY by volume source bj
        #    (Aj - i B_j^T Tj Bj) uj = bj
        uj = factorizations[j].solve(bj_list[j])
        
        # 2. Extract trace on the boundary: Bj @ uj
        interface_vals = Bj_list[j] @ uj
        
        # 3. Map local boundary values to global position
        g_temp += Cj_list[j].T @ interface_vals
    
    # 4. Apply exchange operator to finalize g
    g = Pi_operator(g_temp, nx, J)
    
    return g




def uj_solution(xj: np.ndarray, LU_j, Bj: csr_matrix, 
                Tj: csr_matrix, bj: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full local solution u_j using the converged interface data x_j.

    This function acts as the final step of the DDM solver. Once the interface 
    variables x_j (incoming impedance traces) are found, they are used as 
    boundary conditions to solve the local volumetric problem one last time.

    Mathematical Formulation
    ------------------------
    The local system being solved is:
        A_{local} u_j = b_j + B_j^T T_j x_j

    Where:
        u_j = A_{local}^{-1} (b_j + B_j^T T_j x_j)

    Parameters
    ----------
    xj : np.ndarray
        The converged interface data (incoming Robin traces) for this subdomain.
    LU_j : scipy.sparse.linalg.SuperLU
        The pre-computed LU factorization of the local operator 
        (A_j + B_j^T T_j B_j).
    Bj : scipy.sparse.csr_matrix
        The restriction matrix mapping volume DOFs to boundary DOFs.
    Tj : scipy.sparse.csr_matrix
        The transmission (impedance) matrix.
    bj : np.ndarray
        The original local volume source vector (right-hand side).

    Returns
    -------
    np.ndarray
        The fully reconstructed solution vector u_j on the subdomain mesh.
    """
    
    # Map interface data back to volume source terms
    rhs = bj + Bj.T @ (Tj @ xj)
    
    # Solve the local volume problem
    uj = LU_j.solve(rhs)
    
    return uj