import numpy as np
import numpy.linalg as la

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

na = np.newaxis

class Mesh:
    def __init__(self, J: int, nx: int, ny: int, Lx: float, Ly: float):
        if Lx <= 0 or Ly <= 0:
            raise ValueError(f"Domain dimensions must be positive: Lx={Lx}, Ly={Ly}")
        
        if nx < 2 or ny < 2:
            raise ValueError(f"Mesh must have at least 2 points: nx={nx}, ny={ny}")
        
        if (ny - 1) % J != 0:
            raise ValueError(
                f"Cannot evenly divide mesh: ny-1={ny-1} not divisible by J={J}\n"
                f"Try ny={J * ((ny-1)//J + 1) + 1} or J={J-1}"
            )

        self._num_domains = J
        self._nx = nx
        self._ny = ny
        self._Lx = Lx
        self._Ly = Ly

    def getLocal(self, j: int):
        raise NotImplementedError("This is an abstract class")

    def build(self):
        raise NotImplementedError("This is an abstract class")

class Boundary:
    def __init__(self, J: int, nx: int):
        self._num_domains = J
        self._nx = nx

    def getLocal(self, j: int):
        raise NotImplementedError("This is an abstract class")

    def build(self):
        raise NotImplementedError("This is an abstract class")

# ============================= INTERNAL ROUTINES for MESH ==============================
def mesh(nx,ny,Lx,Ly):
   i = np.arange(0,nx)[na,:] * np.ones((ny,1), np.int64)
   j = np.arange(0,ny)[:,na] * np.ones((1,nx), np.int64)
   p = np.zeros((2,ny-1,nx-1,3), np.int64)
   q = i+nx*j
   p[:,:,:,0] = q[:-1,:-1]
   p[0,:,:,1] = q[1: ,1: ]
   p[0,:,:,2] = q[1: ,:-1]
   p[1,:,:,1] = q[:-1,1: ]
   p[1,:,:,2] = q[1: ,1: ]
   v = np.concatenate(((Lx/(nx-1)*i)[:,:,na], (Ly/(ny-1)*j)[:,:,na]), axis=2)
   vtx = np.reshape(v, (nx*ny,2))
   elt = np.reshape(p, (2*(nx-1)*(ny-1),3))
   return vtx, elt 

def boundary(nx, ny):
    bottom = np.hstack((np.arange(0,nx-1,1)[:,na],
                        np.arange(1,nx,1)[:,na]))
    top    = np.hstack((np.arange(nx*(ny-1),nx*ny-1,1)[:,na],
                        np.arange(nx*(ny-1)+1,nx*ny,1)[:,na]))
    left   = np.hstack((np.arange(0,nx*(ny-1),nx)[:,na],
                        np.arange(nx,nx*ny,nx)[:,na]))
    right  = np.hstack((np.arange(nx-1,nx*(ny-1),nx)[:,na],
                        np.arange(2*nx-1,nx*ny,nx)[:,na]))
    return {
        'bottom': bottom,
        'top': top,
        'left': left,
        'right': right
    }


# ============================== FINAL ROUTINES for MESH ===============================
def get_area(vtx, elt):
    d = np.size(elt, 1)
    if d == 2:
        e = vtx[elt[:, 1], :] - vtx[elt[:, 0], :]
        areas = la.norm(e, axis=1)
    else:
        e1 = vtx[elt[:, 1], :] - vtx[elt[:, 0], :]
        e2 = vtx[elt[:, 2], :] - vtx[elt[:, 0], :]
        areas = 0.5 * np.abs(e1[:,0] * e2[:,1] - e1[:,1] * e2[:,0])
    return areas

def plot_mesh(vtx, elt, val=None, **kwargs):
    trig = mtri.Triangulation(vtx[:,0], vtx[:,1], elt)
    if val is None:
        plt.triplot(trig, **kwargs)
    else:
        plt.tripcolor(trig, val,
                      shading='gouraud',
                      cmap=cm.jet, **kwargs)
    plt.axis('equal')


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