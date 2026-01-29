"""
Basic routines used for system assembly for the 2D Helmholtz problem.
These routines are used both in the sequential and in the parallel setting
"""

from math import pi
import numpy as np
na = np.newaxis
import numpy.linalg as la
from scipy.sparse import csr_matrix

from src.common.mesh import get_area


def mass(vtx, elt):
    """
    Builds the mass matrix of the discretized 2d Helmholtz problem, 
    given the vertices and the triangles
    
    :param vtx: vertices of the mesh
    :param elt: triangles of the mesh
    """
    nv = np.size(vtx, 0)
    d = np.size(elt, 1)
    areas = get_area(vtx, elt)
    M = csr_matrix((nv, nv), dtype=np.float64)
    for j in range(d):
        for k in range(d):
           row = elt[:,j]
           col = elt[:,k]
           val = areas * (1 + (j == k)) / (d*(d+1))
           M += csr_matrix((val, (row, col)), shape=(nv, nv))
    return M

def stiffness(vtx, elt):
    """
    Builds the stiffness matrix of the discretized 2d Helmholtz problem, 
    given the vertices and the triangles
    
    :param vtx: vertices of the mesh
    :param elt: triangles of the mesh
    """
    nv = np.size(vtx, 0)
    d = np.size(elt, 1)
    areas = get_area(vtx, elt)
    ne, d = np.shape(elt)
    E = np.empty((ne, d, d-1), dtype=np.float64)
    E[:,0,:] = 0.5 * (vtx[elt[:,1],0:2] - vtx[elt[:,2],0:2])
    E[:,1,:] = 0.5 * (vtx[elt[:,2],0:2] - vtx[elt[:,0],0:2])
    E[:,2,:] = 0.5 * (vtx[elt[:,0],0:2] - vtx[elt[:,1],0:2])
    K = csr_matrix((nv, nv), dtype=np.float64)
    for j in range(d):
        for k in range(d):
           row = elt[:,j]
           col = elt[:,k]
           val = np.sum(E[:,j,:] * E[:,k,:], axis=1) / areas
           K += csr_matrix((val, (row, col)), shape=(nv, nv))
    return K

def point_source(sp, k):    
    """
    Builds the right hand side for the linear of the discretized 2D Helmholtz problem 
    """
    def ps(x):
        v = np.zeros(np.size(x,0), float)
        for s in sp:
            v += s[2]*np.exp(-10*(k/(2.0*pi))**2 * la.norm(x - s[na,0:2], axis=1)**2)
        return v
    return ps 
