import numpy as np
from src.common.mesh import Mesh, Boundary, mesh, local_mesh, boundary, local_boundary

class FullMesh(Mesh):
    def __init__(self, J: int, nx: int, ny: int, Lx: float, Ly: float):
        super(FullMesh, self).__init__(J, nx, ny, Lx, Ly)

        self._mesh_list = [(np.zeros(0), np.zeros(0))] * J

    def getLocal(self, j: int):
        return self._mesh_list[j]
    
    # def getGlobal(self):
    #     pass

    def build(self):
        for j in range(self._num_domains):
            # Number of y-intervals per subdomain
            ny_per_subdomain = (self._ny - 1) // self._num_domains
            
            # Local number of points in y direction
            ny_local = ny_per_subdomain + 1
            
            # Local domain dimensions
            Ly_local = self._Ly / self._num_domains
            
            # Y-offset for subdomain j
            y_offset = j * Ly_local
            
            # Generate local mesh
            vtxj, eltj = mesh(self._nx, ny_local, self._Lx, Ly_local)
            
            # Shift vertices to correct y-position
            vtxj[:, 1] += y_offset
            self._mesh_list[j] = (vtxj, eltj)
        
        
class FullBoundary(Boundary):
    def __init__(self, J: int, nx: int, mesh: Mesh):
        super(FullBoundary, self).__init__(J, nx)

        self._boundary_list = [(np.zeros(0), np.zeros(0))] * J
        self._mesh = mesh

    def getLocal(self, j: int):
        return self._boundary_list[j]
    
    def build(self):
        for j in range(self._num_domains):
            vtxj = self._mesh.getLocal(j)[0]
            ny_local = len(np.unique(vtxj[:, 1]))

            edges = boundary(self._nx, ny_local)
            
            beltj_phys_list = []
            beltj_artf_list = []
            
            # Bottom: physical if j==0, artificial otherwise
            if j == 0:
                beltj_phys_list.append(edges['bottom'])
            else:
                beltj_artf_list.append(edges['bottom'])
            
            # Top: physical if j==J-1, artificial otherwise
            if j == self._num_domains - 1:
                beltj_phys_list.append(edges['top'])
            else:
                beltj_artf_list.append(edges['top'])
            
            # Left and right are always physical
            beltj_phys_list.extend([edges['left'], edges['right']])
            
            beltj_phys = np.vstack(beltj_phys_list) if beltj_phys_list else np.array([]).reshape(0, 2)
            beltj_artf = np.vstack(beltj_artf_list) if beltj_artf_list else np.array([]).reshape(0, 2)

            self._boundary_list[j] = (beltj_phys, beltj_artf)
