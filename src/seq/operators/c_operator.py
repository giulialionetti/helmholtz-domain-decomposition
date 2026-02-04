import numpy as np
from scipy.sparse import csr_matrix

from src.seq.operators.base_operators import FullRowBlockOperator
from src.common.operators.operators import COperator
from src.seq.mesh.mesh import FullMesh

class FullCOperator(COperator, FullRowBlockOperator[csr_matrix]):
    def __init__(self, num_blocks: int, mesh: FullMesh):
        super(FullCOperator, self).__init__(mesh=mesh, num_blocks=num_blocks)
        self._mesh = mesh
        
    def build(self):
        for j in range(self._num_blocks):
            # Total global interface DOFs: (J-1) interfaces * 2 sides * nx points
            nx_global = self._mesh.getNx()
            ny_global = self._mesh.getNy()
            n_interface_total = 2 * (self._num_blocks - 1) * nx_global
            
            row_indices = []
            col_indices = []
            current_row = 0
            
            # --- 1. Bottom Interface (if exists) ---
            # This is Global Interface (j-1). We are on the top side of it.
            if j > 0:
                interface_idx = j - 1
                # Block index for "Side 1" of interface_idx is: 2 * interface_idx + 1
                global_start_idx = (2 * interface_idx + 1) * nx_global
                
                for i in range(nx_global):
                    row_indices.append(current_row)
                    col_indices.append(global_start_idx + i)
                    current_row += 1
            
            # --- 2. Top Interface (if exists) ---
            # This is Global Interface (j). We are on the bottom side of it.
            if j < self._num_blocks - 1:
                interface_idx = j
                # Block index for "Side 0" of interface_idx is: 2 * interface_idx
                global_start_idx = (2 * interface_idx) * nx_global
                
                for i in range(nx_global):
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
                
            self.setBlock(j,Cj)


    def buildLocal(self, j: int):
        """
        Construct global interface restriction matrix Cj with 2-sided interfaces.
        
        Mapping logic:
        - If Subdomain j has a BOTTOM interface (connecting to j-1):
        It maps to Interface j-1, Side 1 (the 'Down' side belonging to j).
        - If Subdomain j has a TOP interface (connecting to j+1):
        It maps to Interface j, Side 0 (the 'Up' side belonging to j).
        """
        # Total global interface DOFs: (J-1) interfaces * 2 sides * nx points
        nx_global = self._mesh.getNx()
        ny_global = self._mesh.getNy()
        n_interface_total = 2 * (self._num_blocks - 1) * nx_global
        
        row_indices = []
        col_indices = []
        current_row = 0
        
        # --- 1. Bottom Interface (if exists) ---
        # This is Global Interface (j-1). We are on the top side of it.
        if j > 0:
            interface_idx = j - 1
            # Block index for "Side 1" of interface_idx is: 2 * interface_idx + 1
            global_start_idx = (2 * interface_idx + 1) * nx_global
            
            for i in range(nx_global):
                row_indices.append(current_row)
                col_indices.append(global_start_idx + i)
                current_row += 1
        
        # --- 2. Top Interface (if exists) ---
        # This is Global Interface (j). We are on the bottom side of it.
        if j < self._num_blocks - 1:
            interface_idx = j
            # Block index for "Side 0" of interface_idx is: 2 * interface_idx
            global_start_idx = (2 * interface_idx) * nx_global
            
            for i in range(nx_global):
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
            
        self.setBlock(j, Cj)