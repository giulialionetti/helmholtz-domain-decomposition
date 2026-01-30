import numpy as np
from scipy.sparse import csr_matrix

from src.seq.operators.base_operators import FullRowBlockOperator

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

class QOperator[T](FullRowBlockOperator[T]):
    def __init__(self, num_blocks: int):
        super(QOperator, self).__init__(num_blocks)