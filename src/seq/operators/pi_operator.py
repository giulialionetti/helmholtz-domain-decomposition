import numpy as np

class PiOperator:
    def __init__(self, J: int, nx: int):
        self._J = J
        self._nx = nx

    def applyGlobal(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the exchange operator Π to vector x.
        
        For every interface k:
        - Swaps Block 2k (Side 0) with Block 2k+1 (Side 1).
        """
        Px = np.zeros_like(x)
        n_interfaces = self._J - 1
        
        for k in range(n_interfaces):
            # Indices for Side 0 (belonging to subdomain below)
            idx_side0 = slice((2 * k) * self._nx, (2 * k + 1) * self._nx)
            
            # Indices for Side 1 (belonging to subdomain above)
            idx_side1 = slice((2 * k + 1) * self._nx, (2 * k + 2) * self._nx)
            
            # Perform Swap
            Px[idx_side0] = x[idx_side1]
            Px[idx_side1] = x[idx_side0]
            
        return Px
