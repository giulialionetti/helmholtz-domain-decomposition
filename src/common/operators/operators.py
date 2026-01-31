import numpy as np
from scipy.sparse import csr_matrix

from src.common.mesh import Mesh, Boundary

class QOperator:
    def __init__(self, *, mesh: Mesh, **kwargs):
        self._mesh = mesh
        super().__init__(**kwargs)

    def buildLocal(self, j: int, nx: int, ny: int):
        raise NotImplementedError("This is an abstract class")
    
class TOperator:
    def __init__(self, *, mesh: Mesh, boundary: Boundary, **kwargs):
        self._mesh = mesh
        self._boundary = boundary
        super().__init__(**kwargs)

    def buildLocal(self, j: int, vtxj: np.ndarray, beltj_artf: np.ndarray, 
                   Bj: csr_matrix, kappa: float):
        raise NotImplementedError("This is an abstract class")
    
class SFactorization[T]:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def buildLocal(self, j: int, Aj: T, Tj: T, Bj: T):
        raise NotImplementedError("This is an abstract class")
    
class SOperator[T]:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BOperator:
    def __init__(self, *, mesh: Mesh, boundary: Boundary, **kwargs):
        self._mesh = mesh
        self._boundary = boundary
        super().__init__(**kwargs)

    def buildLocal(self, j: int, nx: int, ny: int, beltj_artf: np.ndarray):
        raise NotImplementedError("This is an abstract class")

class AOperator:
    def __init__(self, *, mesh: Mesh, boundary: Boundary, **kwargs):
        self._mesh = mesh
        self._boundary = boundary
        super().__init__(**kwargs)

    def buildLocal(self, j:int, vtxj: np.ndarray, eltj: np.ndarray, 
                   beltj_phys: np.ndarray, kappa: float):
        raise NotImplementedError("This is an abstract class")

class PiOperator:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BVecOperator:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def buildLocal(self, j: int, vtxj: np.ndarray, eltj: np.ndarray, sp: list, kappa: float):
        raise NotImplementedError("This is an abstract class")

class GVecOperator:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
