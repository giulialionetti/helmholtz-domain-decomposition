import numpy as np
from scipy.sparse import csr_matrix

from src.common.operators.base_operators import BlockDiagOperator
from src.common.helmholtz.helmholtz_param import HelmholtzParameters
from src.common.mesh import Mesh, Boundary

class COperator:
    def __init__(self, *, mesh: Mesh, **kwargs):
        self._mesh = mesh
        super().__init__(**kwargs)

    def buildLocal(self, j: int):
        raise NotImplementedError("This is an abstract class")
    
class TOperator:
    def __init__(self, *, mesh: Mesh, boundary: Boundary, B: BlockDiagOperator, 
                 params: HelmholtzParameters, **kwargs):
        self._mesh = mesh
        self._boundary = boundary
        self._B = B
        self._params = params
        super().__init__(**kwargs)

    def buildLocal(self, j: int):
        raise NotImplementedError("This is an abstract class")
    
class SFactorization[T]:
    def __init__(self, A: BlockDiagOperator, T: BlockDiagOperator, B: BlockDiagOperator, **kwargs):
        self._A = A
        self._T = T 
        self._B = B
        super().__init__(**kwargs)

    def buildLocal(self, j: int):
        raise NotImplementedError("This is an abstract class")
    
class SOperator[T]:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BOperator:
    def __init__(self, *, mesh: Mesh, boundary: Boundary, **kwargs):
        self._mesh = mesh
        self._boundary = boundary
        super().__init__(**kwargs)

    def buildLocal(self, j: int):
        raise NotImplementedError("This is an abstract class")

class AOperator:
    def __init__(self, *, mesh: Mesh, boundary: Boundary, params: HelmholtzParameters, **kwargs):
        self._mesh = mesh
        self._boundary = boundary
        self._params = params
        super().__init__(**kwargs)

    def buildLocal(self, j:int):
        raise NotImplementedError("This is an abstract class")

class PiOperator:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BVecOperator:
    def __init__(self, mesh: Mesh, params: HelmholtzParameters, **kwargs):
        self._mesh = mesh
        self._params = params
        super().__init__(**kwargs)

    def buildLocal(self, j: int):
        raise NotImplementedError("This is an abstract class")

class GVecOperator:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
