import numpy as np

class BlockOperator[T]:
    def __init__(self, num_blocks: int):
        self._num_blocks = num_blocks

    def applyLocal(self, j: int, xj: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This is an abstract method")
    
    def applyGlobal(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method SHOULD NOT be implemented like this, " \
                                  "rather having a class Vector who is inherited by" \
                                  "SparseVector and FullVector or something like that")

class BlockDiagOperator[T](BlockOperator[T]):
    def __init__(self, num_blocks: int):
        super(BlockDiagOperator, self).__init__(num_blocks)

    def setBlock(self, j: int, Bj: T):
        raise NotImplementedError("This is an abstract method")

    def getNumBlocks(self) -> int:
        return self._num_blocks

    
class TransposedBlockDiagOperator: # should it inherit also from BlockDiagoOperator ==> which consequences there would be?
    def applyLocal(self, j: int, xj: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This is an abstract method")
        
    def applyGlobal(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method SHOULD NOT be implemented like this, " \
                                  "rather having a class Vector who is inherited by" \
                                  "SparseVector and FullVector or something like that")

class BlockQuasiDiagOperator[T](BlockOperator[T]):
    def __init__(self, num_blocks: int):
        self._num_blocks = num_blocks

    def setBlock(self, j: int, Bj: T, row_offs_from_jm1: int, col_offs_from_jm1: int):
        raise NotImplementedError("This is an abstract method")
    
class TransposedBlockQuasiDiagOperator[T]:
    def applyLocal(self, j: int, xj: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This is an abstract method")
        
    def applyGlobal(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method SHOULD NOT be implemented like this, " \
                                  "rather having a class Vector who is inherited by" \
                                  "SparseVector and FullVector or something like that")

class RowBlockOperator[T](BlockOperator[T]):
    def __init__(self, num_blocks: int):
        self._num_blocks = num_blocks

    def setBlock(self, j: int, Bj: T):
        raise NotImplementedError("This is an abstract method")
    
class TransposedRowBlockOperator[T]:
    def applyLocal(self, j: int, xj: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This is an abstract method")
        
    def applyGlobal(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method SHOULD NOT be implemented like this, " \
                                  "rather having a class Vector who is inherited by" \
                                  "SparseVector and FullVector or something like that")


class BlockVector:
    def __init__(self, rows: int):
        self._rows = rows

    def getBlock(self, j: int) -> np.ndarray:
        return np.zeros(0)
    
    def getFull(self) -> np.ndarray:
        return np.zeros(0)