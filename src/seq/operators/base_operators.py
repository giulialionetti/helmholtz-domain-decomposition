import numpy as np
from src.common.base_operators import (BlockDiagOperator, TransposedBlockDiagOperator, 
                                       BlockQuasiDiagOperator, TransposedBlockQuasiDiagOperator)

class FullBlockDiagOperator[T](BlockDiagOperator[T]):
    def __init__(self, num_blocks: int):
        super(FullBlockDiagOperator, self).__init__(num_blocks)
        self._block_list = [np.zeros((0,0))] * self._num_blocks
        self._shapes = [(0, 0)] * self._num_blocks
        self._num_rows = 0
        self._num_cols = 0
        self.T = TransposedFullBlockDiagOperator(self)

    def setBlock(self, j: int, Bj: T):
        if self._shapes[j] != (0, 0):
            self._num_rows -= self._shapes[j][0]
            self._num_cols -= self._shapes[j][1]

        self._block_list[j] = Bj        # type: ignore
        self._shapes[j] = Bj.shape      # type: ignore
        self._num_rows += Bj.shape[0]   # type: ignore
        self._num_cols += Bj.shape[1]   # type: ignore

    def applyLocal(self, j: int, xj: np.ndarray) -> np.ndarray:
        return self._block_list[j] @ xj

    def applyGlobal(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self._num_cols:
            raise ValueError("Input vector has wrong size")

        res = np.zeros(self._num_rows, dtype=x.dtype)
        cumulative_rows = 0
        cumulative_cols = 0

        for j in range(self._num_blocks):
            Bj = self._block_list[j]
            mj, nj = self._shapes[j]

            res[cumulative_rows:cumulative_rows + mj] = Bj @ x[cumulative_cols:cumulative_cols + nj]

            cumulative_rows += mj
            cumulative_cols += nj

        return res


class TransposedFullBlockDiagOperator[T](TransposedBlockDiagOperator):
    def __init__(self, op: FullBlockDiagOperator[T]):
        self._op = op

    def applyLocal(self, j: int, xj: np.ndarray) -> np.ndarray:
        return self._op._block_list[j].T @ xj
    
    def applyGlobal(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self._op._num_rows:
            raise ValueError("Input vector has wrong size")

        res = np.zeros(self._op._num_rows, dtype=x.dtype)
        cumulative_rows = 0
        cumulative_cols = 0

        for j in range(self._op._num_blocks):
            Bj = self._op._block_list[j]
            rows, cols = self._op._shapes[j]

            res[cumulative_cols:cumulative_cols + rows] = Bj @ x[cumulative_rows:cumulative_rows + cols]

            cumulative_cols += rows   # swapped because doing the transpose
            cumulative_rows += cols

        return res


class FullRowBlockDiagOperator[T](BlockQuasiDiagOperator[T]):
    def __init__(self, num_blocks: int):
        super(FullRowBlockDiagOperator, self).__init__(num_blocks)
        self._offsets = [0] * num_blocks
        self._block_list = [None] * num_blocks
        self._num_rows = 0
        self._num_cols = 0
        self.T = TransposedFullRowBlockDiagOperator(self)

    def setBlock(self, j: int, Bj: T, row_offs_from_jm1: int = 0, col_offs_from_jm1: int = -1):
        # Ignoring row_offs_from_jm1... it is only for rows blocks!
        if self._offsets[j] != 0:
            self._num_rows -= self._block_list[j].shape[0]                     # type: ignore
            self._num_cols -= self._block_list[j].shape[1] + self._offsets[j]  # type: ignore

        self._offsets[j] = col_offs_from_jm1
        self._block_list[j] = Bj                                # type: ignore
        self._num_rows += Bj.shape[0]                           # type: ignore
        self._num_cols += Bj.shape[1] + col_offs_from_jm1       # type: ignore

    def applyLocal(self, j: int, xj: np.ndarray) -> np.ndarray:
        return self._block_list[j] @ xj
    
    def applyGlobal(self, x: np.ndarray) -> np.ndarray:
        res = np.zeros(0)
        
        cumulative_cols = 0
        print("\n\n")
        for j in range(self._num_blocks):
            cumulative_cols += self._offsets[j]
            pres = self._block_list[j] @ x[cumulative_cols:cumulative_cols+self._block_list[j].shape[1]]
            res = np.concatenate((res, pres))    # type: ignore
            cumulative_cols += self._block_list[j].shape[1]     # type: ignore

        return res
        

class TransposedFullRowBlockDiagOperator[T](TransposedBlockQuasiDiagOperator[T]):
    def __init__(self, op: FullRowBlockDiagOperator[T]):
        self._op = op

    def applyLocal(self, j: int, xj: np.ndarray) -> np.ndarray:
        return self._op._block_list[j].T @ xj
    
    def applyGlobal(self, x: np.ndarray) -> np.ndarray:
        res = np.zeros(0)
        
        cumulative_cols = 0
        for j in range(self._op._num_blocks):
            partial_res = self._op._block_list[j].T @ x[cumulative_cols:cumulative_cols+self._op._block_list[j].T.shape[1]] # type: ignore
            offset = self._op._offsets[j]
            if offset < 0:
                res = np.concatenate((res[0:offset], 
                                     res[offset:] + partial_res[0:-1*offset], 
                                     partial_res[-1*offset:]))
            elif offset == 0:
                res = np.concatenate((res, partial_res))
            else: # offset > 0
                p = np.concatenate((np.zeros((offset,1)),partial_res))
                res = np.concatenate((res, p))
            cumulative_cols += self._op._block_list[j].T.shape[1]     # type: ignore

        return res

        
        
