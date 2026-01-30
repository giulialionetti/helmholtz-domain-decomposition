import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
while not os.path.exists(os.path.join(project_root, 'src')):
    parent = os.path.dirname(project_root)
    if parent == project_root: 
        # Fallback: assume typical structure if loop fails
        project_root = os.path.abspath(os.path.join(current_dir, "../../"))
        break
    project_root = parent

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from src.seq.operators.base_operators import FullRowBlockDiagOperator

import numpy as np

# ============================================================
# TEST
# ============================================================

def main():
    B0 = np.array([
        [ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12]
    ])

    B1 = np.array([
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24]
    ])

    B2 = np.array([
        [25, 26, 27, 28],
        [29, 30, 31, 32],
        [33, 34, 35, 36]
    ])

    Aop = FullRowBlockDiagOperator(num_blocks=3)
    Aop.setBlock(0, B0, col_offs_from_jm1=0)
    Aop.setBlock(1, B1, col_offs_from_jm1=-2)
    Aop.setBlock(2, B2, col_offs_from_jm1=-2)

    # Matrice esplicita (9 x 8)
    A = np.array([
        [ 1,  2,  3,  4,  0,  0,  0,  0],
        [ 5,  6,  7,  8,  0,  0,  0,  0],
        [ 9, 10, 11, 12,  0,  0,  0,  0],

        [ 0,  0, 13, 14, 15, 16,  0,  0],
        [ 0,  0, 17, 18, 19, 20,  0,  0],
        [ 0,  0, 21, 22, 23, 24,  0,  0],

        [ 0,  0,  0,  0, 25, 26, 27, 28],
        [ 0,  0,  0,  0, 29, 30, 31, 32],
        [ 0,  0,  0,  0, 33, 34, 35, 36],
    ])

    print("Matrice A:")
    print(A)

    x = np.arange(1, 9)  # [1 2 3 4 5 6 7 8]

    y_op = Aop.applyGlobal(x)
    y_exp = A @ x

    print("\nA @ x (operatore):")
    print(y_op)

    print("\nA @ x (esplicito):")
    print(y_exp)

    print("\nErrore diretto:",
          np.linalg.norm(y_op - y_exp))

    y = np.ones(9)

    z_op = Aop.T.applyGlobal(y)
    z_exp = A.T @ y

    print("\nA^T @ y (operatore):")
    print(z_op)

    print("\nA^T @ y (esplicito):")
    print(z_exp)

    print("\nErrore trasposto:",
          np.linalg.norm(z_op - z_exp))


if __name__ == "__main__":
    main()
