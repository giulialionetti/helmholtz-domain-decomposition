import scipy.sparse.linalg as spla
import os
import sys

# Ensure imports work from project root (Path fix)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_path = current_dir
while not os.path.exists(os.path.join(project_root, 'src')):
    parent = os.path.dirname(project_root)
    if parent == project_root:
        project_root = os.path.abspath(os.path.join(current_dir, "../"))
        break
    project_root = parent

if project_root not in sys.path:
    sys.path.insert(0, project_root)


from src.common.ddm_operators import(S_operator, Pi_operator)


def solve_ddm_gmres(factorizations, Bj_list, Cj_list, Tj_list, g, nx, J):
    """Solve DDM interface problem with GMRES"""
    def S_op(x):
        return S_operator(x, factorizations, Bj_list, Tj_list, Cj_list)
    
    def Pi_op(x):
        return Pi_operator(x, nx, J)
    
    def matvec(x):
        return x + Pi_op(S_op(x))
    
    n_skeleton = len(g)
    A_op = spla.LinearOperator((n_skeleton, n_skeleton), matvec=matvec, dtype=complex)
    
    residuals = []
    def callback(rk):
        residuals.append(rk)
    
    x, info = spla.gmres(A_op, -g, rtol=1e-10, callback=callback, 
                         callback_type='pr_norm', maxiter=500)
    
    return x, residuals, info, S_op, Pi_op
