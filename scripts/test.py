#!/usr/bin/env python3
"""
Test to find the correct reconstruction factor
"""

import numpy as np
import scipy.sparse.linalg as spla
import sys
import os

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

from src.seq.helmholtz_solver import HelmholtzSolver
from src.common.helmholtz.helmholtz_param import HelmholtzParameters
from src.seq.operators.u_operator import uj_solution
from src.common.mesh import mesh as mesh_fnc
from src.common.mesh import boundary as boundary_fnc
from src.common.helmholtz.system_assembly import mass, stiffness, point_source

def merge_and_compare(mesh_ddm, uj_list, vtx_full, u_full):
    """Merge DDM solution and compute error vs full solution."""
    # Simple approach: sort by coordinates
    all_vtx = []
    all_u = []
    for j in range(len(uj_list)):
        vtxj, _ = mesh_ddm.getLocal(j)
        all_vtx.append(vtxj)
        all_u.append(uj_list[j])
    
    raw_vtx = np.vstack(all_vtx)
    raw_u = np.concatenate(all_u)
    
    # Sort both by coordinates
    tol = 1e-10
    scale = 1.0 / tol
    
    keys_ddm = np.round(raw_vtx[:, 0], 10) * scale + np.round(raw_vtx[:, 1], 10) * scale * 1e10
    keys_full = np.round(vtx_full[:, 0], 10) * scale + np.round(vtx_full[:, 1], 10) * scale * 1e10
    
    sort_ddm = np.argsort(keys_ddm)
    sort_full = np.argsort(keys_full)
    
    # Find unique DDM vertices (average duplicates)
    sorted_keys_ddm = keys_ddm[sort_ddm]
    sorted_u_ddm = raw_u[sort_ddm]
    
    unique_mask = np.concatenate([[True], np.abs(np.diff(sorted_keys_ddm)) > 0.5])
    unique_indices = np.cumsum(unique_mask) - 1
    
    n_unique = unique_mask.sum()
    merged_u = np.zeros(n_unique, dtype=complex)
    counts = np.zeros(n_unique)
    
    for i, idx in enumerate(unique_indices):
        merged_u[idx] += sorted_u_ddm[i]
        counts[idx] += 1
    merged_u /= counts
    
    # Compare
    u_full_sorted = u_full[sort_full]
    
    if len(merged_u) == len(u_full_sorted):
        error = np.linalg.norm(merged_u - u_full_sorted) / np.linalg.norm(u_full_sorted)
        return error
    else:
        return float('inf')

print("="*70)
print("RECONSTRUCTION FACTOR TEST")
print("="*70)

# Setup
nx, ny, J = 33, 65, 4
params = HelmholtzParameters()
params.nx = nx
params.ny = ny

# Build DDM solver
solver = HelmholtzSolver(params, J)
solver.assembly()
components = solver.getComponents()
s_fact, B, Q, T, BVec, mesh_ddm, g_vec, S_op, Pi_op = components

# Solve DDM
x_ddm, _, _, _, _ = solver.solve()

# Solve full system
vtx, elt = mesh_fnc(nx, ny, params.Lx, params.Ly)
boundary_dict = boundary_fnc(nx, ny)
belt = np.vstack(list(boundary_dict.values()))
M = mass(vtx, elt)
Mb = mass(vtx, belt)
K = stiffness(vtx, elt)
A_full = K - params.kappa**2 * M - 1j * params.kappa * Mb
b_full = M @ point_source(params.sp, params.kappa)(vtx)
x_full, _ = spla.gmres(A_full, b_full, rtol=1e-10, maxiter=5000)

print(f"\nReference: ||u_full|| = {np.linalg.norm(x_full):.4e}")

# Custom reconstruction function with parameters
def reconstruct_custom(x_skeleton, s_fact, B, Q, T, BVec, Pi_op, J, 
                       use_pi=False, factor=1.0):
    """
    Reconstruct with:
    u_j = S_j^{-1} (b_j + factor * B_j^T T_j x_j)
    
    where x can be transformed by Pi first.
    """
    x_use = Pi_op.applyGlobal(x_skeleton) if use_pi else x_skeleton
    
    uj_list = []
    for j in range(J):
        xj = Q.applyLocal(j, x_use)
        bj = BVec.getBlock(j)
        Bj = B.getBlock(j)
        Tj = T.getBlock(j)
        
        rhs = bj + factor * (Bj.T @ (Tj @ xj))
        uj = s_fact.applyLocal(j, rhs)
        uj_list.append(uj)
    
    return uj_list

print("\n" + "-"*70)
print("Testing reconstruction formulas:")
print("-"*70)

# Test various combinations
test_cases = [
    # (use_pi, factor, description)
    (False, 1.0, "Standard: b + B'Tx"),
    (True, 1.0, "Pi first: b + B'T(Pi x)"),
    (False, -1.0, "Negated: b - B'Tx"),
    (True, -1.0, "Pi + negated: b - B'T(Pi x)"),
    (False, 0.5, "Half: b + 0.5*B'Tx"),
    (True, 0.5, "Pi + half: b + 0.5*B'T(Pi x)"),
    (False, 2.0, "Double: b + 2*B'Tx"),
    (True, 2.0, "Pi + double: b + 2*B'T(Pi x)"),
    (False, 1j, "Imag: b + i*B'Tx"),
    (True, 1j, "Pi + imag: b + i*B'T(Pi x)"),
    (False, -1j, "Neg imag: b - i*B'Tx"),
    (True, -1j, "Pi + neg imag: b - i*B'T(Pi x)"),
    (False, 2j, "2i: b + 2i*B'Tx"),
    (True, 2j, "Pi + 2i: b + 2i*B'T(Pi x)"),
    (False, -2j, "-2i: b - 2i*B'Tx"),
    (True, -2j, "Pi + -2i: b - 2i*B'T(Pi x)"),
    (False, 0.5j, "0.5i: b + 0.5i*B'Tx"),
    (True, 0.5j, "Pi + 0.5i: b + 0.5i*B'T(Pi x)"),
    (False, -0.5j, "-0.5i: b - 0.5i*B'Tx"),
    (True, -0.5j, "Pi + -0.5i: b - 0.5i*B'T(Pi x)"),
]

results = []
for use_pi, factor, desc in test_cases:
    uj_list = reconstruct_custom(x_ddm, s_fact, B, Q, T, BVec, Pi_op, J,
                                  use_pi=use_pi, factor=factor)
    error = merge_and_compare(mesh_ddm, uj_list, vtx, x_full)
    results.append((error, desc, use_pi, factor))
    print(f"  {desc:30s}: error = {error:.4e}")

# Sort by error
results.sort(key=lambda x: x[0])

print("\n" + "-"*70)
print("TOP 5 BEST FORMULAS:")
print("-"*70)
for i, (error, desc, use_pi, factor) in enumerate(results[:5]):
    print(f"  {i+1}. {desc:30s}: error = {error:.4e}")

best_error, best_desc, best_use_pi, best_factor = results[0]

print("\n" + "="*70)
print(f"BEST: {best_desc}")
print(f"  use_pi = {best_use_pi}")
print(f"  factor = {best_factor}")
print(f"  error = {best_error:.4e}")
print("="*70)

# If still not good, try finer search around best factor
if best_error > 1e-6:
    print("\n" + "-"*70)
    print("Fine search around best factor...")
    print("-"*70)
    
    base_factor = best_factor
    for mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
        for phase in [1, 1j, -1, -1j]:
            factor = mult * base_factor * phase if base_factor != 0 else mult * phase
            for use_pi in [False, True]:
                uj_list = reconstruct_custom(x_ddm, s_fact, B, Q, T, BVec, Pi_op, J,
                                              use_pi=use_pi, factor=factor)
                error = merge_and_compare(mesh_ddm, uj_list, vtx, x_full)
                if error < best_error:
                    best_error = error
                    best_factor = factor
                    best_use_pi = use_pi
                    print(f"  New best: factor={factor}, use_pi={use_pi}, error={error:.4e}")

print("\n" + "="*70)
print("FINAL RESULT")
print("="*70)
print(f"Best factor: {best_factor}")
print(f"Use Pi: {best_use_pi}")
print(f"Error: {best_error:.4e}")

if best_error < 1e-6:
    print("\nSUCCESS! Found correct reconstruction formula.")
else:
    print("\nWARNING: Could not find a good reconstruction formula.")
    print("The problem might be elsewhere (operators T, B, or S definition).")