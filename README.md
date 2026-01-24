

## Project Overview
This project implements **Domain Decomposition (DD) methods** to solve the **Helmholtz equation** in 2D. The goal is to investigate high-performance computing strategies for numerical methods, moving from a direct sequential solver to iterative interface solvers (Fixed Point, GMRES) and finally to a parallel implementation using MPI.

The project focuses on handling the indefinite nature of the Helmholtz operator for high wave numbers, where standard direct solvers scale poorly on distributed architectures.

## Mathematical Model

### 1. The Helmholtz Problem
We solve the Helmholtz equation on a rectangular domain $\Omega = (0, L_x) \times (0, L_y)$:

$$
-\Delta u - k^2 u = f \quad \text{in } \Omega
$$

with a boundary condition $g=0$ serving as a crude model for an outgoing radiation condition.

### 2. Source Term
The source term $f$ consists of $N_s$ regularized Gaussian point sources:

$$
f(\mathbf{x}) = \sum_{i=1}^{N_s} w_i e^{-\frac{10}{\lambda^2} |\mathbf{x} - \mathbf{s}_i|^2}
$$

where $\mathbf{s}_i$ are source locations, $w_i$ are weights, and $\lambda$ is the wavelength.

### 3. Discretization
* **Mesh:** Uniform triangular mesh using $\mathbb{P}_1$ scalar Lagrange finite elements.
* **Resolution Condition:** To resolve wave oscillations and minimize pollution errors, the mesh size $h$ must satisfy $h < \lambda/10$ (at least 10 points per wavelength).


## Algorithms & Implementation Details

### Domain Decomposition Strategy
The domain $\Omega$ is decomposed into $J$ non-overlapping sub-domains. The decomposition is performed along the $y$-direction only, creating $J$ horizontal slabs of size $(0, L_x) \times (0, L_y/J)$.

### Matrix Definitions
The implementation relies on constructing local operators for each sub-domain $j$:
* **Restriction Matrices:**
    * $R_j$: Restrictions to local degrees of freedom.
    * $B_j$: Restriction to the physical boundary $\partial \Omega_j \cap \partial \Omega$.
    * $C_j$: Boolean matrix selecting the local part of the interface vector.
* **Local Problem Matrices:**
    * $A_j$: The local stiffness/mass matrix for the Helmholtz operator.
    * $T_j$: Local transmission matrix associated with interfaces $\Sigma_j$.
* **Global Operators:**
    * $S$: A block-diagonal operator requiring the factorization of local problems.
    * $\Pi$: An exchange operator that swaps coefficients between neighboring sub-domains.

### Solvers
The interface problem $(\mathbf{I} + \mathbf{\Pi}\mathbf{S})$ is solved using:
1.  **Fixed Point Method:** With relaxation parameter $\omega \in (0, 1)$.
2.  **GMRES:** Using `scipy.sparse.linalg.gmres` for better convergence properties on indefinite systems.


## Installation

### 1. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source ./venv/bin/activate

```

### 2. Install Dependencies

Install the required packages (NumPy, SciPy, Matplotlib) using `requirements.txt`.

```bash
pip install -r requirements.txt

```

## Usage

This project is structured as a local package. To ensure imports work correctly, **always run scripts from the project root directory** using the `-m` flag.

### Running Verification Tests

To run the convergence checks or unit tests:

```bash
# Run the convergence test
python3 -m src.GMRES

# Run the fixed point solver
python3 -m src.tests.test_fixed_point_solver
