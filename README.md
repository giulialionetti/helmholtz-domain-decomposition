

## Project Overview
This project implements **Domain Decomposition (DD) methods** to solve the **Helmholtz equation** in 2D. The goal is to investigate high-performance computing strategies for numerical methods, moving from a direct sequential solver to iterative interface solvers (Fixed Point, GMRES) and finally to a parallel implementation using MPI.


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
