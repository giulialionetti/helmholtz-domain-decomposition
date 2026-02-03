import numpy as np

np.random.seed(42)

class HelmholtzParameters:
    def __init__(self, Lx = 1.0, Ly = 2.0, kappa = 16.0, ns = 8, 
                 sp_coeff1 = 3, sp_coeff2 = 50.0, sp = None, nx = 10, ny = 10):
        self.Lx = Lx
        self.Ly = Ly
        self.kappa = kappa
        self.ns = ns
        self.nx = nx
        self.ny = ny
        if sp == None:
            self.sp = [np.random.rand(sp_coeff1) * [self.Lx, self.Ly, sp_coeff2] for _ in range(self.ns)]
        else:
            self.sp = sp
