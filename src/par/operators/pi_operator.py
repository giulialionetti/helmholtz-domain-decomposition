import numpy as np
from mpi4py import MPI

class SparsePiOperator:
    """
    Exchange operator Π for MPI-distributed setting.
    
    Each processor handles ONE subdomain j and owns TWO interface blocks:
    - Bottom interface (connecting to j-1): Side 1 of interface (j-1)
    - Top interface (connecting to j+1): Side 0 of interface j
    
    The exchange swaps data between neighboring subdomains via MPI communication.
    """
    
    def __init__(self, comm: MPI.Comm, J: int, nx: int):
        """
        Parameters
        ----------
        comm : MPI.Comm
            MPI communicator
        J : int
            Total number of subdomains
        nx : int
            Number of points per interface
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.J = J
        self.nx = nx
        self.counter = 0
        
        assert self.size == J, f"Number of MPI processes ({self.size}) must equal subdomains ({J})"
    
    def applyGlobal(self, x_local: np.ndarray) -> np.ndarray:
        """
        Apply exchange operator Π to the local portion of the global vector.
        
        Each process holds x_local of size:
        - 0 if J=1 (no interfaces)
        - nx if rank=0 or rank=J-1 (boundary subdomains, 1 interface)
        - 2*nx for interior subdomains (2 interfaces)
        
        After exchange:
        - Bottom interface receives data from rank-1 (its top interface)
        - Top interface receives data from rank+1 (its bottom interface)
        
        Parameters
        ----------
        x_local : np.ndarray
            Local interface values owned by this process
            
        Returns
        -------
        Px_local : np.ndarray
            Exchanged values (same size as input)
        """
        Px_local = np.zeros_like(x_local)
        
        # Handle different subdomain positions
        has_top = self.rank > 0
        has_bottom = self.rank < self.J - 1
        
        if has_bottom and has_top:
            # Interior subdomain: 2 interfaces
            start_index = (2*self.rank-1)*self.nx
            x_local = x_local[start_index:start_index+self.nx*2]

            assert len(x_local) == 2 * self.nx, \
                f"Rank {self.rank}: expected 2*nx={2*self.nx}, got {len(x_local)}"
            
            bottom_send = x_local[:self.nx]
            top_send = x_local[self.nx:]
            
            # Exchange with neighbors
            bottom_recv = np.empty(self.nx, dtype=x_local.dtype)
            top_recv = np.empty(self.nx, dtype=x_local.dtype)
            
            print(f"Rank {self.rank}: {self.counter}")
            # Send bottom to rank+1, receive from rank+1's top
            self.comm.Sendrecv(bottom_send, dest=self.rank+1, sendtag=0,
                             recvbuf=bottom_recv, source=self.rank+1, recvtag=0)
            # Send top to rank-1, receive from rank-1's bottom
            self.comm.Sendrecv(top_send, dest=self.rank-1, sendtag=0,
                             recvbuf=top_recv, source=self.rank-1, recvtag=0)
            
            Px_local[start_index:start_index+self.nx] = bottom_recv
            Px_local[start_index+self.nx:start_index+2*self.nx] = top_recv
            
        elif has_bottom:
            # Last subdomain: only bottom interface
            x_local = x_local[0:self.nx]
            assert len(x_local) == self.nx, \
                f"Rank {self.rank}: expected nx={self.nx}, got {len(x_local)}"
            bottom_send = x_local
            bottom_recv = np.empty(self.nx, dtype=x_local.dtype)
            
            print(f"Rank {self.rank}: {self.counter}")
            self.comm.Sendrecv(bottom_send, dest=self.rank+1, sendtag=0,
                             recvbuf=bottom_recv, source=self.rank+1, recvtag=0)
            
            Px_local[:self.nx] = bottom_recv
            
        elif has_top:
            # First subdomain: only top interface
            x_local = x_local[-self.nx:]
            assert len(x_local) == self.nx, \
                f"Rank {self.rank}: expected nx={self.nx}, got {len(x_local)}"
            top_send = x_local
            top_recv = np.empty(self.nx, dtype=x_local.dtype)
            
            print(f"Rank {self.rank}: {self.counter}")
            self.comm.Sendrecv(top_send, dest=self.rank-1, sendtag=0,
                             recvbuf=top_recv, source=self.rank-1, recvtag=0)
            
            Px_local[-self.nx:] = top_recv
        
        self.counter += 1
        return Px_local
