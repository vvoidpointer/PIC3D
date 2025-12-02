"""
Grid module for PIC3D simulation.

This module handles spatial discretization and provides the computational
grid on which fields are defined and particles move.
"""

import numpy as np


class Grid:
    """
    3D Cartesian grid for PIC simulation.
    
    Implements a Yee-lattice style grid where electric and magnetic field
    components are staggered in space for accurate FDTD field solving.
    
    Parameters
    ----------
    nx, ny, nz : int
        Number of grid cells in each direction
    x_range : tuple
        (x_min, x_max) extent of simulation domain in x
    y_range : tuple
        (y_min, y_max) extent of simulation domain in y
    z_range : tuple
        (z_min, z_max) extent of simulation domain in z
    """
    
    def __init__(self, nx, ny, nz, x_range, y_range, z_range):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        
        # Grid spacings
        self.dx = (self.x_max - self.x_min) / nx
        self.dy = (self.y_max - self.y_min) / ny
        self.dz = (self.z_max - self.z_min) / nz
        
        # Grid coordinates (cell centers)
        self.x = np.linspace(self.x_min + self.dx/2, 
                            self.x_max - self.dx/2, nx)
        self.y = np.linspace(self.y_min + self.dy/2,
                            self.y_max - self.dy/2, ny)
        self.z = np.linspace(self.z_min + self.dz/2,
                            self.z_max - self.dz/2, nz)
        
        # 3D mesh grids for convenience
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, 
                                              indexing='ij')
    
    def cell_volume(self):
        """Return the volume of a single grid cell."""
        return self.dx * self.dy * self.dz
    
    def get_cell_index(self, position):
        """
        Get the cell indices for a particle position.
        
        Parameters
        ----------
        position : ndarray
            Array of shape (3,) or (N, 3) containing particle positions
        
        Returns
        -------
        indices : ndarray
            Cell indices (ix, iy, iz) for each position
        """
        position = np.atleast_2d(position)
        
        ix = np.floor((position[:, 0] - self.x_min) / self.dx).astype(int)
        iy = np.floor((position[:, 1] - self.y_min) / self.dy).astype(int)
        iz = np.floor((position[:, 2] - self.z_min) / self.dz).astype(int)
        
        # Clamp to valid range
        ix = np.clip(ix, 0, self.nx - 1)
        iy = np.clip(iy, 0, self.ny - 1)
        iz = np.clip(iz, 0, self.nz - 1)
        
        return np.column_stack([ix, iy, iz])
    
    def get_interpolation_weights(self, position):
        """
        Calculate linear interpolation weights for field interpolation.
        
        Uses Cloud-In-Cell (CIC) / linear interpolation scheme.
        
        Parameters
        ----------
        position : ndarray
            Particle positions of shape (N, 3)
        
        Returns
        -------
        indices : ndarray
            Base cell indices
        weights : ndarray
            Interpolation weights for 8 surrounding cells
        """
        position = np.atleast_2d(position)
        n_particles = position.shape[0]
        
        # Normalized positions within grid
        xn = (position[:, 0] - self.x_min) / self.dx - 0.5
        yn = (position[:, 1] - self.y_min) / self.dy - 0.5
        zn = (position[:, 2] - self.z_min) / self.dz - 0.5
        
        # Base cell indices
        ix = np.floor(xn).astype(int)
        iy = np.floor(yn).astype(int)
        iz = np.floor(zn).astype(int)
        
        # Fractional positions within cell
        fx = xn - ix
        fy = yn - iy
        fz = zn - iz
        
        # Clamp indices
        ix = np.clip(ix, 0, self.nx - 2)
        iy = np.clip(iy, 0, self.ny - 2)
        iz = np.clip(iz, 0, self.nz - 2)
        
        # Calculate 8 weights for trilinear interpolation
        weights = np.zeros((n_particles, 8))
        weights[:, 0] = (1 - fx) * (1 - fy) * (1 - fz)  # (0,0,0)
        weights[:, 1] = fx * (1 - fy) * (1 - fz)        # (1,0,0)
        weights[:, 2] = (1 - fx) * fy * (1 - fz)        # (0,1,0)
        weights[:, 3] = fx * fy * (1 - fz)              # (1,1,0)
        weights[:, 4] = (1 - fx) * (1 - fy) * fz        # (0,0,1)
        weights[:, 5] = fx * (1 - fy) * fz              # (1,0,1)
        weights[:, 6] = (1 - fx) * fy * fz              # (0,1,1)
        weights[:, 7] = fx * fy * fz                    # (1,1,1)
        
        indices = np.column_stack([ix, iy, iz])
        
        return indices, weights
    
    def is_inside(self, position):
        """
        Check if positions are inside the simulation domain.
        
        Parameters
        ----------
        position : ndarray
            Positions to check
        
        Returns
        -------
        bool or ndarray
            True if inside domain
        """
        position = np.atleast_2d(position)
        inside = ((position[:, 0] >= self.x_min) & 
                  (position[:, 0] <= self.x_max) &
                  (position[:, 1] >= self.y_min) &
                  (position[:, 1] <= self.y_max) &
                  (position[:, 2] >= self.z_min) &
                  (position[:, 2] <= self.z_max))
        return inside
    
    def __repr__(self):
        return (f"Grid({self.nx}x{self.ny}x{self.nz}, "
                f"x=[{self.x_min:.2e}, {self.x_max:.2e}], "
                f"y=[{self.y_min:.2e}, {self.y_max:.2e}], "
                f"z=[{self.z_min:.2e}, {self.z_max:.2e}])")
