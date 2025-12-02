"""
Electromagnetic fields module for PIC3D simulation.

This module implements the FDTD (Finite-Difference Time-Domain) method
for solving Maxwell's equations on a Yee lattice.
"""

import numpy as np
from . import constants as const


class Fields:
    """
    Electromagnetic field solver using the FDTD method.
    
    Implements a standard Yee lattice where E and B fields are staggered
    in both space and time for second-order accuracy.
    
    Parameters
    ----------
    grid : Grid
        The simulation grid
    """
    
    def __init__(self, grid):
        self.grid = grid
        
        # Electric field components (at cell edges)
        self.Ex = np.zeros((grid.nx, grid.ny, grid.nz))
        self.Ey = np.zeros((grid.nx, grid.ny, grid.nz))
        self.Ez = np.zeros((grid.nx, grid.ny, grid.nz))
        
        # Magnetic field components (at cell faces)
        self.Bx = np.zeros((grid.nx, grid.ny, grid.nz))
        self.By = np.zeros((grid.nx, grid.ny, grid.nz))
        self.Bz = np.zeros((grid.nx, grid.ny, grid.nz))
        
        # Current density (source term)
        self.Jx = np.zeros((grid.nx, grid.ny, grid.nz))
        self.Jy = np.zeros((grid.nx, grid.ny, grid.nz))
        self.Jz = np.zeros((grid.nx, grid.ny, grid.nz))
        
        # Pre-compute FDTD coefficients
        self._compute_coefficients()
    
    def _compute_coefficients(self):
        """Compute FDTD update coefficients."""
        dx = self.grid.dx
        dy = self.grid.dy
        dz = self.grid.dz
        
        # Coefficients for curl operations
        self.c1 = const.C * const.C * const.MU_0  # = 1/epsilon_0
        self.c2 = const.C * const.C               # For vacuum
    
    def update_B(self, dt):
        """
        Update magnetic field using Faraday's law: dB/dt = -curl(E)
        
        Parameters
        ----------
        dt : float
            Time step in seconds
        """
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz
        
        # Faraday's law with periodic boundary conditions
        # Bx: dBx/dt = -(dEz/dy - dEy/dz)
        self.Bx -= dt * (
            (np.roll(self.Ez, -1, axis=1) - self.Ez) / dy -
            (np.roll(self.Ey, -1, axis=2) - self.Ey) / dz
        )
        
        # By: dBy/dt = -(dEx/dz - dEz/dx)
        self.By -= dt * (
            (np.roll(self.Ex, -1, axis=2) - self.Ex) / dz -
            (np.roll(self.Ez, -1, axis=0) - self.Ez) / dx
        )
        
        # Bz: dBz/dt = -(dEy/dx - dEx/dy)
        self.Bz -= dt * (
            (np.roll(self.Ey, -1, axis=0) - self.Ey) / dx -
            (np.roll(self.Ex, -1, axis=1) - self.Ex) / dy
        )
    
    def update_E(self, dt):
        """
        Update electric field using Ampere's law: dE/dt = c^2*curl(B) - J/eps0
        
        Parameters
        ----------
        dt : float
            Time step in seconds
        """
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz
        c2 = const.C**2
        
        # Ampere's law with current source
        # Ex: dEx/dt = c^2*(dBz/dy - dBy/dz) - Jx/eps0
        self.Ex += dt * (
            c2 * (
                (self.Bz - np.roll(self.Bz, 1, axis=1)) / dy -
                (self.By - np.roll(self.By, 1, axis=2)) / dz
            ) - self.Jx / const.EPSILON_0
        )
        
        # Ey: dEy/dt = c^2*(dBx/dz - dBz/dx) - Jy/eps0
        self.Ey += dt * (
            c2 * (
                (self.Bx - np.roll(self.Bx, 1, axis=2)) / dz -
                (self.Bz - np.roll(self.Bz, 1, axis=0)) / dx
            ) - self.Jy / const.EPSILON_0
        )
        
        # Ez: dEz/dt = c^2*(dBy/dx - dBx/dy) - Jz/eps0
        self.Ez += dt * (
            c2 * (
                (self.By - np.roll(self.By, 1, axis=0)) / dx -
                (self.Bx - np.roll(self.Bx, 1, axis=1)) / dy
            ) - self.Jz / const.EPSILON_0
        )
    
    def set_current(self, J):
        """
        Set the current density from particle deposition.
        
        Parameters
        ----------
        J : ndarray
            Current density array of shape (nx, ny, nz, 3)
        """
        self.Jx = J[:, :, :, 0]
        self.Jy = J[:, :, :, 1]
        self.Jz = J[:, :, :, 2]
    
    def add_current(self, J):
        """
        Add current density contribution from a particle species.
        
        Parameters
        ----------
        J : ndarray
            Current density array of shape (nx, ny, nz, 3)
        """
        self.Jx += J[:, :, :, 0]
        self.Jy += J[:, :, :, 1]
        self.Jz += J[:, :, :, 2]
    
    def clear_current(self):
        """Reset current density to zero."""
        self.Jx.fill(0)
        self.Jy.fill(0)
        self.Jz.fill(0)
    
    def interpolate_to_particles(self, positions):
        """
        Interpolate fields to particle positions using CIC.
        
        Parameters
        ----------
        positions : ndarray
            Particle positions of shape (N, 3)
        
        Returns
        -------
        E : ndarray
            Electric field at particle positions (N, 3)
        B : ndarray
            Magnetic field at particle positions (N, 3)
        """
        positions = np.atleast_2d(positions)
        n_particles = positions.shape[0]
        
        E = np.zeros((n_particles, 3))
        B = np.zeros((n_particles, 3))
        
        # Get interpolation weights
        indices, weights = self.grid.get_interpolation_weights(positions)
        
        for i in range(n_particles):
            ix, iy, iz = indices[i]
            w = weights[i]
            
            # Handle boundary cases
            ix1 = min(ix + 1, self.grid.nx - 1)
            iy1 = min(iy + 1, self.grid.ny - 1)
            iz1 = min(iz + 1, self.grid.nz - 1)
            
            # Interpolate E-field
            E[i, 0] = (w[0] * self.Ex[ix, iy, iz] + w[1] * self.Ex[ix1, iy, iz] +
                       w[2] * self.Ex[ix, iy1, iz] + w[3] * self.Ex[ix1, iy1, iz] +
                       w[4] * self.Ex[ix, iy, iz1] + w[5] * self.Ex[ix1, iy, iz1] +
                       w[6] * self.Ex[ix, iy1, iz1] + w[7] * self.Ex[ix1, iy1, iz1])
            
            E[i, 1] = (w[0] * self.Ey[ix, iy, iz] + w[1] * self.Ey[ix1, iy, iz] +
                       w[2] * self.Ey[ix, iy1, iz] + w[3] * self.Ey[ix1, iy1, iz] +
                       w[4] * self.Ey[ix, iy, iz1] + w[5] * self.Ey[ix1, iy, iz1] +
                       w[6] * self.Ey[ix, iy1, iz1] + w[7] * self.Ey[ix1, iy1, iz1])
            
            E[i, 2] = (w[0] * self.Ez[ix, iy, iz] + w[1] * self.Ez[ix1, iy, iz] +
                       w[2] * self.Ez[ix, iy1, iz] + w[3] * self.Ez[ix1, iy1, iz] +
                       w[4] * self.Ez[ix, iy, iz1] + w[5] * self.Ez[ix1, iy, iz1] +
                       w[6] * self.Ez[ix, iy1, iz1] + w[7] * self.Ez[ix1, iy1, iz1])
            
            # Interpolate B-field
            B[i, 0] = (w[0] * self.Bx[ix, iy, iz] + w[1] * self.Bx[ix1, iy, iz] +
                       w[2] * self.Bx[ix, iy1, iz] + w[3] * self.Bx[ix1, iy1, iz] +
                       w[4] * self.Bx[ix, iy, iz1] + w[5] * self.Bx[ix1, iy, iz1] +
                       w[6] * self.Bx[ix, iy1, iz1] + w[7] * self.Bx[ix1, iy1, iz1])
            
            B[i, 1] = (w[0] * self.By[ix, iy, iz] + w[1] * self.By[ix1, iy, iz] +
                       w[2] * self.By[ix, iy1, iz] + w[3] * self.By[ix1, iy1, iz] +
                       w[4] * self.By[ix, iy, iz1] + w[5] * self.By[ix1, iy, iz1] +
                       w[6] * self.By[ix, iy1, iz1] + w[7] * self.By[ix1, iy1, iz1])
            
            B[i, 2] = (w[0] * self.Bz[ix, iy, iz] + w[1] * self.Bz[ix1, iy, iz] +
                       w[2] * self.Bz[ix, iy1, iz] + w[3] * self.Bz[ix1, iy1, iz] +
                       w[4] * self.Bz[ix, iy, iz1] + w[5] * self.Bz[ix1, iy, iz1] +
                       w[6] * self.Bz[ix, iy1, iz1] + w[7] * self.Bz[ix1, iy1, iz1])
        
        return E, B
    
    def apply_boundary_conditions(self, bc_type='periodic'):
        """
        Apply boundary conditions to fields.
        
        Parameters
        ----------
        bc_type : str
            Type of boundary condition: 'periodic', 'absorbing', 'reflecting'
        """
        if bc_type == 'periodic':
            # Periodic BCs are handled implicitly by np.roll in the update
            pass
        elif bc_type == 'absorbing':
            # Simple absorbing boundaries (set fields to zero at edges)
            # More sophisticated PML would be used in production codes
            for field in [self.Ex, self.Ey, self.Ez, self.Bx, self.By, self.Bz]:
                field[0, :, :] = 0
                field[-1, :, :] = 0
                field[:, 0, :] = 0
                field[:, -1, :] = 0
                field[:, :, 0] = 0
                field[:, :, -1] = 0
        elif bc_type == 'reflecting':
            # Perfect conductor (tangential E = 0)
            self.Ey[0, :, :] = 0
            self.Ez[0, :, :] = 0
            self.Ey[-1, :, :] = 0
            self.Ez[-1, :, :] = 0
            self.Ex[:, 0, :] = 0
            self.Ez[:, 0, :] = 0
            self.Ex[:, -1, :] = 0
            self.Ez[:, -1, :] = 0
            self.Ex[:, :, 0] = 0
            self.Ey[:, :, 0] = 0
            self.Ex[:, :, -1] = 0
            self.Ey[:, :, -1] = 0
    
    def field_energy(self):
        """
        Calculate total electromagnetic field energy.
        
        Returns
        -------
        tuple
            (E_field_energy, B_field_energy) in Joules
        """
        dV = self.grid.cell_volume()
        
        E_squared = self.Ex**2 + self.Ey**2 + self.Ez**2
        B_squared = self.Bx**2 + self.By**2 + self.Bz**2
        
        E_energy = 0.5 * const.EPSILON_0 * np.sum(E_squared) * dV
        B_energy = 0.5 / const.MU_0 * np.sum(B_squared) * dV
        
        return E_energy, B_energy
    
    def get_E_field(self):
        """Return E-field as single array of shape (nx, ny, nz, 3)."""
        return np.stack([self.Ex, self.Ey, self.Ez], axis=-1)
    
    def get_B_field(self):
        """Return B-field as single array of shape (nx, ny, nz, 3)."""
        return np.stack([self.Bx, self.By, self.Bz], axis=-1)
    
    def __repr__(self):
        E_max = max(np.max(np.abs(self.Ex)), np.max(np.abs(self.Ey)), 
                    np.max(np.abs(self.Ez)))
        B_max = max(np.max(np.abs(self.Bx)), np.max(np.abs(self.By)),
                    np.max(np.abs(self.Bz)))
        return f"Fields(max|E|={E_max:.2e} V/m, max|B|={B_max:.2e} T)"
