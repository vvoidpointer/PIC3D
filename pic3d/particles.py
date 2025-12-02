"""
Particle species module for PIC3D simulation.

This module handles particle data storage, initialization, and the Boris
pusher algorithm for advancing particles in electromagnetic fields.
"""

import numpy as np
from . import constants as const


class ParticleSpecies:
    """
    A species of particles in the PIC simulation.
    
    Parameters
    ----------
    name : str
        Name of the species (e.g., 'electrons', 'ions')
    charge : float
        Charge per particle (in units of e)
    mass : float
        Mass per particle (in units of electron mass)
    num_particles : int
        Number of macro-particles
    weight : float, optional
        Number of real particles per macro-particle
    """
    
    def __init__(self, name, charge, mass, num_particles, weight=1.0):
        self.name = name
        self.charge = charge  # In units of e
        self.mass = mass      # In units of m_e
        self.num_particles = num_particles
        self.weight = weight  # Macro-particle weight
        
        # Particle data arrays
        self.positions = np.zeros((num_particles, 3))
        self.velocities = np.zeros((num_particles, 3))
        self.gamma = np.ones(num_particles)  # Lorentz factor
        
        # Active particle mask
        self.active = np.ones(num_particles, dtype=bool)
    
    @classmethod
    def create_electrons(cls, num_particles, weight=1.0):
        """Create an electron species."""
        return cls('electrons', charge=-1.0, mass=1.0, 
                   num_particles=num_particles, weight=weight)
    
    @classmethod
    def create_protons(cls, num_particles, weight=1.0):
        """Create a proton species."""
        return cls('protons', charge=1.0, mass=const.PROTON_MASS/const.E_MASS,
                   num_particles=num_particles, weight=weight)
    
    @classmethod
    def create_ions(cls, num_particles, charge_state, mass_amu, weight=1.0):
        """
        Create an ion species.
        
        Parameters
        ----------
        num_particles : int
            Number of macro-particles
        charge_state : int
            Ion charge state (number of electrons removed)
        mass_amu : float
            Ion mass in atomic mass units
        weight : float, optional
            Macro-particle weight
        """
        mass_kg = mass_amu * 1.66054e-27
        mass_normalized = mass_kg / const.E_MASS
        return cls(f'ions_Z{charge_state}', charge=float(charge_state),
                   mass=mass_normalized, num_particles=num_particles, 
                   weight=weight)
    
    def initialize_uniform(self, grid, density):
        """
        Initialize particles uniformly in the simulation domain.
        
        Parameters
        ----------
        grid : Grid
            The simulation grid
        density : float or callable
            Number density (m^-3) or function density(x, y, z)
        """
        # Random uniform positions
        self.positions[:, 0] = np.random.uniform(
            grid.x_min, grid.x_max, self.num_particles)
        self.positions[:, 1] = np.random.uniform(
            grid.y_min, grid.y_max, self.num_particles)
        self.positions[:, 2] = np.random.uniform(
            grid.z_min, grid.z_max, self.num_particles)
        
        # Calculate weight based on density
        if callable(density):
            # Use average density for weight calculation
            avg_density = np.mean([
                density(x, y, z) 
                for x, y, z in zip(
                    self.positions[:, 0],
                    self.positions[:, 1], 
                    self.positions[:, 2]
                )
            ])
        else:
            avg_density = density
        
        volume = ((grid.x_max - grid.x_min) * 
                  (grid.y_max - grid.y_min) * 
                  (grid.z_max - grid.z_min))
        
        total_particles = avg_density * volume
        self.weight = total_particles / self.num_particles
    
    def initialize_maxwellian(self, temperature_ev):
        """
        Initialize particle velocities with a Maxwellian distribution.
        
        Parameters
        ----------
        temperature_ev : float
            Temperature in electron volts
        """
        # Thermal velocity: v_th = sqrt(kT/m)
        # In normalized units: v_th = sqrt(T[eV] * e / (m * m_e * c^2))
        T_joules = temperature_ev * const.E_CHARGE
        m_kg = self.mass * const.E_MASS
        v_thermal = np.sqrt(T_joules / m_kg)
        
        # Sample from Maxwellian distribution
        self.velocities[:, 0] = np.random.normal(0, v_thermal, self.num_particles)
        self.velocities[:, 1] = np.random.normal(0, v_thermal, self.num_particles)
        self.velocities[:, 2] = np.random.normal(0, v_thermal, self.num_particles)
        
        # Update Lorentz factor
        self._update_gamma()
    
    def _update_gamma(self):
        """Update relativistic Lorentz factor for all particles."""
        v_squared = np.sum(self.velocities**2, axis=1)
        c_squared = const.C**2
        self.gamma = 1.0 / np.sqrt(1.0 - np.minimum(v_squared / c_squared, 0.9999))
    
    def push_boris(self, E, B, dt):
        """
        Advance particle velocities and positions using the Boris pusher.
        
        The Boris algorithm is time-centered and preserves phase space
        volume, making it ideal for PIC simulations.
        
        Parameters
        ----------
        E : ndarray
            Electric field at particle positions (N, 3) in V/m
        B : ndarray
            Magnetic field at particle positions (N, 3) in T
        dt : float
            Time step in seconds
        """
        # Only push active particles
        mask = self.active
        n_active = np.sum(mask)
        if n_active == 0:
            return
        
        # Get charge-to-mass ratio in SI units
        q = self.charge * const.E_CHARGE
        m = self.mass * const.E_MASS
        qm = q / m
        
        # Get fields for active particles
        E_active = E[mask]
        B_active = B[mask]
        v_active = self.velocities[mask].copy()
        
        # Half acceleration from E-field
        v_minus = v_active + qm * E_active * dt / 2
        
        # Rotation from B-field (Boris rotation)
        gamma_minus = 1.0 / np.sqrt(1.0 - np.sum(v_minus**2, axis=1, keepdims=True) / const.C**2)
        gamma_minus = np.clip(gamma_minus, 1.0, 1e6)
        
        t = qm * B_active * dt / (2 * gamma_minus)
        s = 2 * t / (1 + np.sum(t**2, axis=1, keepdims=True))
        
        v_prime = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + np.cross(v_prime, s)
        
        # Second half acceleration from E-field
        v_new = v_plus + qm * E_active * dt / 2
        
        # Update velocities
        self.velocities[mask] = v_new
        
        # Update positions (using new velocities)
        self.positions[mask] += v_new * dt
        
        # Update Lorentz factors
        self._update_gamma()
    
    def apply_boundary_conditions(self, grid, bc_type='periodic'):
        """
        Apply boundary conditions to particles.
        
        Parameters
        ----------
        grid : Grid
            The simulation grid
        bc_type : str
            Type of boundary condition: 'periodic', 'reflecting', 'absorbing'
        """
        if bc_type == 'periodic':
            # Wrap particles around
            self.positions[:, 0] = ((self.positions[:, 0] - grid.x_min) % 
                                    (grid.x_max - grid.x_min) + grid.x_min)
            self.positions[:, 1] = ((self.positions[:, 1] - grid.y_min) % 
                                    (grid.y_max - grid.y_min) + grid.y_min)
            self.positions[:, 2] = ((self.positions[:, 2] - grid.z_min) % 
                                    (grid.z_max - grid.z_min) + grid.z_min)
        
        elif bc_type == 'reflecting':
            # Reflect particles at boundaries
            for dim, (min_val, max_val) in enumerate([
                (grid.x_min, grid.x_max),
                (grid.y_min, grid.y_max),
                (grid.z_min, grid.z_max)
            ]):
                below = self.positions[:, dim] < min_val
                above = self.positions[:, dim] > max_val
                
                self.positions[below, dim] = 2 * min_val - self.positions[below, dim]
                self.positions[above, dim] = 2 * max_val - self.positions[above, dim]
                
                self.velocities[below | above, dim] *= -1
        
        elif bc_type == 'absorbing':
            # Remove particles that leave the domain
            inside = grid.is_inside(self.positions)
            self.active &= inside
    
    def get_charge_density(self, grid):
        """
        Deposit particle charge onto grid using CIC interpolation.
        
        Parameters
        ----------
        grid : Grid
            The simulation grid
        
        Returns
        -------
        rho : ndarray
            Charge density on grid (C/m^3)
        """
        rho = np.zeros((grid.nx, grid.ny, grid.nz))
        
        # Get active particles
        mask = self.active
        positions = self.positions[mask]
        
        if len(positions) == 0:
            return rho
        
        # Get interpolation weights
        indices, weights = grid.get_interpolation_weights(positions)
        
        # Charge per macro-particle
        q = self.charge * const.E_CHARGE * self.weight
        
        # Deposit charge to 8 surrounding cells
        for i in range(len(positions)):
            ix, iy, iz = indices[i]
            
            # Handle boundary cases
            ix1 = min(ix + 1, grid.nx - 1)
            iy1 = min(iy + 1, grid.ny - 1)
            iz1 = min(iz + 1, grid.nz - 1)
            
            rho[ix, iy, iz] += weights[i, 0] * q
            rho[ix1, iy, iz] += weights[i, 1] * q
            rho[ix, iy1, iz] += weights[i, 2] * q
            rho[ix1, iy1, iz] += weights[i, 3] * q
            rho[ix, iy, iz1] += weights[i, 4] * q
            rho[ix1, iy, iz1] += weights[i, 5] * q
            rho[ix, iy1, iz1] += weights[i, 6] * q
            rho[ix1, iy1, iz1] += weights[i, 7] * q
        
        # Normalize by cell volume
        rho /= grid.cell_volume()
        
        return rho
    
    def get_current_density(self, grid):
        """
        Deposit particle current onto grid using CIC interpolation.
        
        Parameters
        ----------
        grid : Grid
            The simulation grid
        
        Returns
        -------
        J : ndarray
            Current density on grid (A/m^2), shape (nx, ny, nz, 3)
        """
        J = np.zeros((grid.nx, grid.ny, grid.nz, 3))
        
        # Get active particles
        mask = self.active
        positions = self.positions[mask]
        velocities = self.velocities[mask]
        
        if len(positions) == 0:
            return J
        
        # Get interpolation weights
        indices, weights = grid.get_interpolation_weights(positions)
        
        # Current contribution per macro-particle
        q = self.charge * const.E_CHARGE * self.weight
        
        # Deposit current to 8 surrounding cells
        for i in range(len(positions)):
            ix, iy, iz = indices[i]
            v = velocities[i]
            
            # Handle boundary cases
            ix1 = min(ix + 1, grid.nx - 1)
            iy1 = min(iy + 1, grid.ny - 1)
            iz1 = min(iz + 1, grid.nz - 1)
            
            j_particle = q * v
            
            J[ix, iy, iz] += weights[i, 0] * j_particle
            J[ix1, iy, iz] += weights[i, 1] * j_particle
            J[ix, iy1, iz] += weights[i, 2] * j_particle
            J[ix1, iy1, iz] += weights[i, 3] * j_particle
            J[ix, iy, iz1] += weights[i, 4] * j_particle
            J[ix1, iy, iz1] += weights[i, 5] * j_particle
            J[ix, iy1, iz1] += weights[i, 6] * j_particle
            J[ix1, iy1, iz1] += weights[i, 7] * j_particle
        
        # Normalize by cell volume
        J /= grid.cell_volume()
        
        return J
    
    def kinetic_energy(self):
        """
        Calculate total kinetic energy of the species.
        
        Returns
        -------
        float
            Total kinetic energy in Joules
        """
        mask = self.active
        gamma = self.gamma[mask]
        m = self.mass * const.E_MASS
        
        # Relativistic kinetic energy: E_k = (gamma - 1) * m * c^2
        E_k_per_particle = (gamma - 1) * m * const.C**2
        
        return np.sum(E_k_per_particle) * self.weight
    
    def __repr__(self):
        n_active = np.sum(self.active)
        return (f"ParticleSpecies('{self.name}', q={self.charge}, m={self.mass:.3f}, "
                f"N={n_active}/{self.num_particles}, weight={self.weight:.2e})")
