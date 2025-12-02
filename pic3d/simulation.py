"""
Main simulation driver for PIC3D laser-plasma interactions.

This module provides the high-level simulation class that coordinates
all components of the PIC simulation.
"""

import numpy as np
from .grid import Grid
from .particles import ParticleSpecies
from .fields import Fields
from .laser import LaserPulse
from . import constants as const


class Simulation:
    """
    PIC simulation driver for laser-plasma interactions.
    
    Manages the simulation loop, particle-field coupling, and diagnostics.
    
    Parameters
    ----------
    grid : Grid
        The simulation grid
    dt : float, optional
        Time step in seconds. If not provided, calculated from CFL condition.
    boundary_conditions : str
        Type of boundary conditions: 'periodic', 'absorbing', 'reflecting'
    """
    
    def __init__(self, grid, dt=None, boundary_conditions='periodic'):
        self.grid = grid
        self.boundary_conditions = boundary_conditions
        
        # Initialize fields
        self.fields = Fields(grid)
        
        # Particle species
        self.species = []
        
        # Laser pulses
        self.lasers = []
        
        # Time step (CFL condition: dt < dx/(c*sqrt(3)))
        if dt is None:
            dx_min = min(grid.dx, grid.dy, grid.dz)
            self.dt = 0.9 * dx_min / (const.C * np.sqrt(3))
        else:
            self.dt = dt
        
        # Simulation state
        self.time = 0.0
        self.step = 0
        
        # Diagnostics storage
        self.diagnostics = {
            'time': [],
            'field_energy_E': [],
            'field_energy_B': [],
            'kinetic_energy': [],
            'total_energy': [],
        }
    
    @classmethod
    def create_laser_plasma_simulation(cls, nx=64, ny=32, nz=32,
                                       plasma_density=1e25,  # m^-3
                                       plasma_temperature=10.0,  # eV
                                       laser_wavelength=800e-9,
                                       laser_intensity=1e18,  # W/cm^2
                                       laser_pulse_duration=30e-15,
                                       laser_spot_size=5e-6,
                                       particles_per_cell=8):
        """
        Create a pre-configured laser-plasma interaction simulation.
        
        Parameters
        ----------
        nx, ny, nz : int
            Grid dimensions
        plasma_density : float
            Electron density in m^-3
        plasma_temperature : float
            Initial plasma temperature in eV
        laser_wavelength : float
            Laser wavelength in meters
        laser_intensity : float
            Peak laser intensity in W/cm^2
        laser_pulse_duration : float
            Laser FWHM pulse duration in seconds
        laser_spot_size : float
            Laser spot size at focus in meters
        particles_per_cell : int
            Number of macro-particles per cell per species
        
        Returns
        -------
        Simulation
            Configured simulation object ready to run
        """
        # Calculate domain size based on laser wavelength
        lambda_laser = laser_wavelength
        x_size = 50 * lambda_laser  # 50 wavelengths in x
        y_size = 20 * lambda_laser  # 20 wavelengths in y
        z_size = 20 * lambda_laser  # 20 wavelengths in z
        
        # Create grid
        grid = Grid(nx, ny, nz,
                   x_range=(0, x_size),
                   y_range=(-y_size/2, y_size/2),
                   z_range=(-z_size/2, z_size/2))
        
        # Create simulation
        sim = cls(grid, boundary_conditions='absorbing')
        
        # Create laser pulse
        laser = LaserPulse(
            wavelength=laser_wavelength,
            intensity=laser_intensity,
            pulse_duration=laser_pulse_duration,
            spot_size=laser_spot_size,
            polarization='y',
            focus_position=[x_size/4, 0, 0],  # Focus at 1/4 of domain
            propagation_direction='+x'
        )
        sim.add_laser(laser)
        
        # Create plasma (starting at 1/4 of domain)
        total_particles = particles_per_cell * nx * ny * nz
        
        # Electrons
        electrons = ParticleSpecies.create_electrons(total_particles // 2)
        
        # Initialize only in plasma region (x > x_size/4)
        electrons.positions[:, 0] = np.random.uniform(
            x_size/4, x_size, electrons.num_particles)
        electrons.positions[:, 1] = np.random.uniform(
            -y_size/2, y_size/2, electrons.num_particles)
        electrons.positions[:, 2] = np.random.uniform(
            -z_size/2, z_size/2, electrons.num_particles)
        
        # Set particle weight for correct density
        plasma_volume = 0.75 * x_size * y_size * z_size
        electrons.weight = plasma_density * plasma_volume / electrons.num_particles
        
        # Initialize thermal velocities
        electrons.initialize_maxwellian(plasma_temperature)
        
        sim.add_species(electrons)
        
        # Ions (protons for simplicity)
        ions = ParticleSpecies.create_protons(total_particles // 2)
        ions.positions[:] = electrons.positions.copy()
        ions.weight = electrons.weight
        ions.initialize_maxwellian(plasma_temperature / 100)  # Cold ions
        
        sim.add_species(ions)
        
        print(f"Created laser-plasma simulation:")
        print(f"  Grid: {grid}")
        print(f"  Time step: {sim.dt:.2e} s")
        print(f"  Laser: {laser}")
        print(f"  Critical density: {laser.critical_density():.2e} m^-3")
        print(f"  Plasma density: {plasma_density:.2e} m^-3 "
              f"({plasma_density/laser.critical_density():.2f} n_c)")
        print(f"  Electrons: {electrons}")
        print(f"  Ions: {ions}")
        
        return sim
    
    def add_species(self, species):
        """Add a particle species to the simulation."""
        self.species.append(species)
        self.diagnostics[f'{species.name}_kinetic_energy'] = []
    
    def add_laser(self, laser):
        """Add a laser pulse to the simulation."""
        self.lasers.append(laser)
    
    def advance_one_step(self):
        """
        Advance the simulation by one time step.
        
        Uses a leap-frog algorithm:
        1. Inject laser fields at boundaries
        2. Push B by dt/2
        3. Push E by dt
        4. Push B by dt/2
        5. Interpolate fields to particles
        6. Push particles (Boris pusher)
        7. Deposit current
        8. Apply boundary conditions
        """
        dt = self.dt
        
        # Inject laser fields at boundaries
        for laser in self.lasers:
            laser.inject_at_boundary(self.fields, self.time, 
                                    injection_plane='x_min')
        
        # Clear current density
        self.fields.clear_current()
        
        # Deposit current from all species
        for species in self.species:
            J = species.get_current_density(self.grid)
            self.fields.add_current(J)
        
        # Update B field by dt/2
        self.fields.update_B(dt / 2)
        
        # Update E field by dt
        self.fields.update_E(dt)
        
        # Update B field by another dt/2
        self.fields.update_B(dt / 2)
        
        # Push particles
        for species in self.species:
            # Interpolate fields to particle positions
            mask = species.active
            E, B = self.fields.interpolate_to_particles(species.positions[mask])
            
            # Create full arrays for all particles (including inactive)
            E_full = np.zeros((species.num_particles, 3))
            B_full = np.zeros((species.num_particles, 3))
            E_full[mask] = E
            B_full[mask] = B
            
            # Push particles with Boris algorithm
            species.push_boris(E_full, B_full, dt)
            
            # Apply boundary conditions
            species.apply_boundary_conditions(self.grid, self.boundary_conditions)
        
        # Apply field boundary conditions
        self.fields.apply_boundary_conditions(self.boundary_conditions)
        
        # Advance time
        self.time += dt
        self.step += 1
    
    def run(self, num_steps=None, end_time=None, diagnostic_interval=10,
            progress_interval=100):
        """
        Run the simulation for a specified duration.
        
        Parameters
        ----------
        num_steps : int, optional
            Number of time steps to run
        end_time : float, optional
            Run until this simulation time is reached
        diagnostic_interval : int
            Save diagnostics every this many steps
        progress_interval : int
            Print progress every this many steps
        """
        if num_steps is None and end_time is None:
            raise ValueError("Specify either num_steps or end_time")
        
        if num_steps is None:
            num_steps = int((end_time - self.time) / self.dt)
        
        print(f"\nStarting simulation run for {num_steps} steps...")
        print(f"Current time: {self.time:.2e} s, Step: {self.step}")
        
        for _ in range(num_steps):
            # Advance simulation
            self.advance_one_step()
            
            # Save diagnostics
            if self.step % diagnostic_interval == 0:
                self._record_diagnostics()
            
            # Print progress
            if self.step % progress_interval == 0:
                E_energy, B_energy = self.fields.field_energy()
                K_energy = sum(sp.kinetic_energy() for sp in self.species)
                print(f"Step {self.step}, t={self.time:.2e} s, "
                      f"E_field={E_energy:.2e} J, B_field={B_energy:.2e} J, "
                      f"K_particles={K_energy:.2e} J")
        
        print(f"\nSimulation completed at t={self.time:.2e} s, step {self.step}")
    
    def _record_diagnostics(self):
        """Record diagnostic quantities."""
        self.diagnostics['time'].append(self.time)
        
        E_energy, B_energy = self.fields.field_energy()
        self.diagnostics['field_energy_E'].append(E_energy)
        self.diagnostics['field_energy_B'].append(B_energy)
        
        K_total = 0.0
        for species in self.species:
            K = species.kinetic_energy()
            self.diagnostics[f'{species.name}_kinetic_energy'].append(K)
            K_total += K
        
        self.diagnostics['kinetic_energy'].append(K_total)
        self.diagnostics['total_energy'].append(E_energy + B_energy + K_total)
    
    def get_diagnostics_array(self):
        """Convert diagnostics to numpy arrays."""
        return {k: np.array(v) for k, v in self.diagnostics.items()}
    
    def get_electron_density(self):
        """
        Calculate electron density on the grid.
        
        Returns
        -------
        ndarray
            Electron density in m^-3
        """
        for species in self.species:
            if species.name == 'electrons':
                rho = species.get_charge_density(self.grid)
                return -rho / const.E_CHARGE  # Convert to number density
        return np.zeros((self.grid.nx, self.grid.ny, self.grid.nz))
    
    def get_field_slice(self, component='Ey', slice_axis='z', slice_index=None):
        """
        Get a 2D slice of a field component.
        
        Parameters
        ----------
        component : str
            Field component: 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'
        slice_axis : str
            Axis perpendicular to slice: 'x', 'y', or 'z'
        slice_index : int, optional
            Index along slice axis. Default is middle of domain.
        
        Returns
        -------
        ndarray
            2D slice of the field
        """
        field = getattr(self.fields, component)
        
        if slice_axis == 'x':
            if slice_index is None:
                slice_index = self.grid.nx // 2
            return field[slice_index, :, :]
        elif slice_axis == 'y':
            if slice_index is None:
                slice_index = self.grid.ny // 2
            return field[:, slice_index, :]
        elif slice_axis == 'z':
            if slice_index is None:
                slice_index = self.grid.nz // 2
            return field[:, :, slice_index]
    
    def save_state(self, filename):
        """
        Save simulation state to file.
        
        Parameters
        ----------
        filename : str
            Output filename (numpy .npz format)
        """
        data = {
            'time': self.time,
            'step': self.step,
            'dt': self.dt,
            'Ex': self.fields.Ex,
            'Ey': self.fields.Ey,
            'Ez': self.fields.Ez,
            'Bx': self.fields.Bx,
            'By': self.fields.By,
            'Bz': self.fields.Bz,
        }
        
        for i, species in enumerate(self.species):
            data[f'species_{i}_name'] = species.name
            data[f'species_{i}_positions'] = species.positions
            data[f'species_{i}_velocities'] = species.velocities
            data[f'species_{i}_active'] = species.active
        
        np.savez(filename, **data)
        print(f"Saved simulation state to {filename}")
    
    def __repr__(self):
        return (f"Simulation(grid={self.grid}, dt={self.dt:.2e} s, "
                f"species={len(self.species)}, lasers={len(self.lasers)}, "
                f"time={self.time:.2e} s, step={self.step})")
