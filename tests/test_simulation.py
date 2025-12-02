"""
Tests for the Simulation module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pic3d.simulation import Simulation
from pic3d.grid import Grid
from pic3d.particles import ParticleSpecies
from pic3d.laser import LaserPulse
from pic3d import constants as const


class TestSimulation:
    """Test cases for the Simulation class."""
    
    def test_simulation_creation(self):
        """Test basic simulation creation."""
        grid = Grid(10, 10, 10, (0, 1e-5), (0, 1e-5), (0, 1e-5))
        sim = Simulation(grid)
        
        assert sim.grid is grid
        assert sim.time == 0.0
        assert sim.step == 0
        assert len(sim.species) == 0
        assert len(sim.lasers) == 0
    
    def test_cfl_timestep(self):
        """Test CFL-limited time step calculation."""
        grid = Grid(10, 10, 10, (0, 1e-6), (0, 1e-6), (0, 1e-6))
        sim = Simulation(grid)
        
        dx = grid.dx
        max_dt = dx / (const.C * np.sqrt(3))
        
        assert sim.dt < max_dt
    
    def test_custom_timestep(self):
        """Test custom time step."""
        grid = Grid(10, 10, 10, (0, 1e-6), (0, 1e-6), (0, 1e-6))
        dt = 1e-16
        sim = Simulation(grid, dt=dt)
        
        assert sim.dt == dt
    
    def test_add_species(self):
        """Test adding particle species."""
        grid = Grid(10, 10, 10, (0, 1e-5), (0, 1e-5), (0, 1e-5))
        sim = Simulation(grid)
        
        electrons = ParticleSpecies.create_electrons(100)
        sim.add_species(electrons)
        
        assert len(sim.species) == 1
        assert sim.species[0] is electrons
    
    def test_add_laser(self):
        """Test adding laser pulse."""
        grid = Grid(10, 10, 10, (0, 1e-5), (0, 1e-5), (0, 1e-5))
        sim = Simulation(grid)
        
        laser = LaserPulse(
            wavelength=800e-9,
            intensity=1e18,
            pulse_duration=30e-15,
            spot_size=5e-6
        )
        sim.add_laser(laser)
        
        assert len(sim.lasers) == 1
        assert sim.lasers[0] is laser
    
    def test_advance_one_step(self):
        """Test single time step advance."""
        grid = Grid(10, 10, 10, (0, 1e-5), (0, 1e-5), (0, 1e-5))
        sim = Simulation(grid)
        
        initial_time = sim.time
        initial_step = sim.step
        
        sim.advance_one_step()
        
        assert sim.time == initial_time + sim.dt
        assert sim.step == initial_step + 1
    
    def test_run_simulation(self):
        """Test running simulation for multiple steps."""
        grid = Grid(10, 10, 10, (0, 1e-5), (0, 1e-5), (0, 1e-5))
        sim = Simulation(grid)
        
        num_steps = 10
        sim.run(num_steps=num_steps, diagnostic_interval=5, progress_interval=100)
        
        assert sim.step == num_steps
        assert sim.time == pytest.approx(num_steps * sim.dt)
    
    def test_diagnostics_recording(self):
        """Test that diagnostics are recorded."""
        grid = Grid(10, 10, 10, (0, 1e-5), (0, 1e-5), (0, 1e-5))
        sim = Simulation(grid)
        
        sim.run(num_steps=20, diagnostic_interval=5, progress_interval=100)
        
        diag = sim.get_diagnostics_array()
        
        assert len(diag['time']) > 0
        assert len(diag['field_energy_E']) > 0
        assert len(diag['field_energy_B']) > 0
        assert len(diag['kinetic_energy']) > 0
    
    def test_laser_plasma_factory(self):
        """Test laser-plasma simulation factory method."""
        sim = Simulation.create_laser_plasma_simulation(
            nx=16, ny=8, nz=8,
            particles_per_cell=2
        )
        
        assert len(sim.species) == 2  # electrons and ions
        assert len(sim.lasers) == 1
        assert sim.grid.nx == 16
    
    def test_field_slice(self):
        """Test getting field slices."""
        grid = Grid(10, 10, 10, (0, 1e-5), (0, 1e-5), (0, 1e-5))
        sim = Simulation(grid)
        
        # Set some field
        sim.fields.Ey[:] = 1e6
        
        slice_z = sim.get_field_slice('Ey', slice_axis='z')
        
        assert slice_z.shape == (10, 10)
        assert np.allclose(slice_z, 1e6)
    
    def test_electron_density(self):
        """Test electron density calculation."""
        grid = Grid(10, 10, 10, (0, 1e-5), (0, 1e-5), (0, 1e-5))
        sim = Simulation(grid)
        
        electrons = ParticleSpecies.create_electrons(100)
        electrons.positions[:] = np.random.uniform(0, 1e-5, (100, 3))
        electrons.weight = 1e10
        sim.add_species(electrons)
        
        n_e = sim.get_electron_density()
        
        assert n_e.shape == (10, 10, 10)
        # Should have some non-zero density
        assert np.sum(n_e) > 0
    
    def test_with_particles(self):
        """Test simulation with particles."""
        grid = Grid(10, 10, 10, (0, 1e-5), (0, 1e-5), (0, 1e-5))
        sim = Simulation(grid)
        
        # Add electrons
        electrons = ParticleSpecies.create_electrons(100)
        electrons.positions[:] = np.random.uniform(1e-6, 9e-6, (100, 3))
        electrons.weight = 1e10
        sim.add_species(electrons)
        
        # Run a few steps
        initial_positions = electrons.positions.copy()
        sim.run(num_steps=5, diagnostic_interval=5, progress_interval=100)
        
        # Positions should have changed (due to thermal motion)
        # or stayed the same if no initial velocity
        # This just checks the simulation runs without error
        assert sim.step == 5
    
    def test_repr(self):
        """Test string representation."""
        grid = Grid(10, 10, 10, (0, 1e-5), (0, 1e-5), (0, 1e-5))
        sim = Simulation(grid)
        
        repr_str = repr(sim)
        
        assert "Simulation" in repr_str
        assert "dt=" in repr_str


class TestLaserPlasmaPhysics:
    """Integration tests for laser-plasma physics."""
    
    def test_laser_injection(self):
        """Test that laser is properly injected into simulation."""
        sim = Simulation.create_laser_plasma_simulation(
            nx=32, ny=8, nz=8,
            particles_per_cell=1
        )
        
        # Run a few steps to inject laser
        sim.run(num_steps=10, diagnostic_interval=10, progress_interval=100)
        
        # Check that there are non-zero fields
        E_energy, B_energy = sim.fields.field_energy()
        assert E_energy > 0 or B_energy > 0
    
    def test_particle_field_interaction(self):
        """Test that particles respond to fields."""
        grid = Grid(10, 10, 10, (0, 1e-5), (-5e-6, 5e-6), (-5e-6, 5e-6))
        sim = Simulation(grid)
        
        # Add uniform E-field
        sim.fields.Ex[:] = 1e9  # Strong E-field
        
        # Add single electron
        electrons = ParticleSpecies.create_electrons(1)
        electrons.positions[0] = [5e-6, 0, 0]
        electrons.velocities[0] = [0, 0, 0]
        electrons.weight = 1.0
        sim.add_species(electrons)
        
        # Run
        sim.run(num_steps=10, diagnostic_interval=10, progress_interval=100)
        
        # Electron should have accelerated in -x direction (negative charge)
        assert electrons.velocities[0, 0] < 0
