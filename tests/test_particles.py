"""
Tests for the ParticleSpecies module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pic3d.particles import ParticleSpecies
from pic3d.grid import Grid
from pic3d import constants as const


class TestParticleSpecies:
    """Test cases for the ParticleSpecies class."""
    
    def test_electron_creation(self):
        """Test electron species creation."""
        electrons = ParticleSpecies.create_electrons(1000)
        
        assert electrons.name == 'electrons'
        assert electrons.charge == -1.0
        assert electrons.mass == 1.0
        assert electrons.num_particles == 1000
    
    def test_proton_creation(self):
        """Test proton species creation."""
        protons = ParticleSpecies.create_protons(500)
        
        assert protons.name == 'protons'
        assert protons.charge == 1.0
        assert protons.mass > 1800  # mp/me ratio
        assert protons.num_particles == 500
    
    def test_ion_creation(self):
        """Test ion species creation."""
        # Carbon ion C6+
        carbon = ParticleSpecies.create_ions(100, charge_state=6, mass_amu=12)
        
        assert carbon.charge == 6.0
        assert carbon.num_particles == 100
    
    def test_particle_arrays_initialized(self):
        """Test that particle arrays are properly initialized."""
        species = ParticleSpecies.create_electrons(100)
        
        assert species.positions.shape == (100, 3)
        assert species.velocities.shape == (100, 3)
        assert species.gamma.shape == (100,)
        assert species.active.shape == (100,)
        
        # All particles should be active initially
        assert np.all(species.active)
    
    def test_uniform_initialization(self):
        """Test uniform distribution initialization."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        electrons = ParticleSpecies.create_electrons(1000)
        
        density = 1e20  # m^-3
        electrons.initialize_uniform(grid, density)
        
        # All positions should be within domain
        assert np.all(electrons.positions[:, 0] >= 0)
        assert np.all(electrons.positions[:, 0] <= 1)
        assert np.all(electrons.positions[:, 1] >= 0)
        assert np.all(electrons.positions[:, 1] <= 1)
        assert np.all(electrons.positions[:, 2] >= 0)
        assert np.all(electrons.positions[:, 2] <= 1)
    
    def test_maxwellian_initialization(self):
        """Test Maxwellian velocity initialization."""
        electrons = ParticleSpecies.create_electrons(10000)
        T_eV = 100.0  # 100 eV
        
        electrons.initialize_maxwellian(T_eV)
        
        # Check that velocities are distributed
        assert np.std(electrons.velocities[:, 0]) > 0
        assert np.std(electrons.velocities[:, 1]) > 0
        assert np.std(electrons.velocities[:, 2]) > 0
        
        # Mean should be close to zero
        assert np.abs(np.mean(electrons.velocities[:, 0])) < np.std(electrons.velocities[:, 0])
    
    def test_charge_density_deposition(self):
        """Test charge density deposition."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        electrons = ParticleSpecies.create_electrons(1)
        
        # Place electron at center of domain
        electrons.positions[0] = [0.5, 0.5, 0.5]
        electrons.weight = 1e10
        
        rho = electrons.get_charge_density(grid)
        
        # Check that total charge is conserved
        total_charge = np.sum(rho) * grid.cell_volume()
        expected_charge = electrons.charge * const.E_CHARGE * electrons.weight
        assert total_charge == pytest.approx(expected_charge, rel=0.01)
    
    def test_current_density_deposition(self):
        """Test current density deposition."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        electrons = ParticleSpecies.create_electrons(1)
        
        # Place electron with velocity in x direction
        electrons.positions[0] = [0.5, 0.5, 0.5]
        electrons.velocities[0] = [1e6, 0, 0]  # 1e6 m/s in x
        electrons.weight = 1e10
        
        J = electrons.get_current_density(grid)
        
        # Check current is in x direction
        assert J.shape == (10, 10, 10, 3)
        
        # Total Jx should be non-zero
        assert np.sum(J[:, :, :, 0]) != 0
    
    def test_periodic_boundary(self):
        """Test periodic boundary conditions."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        electrons = ParticleSpecies.create_electrons(1)
        
        # Place electron outside domain
        electrons.positions[0] = [1.2, 0.5, 0.5]
        
        electrons.apply_boundary_conditions(grid, 'periodic')
        
        # Should wrap around
        assert 0 <= electrons.positions[0, 0] <= 1
    
    def test_absorbing_boundary(self):
        """Test absorbing boundary conditions."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        electrons = ParticleSpecies.create_electrons(1)
        
        # Place electron outside domain
        electrons.positions[0] = [1.2, 0.5, 0.5]
        
        electrons.apply_boundary_conditions(grid, 'absorbing')
        
        # Particle should be marked inactive
        assert not electrons.active[0]
    
    def test_kinetic_energy(self):
        """Test kinetic energy calculation."""
        electrons = ParticleSpecies.create_electrons(1)
        
        # Electron with 0.1c velocity
        v = 0.1 * const.C
        electrons.velocities[0] = [v, 0, 0]
        electrons._update_gamma()
        electrons.weight = 1.0
        
        KE = electrons.kinetic_energy()
        
        # Non-relativistic approximation: KE ≈ 0.5 * m * v^2
        KE_nonrel = 0.5 * const.E_MASS * v**2
        
        # Should be close (within relativistic correction)
        assert KE == pytest.approx(KE_nonrel, rel=0.1)
    
    def test_repr(self):
        """Test string representation."""
        electrons = ParticleSpecies.create_electrons(100)
        repr_str = repr(electrons)
        
        assert 'electrons' in repr_str
        assert '100' in repr_str
