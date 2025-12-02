"""
Tests for the Fields module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pic3d.fields import Fields
from pic3d.grid import Grid
from pic3d import constants as const


class TestFields:
    """Test cases for the Fields class."""
    
    def test_field_creation(self):
        """Test basic field creation."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        fields = Fields(grid)
        
        assert fields.Ex.shape == (10, 10, 10)
        assert fields.Ey.shape == (10, 10, 10)
        assert fields.Ez.shape == (10, 10, 10)
        assert fields.Bx.shape == (10, 10, 10)
        assert fields.By.shape == (10, 10, 10)
        assert fields.Bz.shape == (10, 10, 10)
    
    def test_field_energy_zero(self):
        """Test that initial field energy is zero."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        fields = Fields(grid)
        
        E_energy, B_energy = fields.field_energy()
        
        assert E_energy == 0
        assert B_energy == 0
    
    def test_field_energy_nonzero(self):
        """Test field energy with non-zero fields."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        fields = Fields(grid)
        
        # Set uniform E-field
        E0 = 1e6  # V/m
        fields.Ex[:] = E0
        
        E_energy, B_energy = fields.field_energy()
        
        # E_energy = 0.5 * eps0 * E^2 * V
        volume = grid.cell_volume() * grid.nx * grid.ny * grid.nz
        expected = 0.5 * const.EPSILON_0 * E0**2 * volume
        
        assert E_energy == pytest.approx(expected)
        assert B_energy == 0
    
    def test_faraday_law_plane_wave(self):
        """Test Faraday's law with a simple plane wave."""
        grid = Grid(20, 2, 2, (0, 1e-6), (0, 1e-7), (0, 1e-7))
        fields = Fields(grid)
        
        # Set initial E-field (plane wave in y-direction)
        E0 = 1e6
        for i in range(grid.nx):
            x = grid.x[i]
            phase = 2 * np.pi * x / (grid.x_max - grid.x_min)
            fields.Ey[i, :, :] = E0 * np.sin(phase)
        
        initial_Ey = fields.Ey.copy()
        
        # Update B field
        dt = 1e-16
        fields.update_B(dt)
        
        # Bz should change (from dEy/dx)
        assert not np.allclose(fields.Bz, 0)
    
    def test_ampere_law(self):
        """Test Ampere's law update."""
        grid = Grid(20, 2, 2, (0, 1e-6), (0, 1e-7), (0, 1e-7))
        fields = Fields(grid)
        
        # Set initial B-field
        B0 = 1e-3  # 1 mT
        for i in range(grid.nx):
            x = grid.x[i]
            phase = 2 * np.pi * x / (grid.x_max - grid.x_min)
            fields.Bz[i, :, :] = B0 * np.sin(phase)
        
        # Update E field
        dt = 1e-16
        fields.update_E(dt)
        
        # Ey should change (from dBz/dx)
        assert not np.allclose(fields.Ey, 0)
    
    def test_current_density_setting(self):
        """Test current density setting."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        fields = Fields(grid)
        
        J = np.zeros((10, 10, 10, 3))
        J[:, :, :, 0] = 1e5  # Jx
        
        fields.set_current(J)
        
        assert np.allclose(fields.Jx, 1e5)
        assert np.allclose(fields.Jy, 0)
        assert np.allclose(fields.Jz, 0)
    
    def test_current_density_clearing(self):
        """Test current density clearing."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        fields = Fields(grid)
        
        fields.Jx[:] = 1e5
        fields.Jy[:] = 2e5
        fields.Jz[:] = 3e5
        
        fields.clear_current()
        
        assert np.allclose(fields.Jx, 0)
        assert np.allclose(fields.Jy, 0)
        assert np.allclose(fields.Jz, 0)
    
    def test_field_interpolation(self):
        """Test field interpolation to particle positions."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        fields = Fields(grid)
        
        # Set uniform E-field
        fields.Ex[:] = 1e6
        fields.Ey[:] = 2e6
        fields.Ez[:] = 3e6
        
        # Interpolate at grid center
        pos = np.array([[0.5, 0.5, 0.5]])
        E, B = fields.interpolate_to_particles(pos)
        
        assert E[0, 0] == pytest.approx(1e6, rel=0.01)
        assert E[0, 1] == pytest.approx(2e6, rel=0.01)
        assert E[0, 2] == pytest.approx(3e6, rel=0.01)
    
    def test_absorbing_boundary(self):
        """Test absorbing boundary conditions."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        fields = Fields(grid)
        
        # Set non-zero fields everywhere
        fields.Ex[:] = 1e6
        
        fields.apply_boundary_conditions('absorbing')
        
        # Boundary should be zero
        assert fields.Ex[0, 5, 5] == 0
        assert fields.Ex[-1, 5, 5] == 0
        
        # Interior should be non-zero
        assert fields.Ex[5, 5, 5] == 1e6
    
    def test_get_E_field_array(self):
        """Test combined E-field array getter."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        fields = Fields(grid)
        
        fields.Ex[:] = 1
        fields.Ey[:] = 2
        fields.Ez[:] = 3
        
        E = fields.get_E_field()
        
        assert E.shape == (10, 10, 10, 3)
        assert E[5, 5, 5, 0] == 1
        assert E[5, 5, 5, 1] == 2
        assert E[5, 5, 5, 2] == 3
    
    def test_repr(self):
        """Test string representation."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        fields = Fields(grid)
        
        repr_str = repr(fields)
        
        assert "Fields" in repr_str
        assert "V/m" in repr_str
        assert "T" in repr_str
