"""
Tests for the Grid module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pic3d.grid import Grid


class TestGrid:
    """Test cases for the Grid class."""
    
    def test_grid_creation(self):
        """Test basic grid creation."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        
        assert grid.nx == 10
        assert grid.ny == 10
        assert grid.nz == 10
        assert grid.dx == pytest.approx(0.1)
        assert grid.dy == pytest.approx(0.1)
        assert grid.dz == pytest.approx(0.1)
    
    def test_grid_asymmetric(self):
        """Test asymmetric grid creation."""
        grid = Grid(20, 10, 5, (0, 2), (-1, 1), (0, 1))
        
        assert grid.nx == 20
        assert grid.ny == 10
        assert grid.nz == 5
        assert grid.dx == pytest.approx(0.1)
        assert grid.dy == pytest.approx(0.2)
        assert grid.dz == pytest.approx(0.2)
    
    def test_cell_volume(self):
        """Test cell volume calculation."""
        grid = Grid(10, 10, 10, (0, 1), (0, 2), (0, 3))
        
        expected_volume = 0.1 * 0.2 * 0.3
        assert grid.cell_volume() == pytest.approx(expected_volume)
    
    def test_grid_coordinates(self):
        """Test grid coordinate arrays."""
        grid = Grid(4, 4, 4, (0, 1), (0, 1), (0, 1))
        
        # Check coordinate array lengths
        assert len(grid.x) == 4
        assert len(grid.y) == 4
        assert len(grid.z) == 4
        
        # Cell centers should be at 0.125, 0.375, 0.625, 0.875
        expected = [0.125, 0.375, 0.625, 0.875]
        np.testing.assert_array_almost_equal(grid.x, expected)
    
    def test_cell_index(self):
        """Test cell index calculation."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        
        # Position at center of first cell
        pos = np.array([[0.05, 0.05, 0.05]])
        indices = grid.get_cell_index(pos)
        assert indices[0, 0] == 0
        assert indices[0, 1] == 0
        assert indices[0, 2] == 0
        
        # Position at center of last cell
        pos = np.array([[0.95, 0.95, 0.95]])
        indices = grid.get_cell_index(pos)
        assert indices[0, 0] == 9
        assert indices[0, 1] == 9
        assert indices[0, 2] == 9
    
    def test_is_inside(self):
        """Test boundary checking."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        
        # Inside
        pos = np.array([[0.5, 0.5, 0.5]])
        assert grid.is_inside(pos)[0]
        
        # Outside
        pos = np.array([[1.5, 0.5, 0.5]])
        assert not grid.is_inside(pos)[0]
        
        pos = np.array([[-0.1, 0.5, 0.5]])
        assert not grid.is_inside(pos)[0]
    
    def test_interpolation_weights(self):
        """Test interpolation weight calculation."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        
        # Position at cell center
        pos = np.array([[0.55, 0.55, 0.55]])
        indices, weights = grid.get_interpolation_weights(pos)
        
        # Weights should sum to 1
        assert np.sum(weights[0]) == pytest.approx(1.0)
        
        # All weights should be non-negative
        assert np.all(weights >= 0)
    
    def test_3d_meshgrid(self):
        """Test 3D meshgrid creation."""
        grid = Grid(4, 4, 4, (0, 1), (0, 1), (0, 1))
        
        assert grid.X.shape == (4, 4, 4)
        assert grid.Y.shape == (4, 4, 4)
        assert grid.Z.shape == (4, 4, 4)
    
    def test_repr(self):
        """Test string representation."""
        grid = Grid(10, 10, 10, (0, 1), (0, 1), (0, 1))
        repr_str = repr(grid)
        
        assert "10x10x10" in repr_str
        assert "Grid" in repr_str
