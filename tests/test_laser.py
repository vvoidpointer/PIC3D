"""
Tests for the LaserPulse module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pic3d.laser import LaserPulse
from pic3d.fields import Fields
from pic3d.grid import Grid
from pic3d import constants as const


class TestLaserPulse:
    """Test cases for the LaserPulse class."""
    
    def test_laser_creation(self):
        """Test basic laser creation."""
        laser = LaserPulse(
            wavelength=800e-9,
            intensity=1e18,
            pulse_duration=30e-15,
            spot_size=5e-6
        )
        
        assert laser.wavelength == 800e-9
        assert laser.intensity == 1e18
        assert laser.pulse_duration == 30e-15
        assert laser.spot_size == 5e-6
    
    def test_laser_from_a0(self):
        """Test laser creation from normalized vector potential."""
        a0 = 1.0
        laser = LaserPulse.from_a0(
            wavelength=800e-9,
            a0=a0,
            pulse_duration=30e-15,
            spot_size=5e-6
        )
        
        assert laser.a0 == pytest.approx(a0, rel=0.01)
    
    def test_derived_quantities(self):
        """Test derived laser quantities."""
        laser = LaserPulse(
            wavelength=800e-9,
            intensity=1e18,
            pulse_duration=30e-15,
            spot_size=5e-6
        )
        
        # Angular frequency
        expected_omega = 2 * np.pi * const.C / 800e-9
        assert laser.omega == pytest.approx(expected_omega)
        
        # Wavenumber
        expected_k = 2 * np.pi / 800e-9
        assert laser.k == pytest.approx(expected_k)
        
        # Rayleigh range
        expected_zR = np.pi * (5e-6)**2 / 800e-9
        assert laser.z_R == pytest.approx(expected_zR)
    
    def test_spot_size_evolution(self):
        """Test Gaussian beam spot size evolution."""
        laser = LaserPulse(
            wavelength=800e-9,
            intensity=1e18,
            pulse_duration=30e-15,
            spot_size=5e-6
        )
        
        # At focus
        w0 = laser.spot_size_at_z(0)
        assert w0 == pytest.approx(5e-6)
        
        # At Rayleigh range
        w_zR = laser.spot_size_at_z(laser.z_R)
        assert w_zR == pytest.approx(5e-6 * np.sqrt(2))
    
    def test_gouy_phase(self):
        """Test Gouy phase calculation."""
        laser = LaserPulse(
            wavelength=800e-9,
            intensity=1e18,
            pulse_duration=30e-15,
            spot_size=5e-6
        )
        
        # At focus
        gouy_0 = laser.gouy_phase(0)
        assert gouy_0 == 0
        
        # At Rayleigh range
        gouy_zR = laser.gouy_phase(laser.z_R)
        assert gouy_zR == pytest.approx(np.pi / 4)
    
    def test_temporal_envelope(self):
        """Test temporal envelope."""
        laser = LaserPulse(
            wavelength=800e-9,
            intensity=1e18,
            pulse_duration=30e-15,
            spot_size=5e-6
        )
        
        # At center
        env_0 = laser.temporal_envelope(0, t0=0)
        assert env_0 == pytest.approx(1.0)
        
        # Away from center
        env_far = laser.temporal_envelope(1e-12, t0=0)
        assert env_far < 1.0
    
    def test_field_at_focus(self):
        """Test field values at focus."""
        laser = LaserPulse(
            wavelength=800e-9,
            intensity=1e18,
            pulse_duration=30e-15,
            spot_size=5e-6,
            polarization='y',
            focus_position=[0, 0, 0],
            propagation_direction='+x'
        )
        
        # At focus, on-axis, t=0
        E, B = laser.get_fields_at_position(0, 0, 0, 0)
        
        # Maximum E-field should be close to E0
        assert np.abs(E[1]) == pytest.approx(laser.E0, rel=0.1)
        
        # B-field should be E/c
        assert np.abs(B[2]) == pytest.approx(np.abs(E[1]) / const.C, rel=0.1)
    
    def test_field_off_axis(self):
        """Test that field decreases off-axis."""
        laser = LaserPulse(
            wavelength=800e-9,
            intensity=1e18,
            pulse_duration=30e-15,
            spot_size=5e-6,
            focus_position=[0, 0, 0]
        )
        
        # On-axis
        E_on, _ = laser.get_fields_at_position(0, 0, 0, 0)
        
        # Off-axis
        E_off, _ = laser.get_fields_at_position(0, laser.spot_size, 0, 0)
        
        # Off-axis should be smaller
        assert np.linalg.norm(E_off) < np.linalg.norm(E_on)
    
    def test_critical_density(self):
        """Test critical density calculation."""
        laser = LaserPulse(
            wavelength=1e-6,  # 1 µm
            intensity=1e18,
            pulse_duration=30e-15,
            spot_size=5e-6
        )
        
        n_c = laser.critical_density()
        
        # Critical density for 1 µm is about 1.1e27 m^-3
        assert 1e27 < n_c < 2e27
    
    def test_ponderomotive_energy(self):
        """Test ponderomotive energy calculation."""
        laser = LaserPulse(
            wavelength=800e-9,
            intensity=1e18,
            pulse_duration=30e-15,
            spot_size=5e-6
        )
        
        U_p = laser.ponderomotive_energy()
        
        # Should be positive and in reasonable range for this intensity
        assert U_p > 0
        assert U_p < 1e6  # Less than 1 MeV for this intensity
    
    def test_inject_at_boundary(self):
        """Test laser injection at boundary."""
        grid = Grid(10, 10, 10, (0, 1e-5), (-5e-6, 5e-6), (-5e-6, 5e-6))
        fields = Fields(grid)
        
        laser = LaserPulse(
            wavelength=800e-9,
            intensity=1e18,
            pulse_duration=30e-15,
            spot_size=5e-6,
            focus_position=[0, 0, 0]
        )
        
        laser.inject_at_boundary(fields, t=0, injection_plane='x_min')
        
        # Check that fields were set at boundary
        assert not np.allclose(fields.Ey[0, :, :], 0)
    
    def test_circular_polarization(self):
        """Test circular polarization."""
        laser = LaserPulse(
            wavelength=800e-9,
            intensity=1e18,
            pulse_duration=30e-15,
            spot_size=5e-6,
            polarization='circular',
            focus_position=[0, 0, 0]
        )
        
        # At different phases, Ey and Ez should have different ratios
        E1, _ = laser.get_fields_at_position(0, 0, 0, 0)
        
        # Both Ey and Ez should be non-zero for circular polarization
        # (at least one of them)
        assert E1[1] != 0 or E1[2] != 0
    
    def test_repr(self):
        """Test string representation."""
        laser = LaserPulse(
            wavelength=800e-9,
            intensity=1e18,
            pulse_duration=30e-15,
            spot_size=5e-6
        )
        
        repr_str = repr(laser)
        
        assert "LaserPulse" in repr_str
        assert "800" in repr_str  # wavelength in nm
        assert "a0" in repr_str
