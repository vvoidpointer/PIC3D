"""
Laser pulse module for PIC3D simulation.

This module provides laser pulse initialization and injection for
laser-plasma interaction simulations.
"""

import numpy as np
from . import constants as const


class LaserPulse:
    """
    Gaussian laser pulse for laser-plasma interaction simulations.
    
    Implements a focused Gaussian beam with temporal Gaussian envelope.
    The laser is injected at a boundary and propagates into the simulation.
    
    Parameters
    ----------
    wavelength : float
        Laser wavelength in meters
    intensity : float
        Peak laser intensity in W/cm^2
    pulse_duration : float
        FWHM pulse duration in seconds
    spot_size : float
        1/e^2 intensity radius at focus in meters
    polarization : str
        Polarization direction: 'x', 'y', or 'circular'
    focus_position : array_like
        Position of laser focus (x, y, z) in meters
    propagation_direction : str
        Direction of propagation: '+x', '-x', '+y', '-y', '+z', '-z'
    """
    
    def __init__(self, wavelength, intensity, pulse_duration, spot_size,
                 polarization='y', focus_position=None, propagation_direction='+x'):
        self.wavelength = wavelength
        self.intensity = intensity  # W/cm^2
        self.pulse_duration = pulse_duration  # FWHM in seconds
        self.spot_size = spot_size  # w0 in meters
        self.polarization = polarization
        self.propagation_direction = propagation_direction
        
        # Derived quantities
        self.omega = 2 * np.pi * const.C / wavelength  # Angular frequency
        self.k = 2 * np.pi / wavelength  # Wavenumber
        self.period = wavelength / const.C  # Optical period
        
        # Convert FWHM to sigma for Gaussian envelope
        self.sigma_t = pulse_duration / (2 * np.sqrt(2 * np.log(2)))
        
        # Rayleigh range
        self.z_R = np.pi * spot_size**2 / wavelength
        
        # Focus position (default at origin)
        if focus_position is None:
            focus_position = [0.0, 0.0, 0.0]
        self.focus_position = np.array(focus_position)
        
        # Calculate peak electric field from intensity
        # I = c * eps_0 * E_0^2 / 2
        intensity_si = intensity * 1e4  # Convert W/cm^2 to W/m^2
        self.E0 = np.sqrt(2 * intensity_si / (const.C * const.EPSILON_0))
        
        # Peak magnetic field
        self.B0 = self.E0 / const.C
        
        # Calculate normalized vector potential a0
        self.a0 = const.E_CHARGE * self.E0 / (const.E_MASS * const.C * self.omega)
    
    @classmethod
    def from_a0(cls, wavelength, a0, pulse_duration, spot_size, **kwargs):
        """
        Create laser pulse from normalized vector potential a0.
        
        a0 = eE0/(m_e * c * omega) is a key parameter in laser-plasma physics.
        a0 > 1 indicates relativistic intensities.
        
        Parameters
        ----------
        wavelength : float
            Laser wavelength in meters
        a0 : float
            Normalized vector potential
        pulse_duration : float
            FWHM pulse duration in seconds
        spot_size : float
            1/e^2 intensity radius at focus in meters
        **kwargs
            Additional arguments passed to __init__
        """
        omega = 2 * np.pi * const.C / wavelength
        E0 = a0 * const.E_MASS * const.C * omega / const.E_CHARGE
        intensity_si = const.C * const.EPSILON_0 * E0**2 / 2
        intensity = intensity_si / 1e4  # Convert to W/cm^2
        
        return cls(wavelength, intensity, pulse_duration, spot_size, **kwargs)
    
    def spot_size_at_z(self, z):
        """
        Calculate spot size at distance z from focus.
        
        Parameters
        ----------
        z : float
            Distance from focus along propagation direction
        
        Returns
        -------
        float
            Spot size w(z)
        """
        return self.spot_size * np.sqrt(1 + (z / self.z_R)**2)
    
    def gouy_phase(self, z):
        """
        Calculate Gouy phase at distance z from focus.
        
        Parameters
        ----------
        z : float
            Distance from focus
        
        Returns
        -------
        float
            Gouy phase in radians
        """
        return np.arctan(z / self.z_R)
    
    def radius_of_curvature(self, z):
        """
        Calculate radius of curvature of wavefront at distance z.
        
        Parameters
        ----------
        z : float
            Distance from focus
        
        Returns
        -------
        float
            Radius of curvature R(z)
        """
        if np.abs(z) < 1e-15:
            return np.inf
        return z * (1 + (self.z_R / z)**2)
    
    def temporal_envelope(self, t, t0=0.0):
        """
        Calculate temporal envelope at time t.
        
        Parameters
        ----------
        t : float or ndarray
            Time
        t0 : float
            Time of pulse center
        
        Returns
        -------
        float or ndarray
            Temporal envelope value
        """
        return np.exp(-((t - t0) / self.sigma_t)**2 / 2)
    
    def get_fields_at_position(self, x, y, z, t):
        """
        Calculate laser E and B fields at given position and time.
        
        Uses paraxial Gaussian beam approximation.
        
        Parameters
        ----------
        x, y, z : float or ndarray
            Position coordinates
        t : float
            Time
        
        Returns
        -------
        E : ndarray
            Electric field (Ex, Ey, Ez)
        B : ndarray
            Magnetic field (Bx, By, Bz)
        """
        # Convert to coordinates relative to focus and propagation direction
        if self.propagation_direction == '+x':
            z_prop = x - self.focus_position[0]
            y_trans = y - self.focus_position[1]
            z_trans = z - self.focus_position[2]
            r_trans = np.sqrt(y_trans**2 + z_trans**2)
            pol_axis = 1 if self.polarization == 'y' else 2
        elif self.propagation_direction == '-x':
            z_prop = -(x - self.focus_position[0])
            y_trans = y - self.focus_position[1]
            z_trans = z - self.focus_position[2]
            r_trans = np.sqrt(y_trans**2 + z_trans**2)
            pol_axis = 1 if self.polarization == 'y' else 2
        elif self.propagation_direction == '+z':
            z_prop = z - self.focus_position[2]
            y_trans = x - self.focus_position[0]
            z_trans = y - self.focus_position[1]
            r_trans = np.sqrt(y_trans**2 + z_trans**2)
            pol_axis = 0 if self.polarization == 'x' else 1
        else:
            # Default to +x propagation
            z_prop = x - self.focus_position[0]
            y_trans = y - self.focus_position[1]
            z_trans = z - self.focus_position[2]
            r_trans = np.sqrt(y_trans**2 + z_trans**2)
            pol_axis = 1 if self.polarization == 'y' else 2
        
        # Gaussian beam parameters at this z
        w_z = self.spot_size_at_z(z_prop)
        
        # Spatial envelope
        spatial_env = (self.spot_size / w_z) * np.exp(-(r_trans / w_z)**2)
        
        # Phase
        if np.abs(z_prop) < 1e-15:
            R_z = np.inf
            curvature_phase = 0.0
        else:
            R_z = self.radius_of_curvature(z_prop)
            curvature_phase = self.k * r_trans**2 / (2 * R_z)
        
        gouy = self.gouy_phase(z_prop)
        phase = self.k * z_prop - self.omega * t + curvature_phase - gouy
        
        # Temporal envelope
        t_retarded = t - z_prop / const.C
        temporal_env = self.temporal_envelope(t_retarded, t0=0.0)
        
        # Total envelope
        envelope = spatial_env * temporal_env
        
        # Calculate fields
        E = np.zeros(3)
        B = np.zeros(3)
        
        if self.polarization in ['x', 'y']:
            # Linear polarization
            E[pol_axis] = self.E0 * envelope * np.cos(phase)
            
            # Magnetic field perpendicular to E and k
            if self.propagation_direction in ['+x', '-x']:
                if pol_axis == 1:  # Ey -> Bz
                    B[2] = E[pol_axis] / const.C
                else:  # Ez -> By
                    B[1] = -E[pol_axis] / const.C
            else:
                if pol_axis == 0:  # Ex -> By
                    B[1] = -E[pol_axis] / const.C
                else:  # Ey -> Bx
                    B[0] = E[pol_axis] / const.C
        
        elif self.polarization == 'circular':
            # Circular polarization
            if self.propagation_direction in ['+x', '-x']:
                E[1] = self.E0 * envelope * np.cos(phase) / np.sqrt(2)
                E[2] = self.E0 * envelope * np.sin(phase) / np.sqrt(2)
                B[1] = -E[2] / const.C
                B[2] = E[1] / const.C
            else:
                E[0] = self.E0 * envelope * np.cos(phase) / np.sqrt(2)
                E[1] = self.E0 * envelope * np.sin(phase) / np.sqrt(2)
                B[0] = E[1] / const.C
                B[1] = -E[0] / const.C
        
        # Flip sign for backward propagation
        if self.propagation_direction.startswith('-'):
            B = -B
        
        return E, B
    
    def inject_at_boundary(self, fields, t, injection_plane='x_min'):
        """
        Inject laser pulse at a simulation boundary.
        
        Parameters
        ----------
        fields : Fields
            The simulation fields object
        t : float
            Current simulation time
        injection_plane : str
            Which boundary to inject at: 'x_min', 'x_max', etc.
        """
        grid = fields.grid
        
        if injection_plane == 'x_min':
            x = grid.x_min
            for j, y in enumerate(grid.y):
                for k, z in enumerate(grid.z):
                    E, B = self.get_fields_at_position(x, y, z, t)
                    fields.Ey[0, j, k] = E[1]
                    fields.Ez[0, j, k] = E[2]
                    fields.By[0, j, k] = B[1]
                    fields.Bz[0, j, k] = B[2]
        
        elif injection_plane == 'x_max':
            x = grid.x_max
            for j, y in enumerate(grid.y):
                for k, z in enumerate(grid.z):
                    E, B = self.get_fields_at_position(x, y, z, t)
                    fields.Ey[-1, j, k] = E[1]
                    fields.Ez[-1, j, k] = E[2]
                    fields.By[-1, j, k] = B[1]
                    fields.Bz[-1, j, k] = B[2]
    
    def add_to_fields(self, fields, t):
        """
        Add laser fields throughout the domain (for initialization).
        
        Parameters
        ----------
        fields : Fields
            The simulation fields object
        t : float
            Current simulation time
        """
        grid = fields.grid
        
        for i, x in enumerate(grid.x):
            for j, y in enumerate(grid.y):
                for k, z in enumerate(grid.z):
                    E, B = self.get_fields_at_position(x, y, z, t)
                    fields.Ex[i, j, k] += E[0]
                    fields.Ey[i, j, k] += E[1]
                    fields.Ez[i, j, k] += E[2]
                    fields.Bx[i, j, k] += B[0]
                    fields.By[i, j, k] += B[1]
                    fields.Bz[i, j, k] += B[2]
    
    def critical_density(self):
        """Return critical density for this laser wavelength."""
        return const.critical_density(self.wavelength)
    
    def ponderomotive_energy(self):
        """
        Calculate ponderomotive energy in eV.
        
        The ponderomotive energy is the cycle-averaged quiver energy
        of an electron in the laser field.
        """
        # U_p = e^2 * E_0^2 / (4 * m_e * omega^2)
        U_p_joules = (const.E_CHARGE**2 * self.E0**2 / 
                     (4 * const.E_MASS * self.omega**2))
        return U_p_joules / const.E_CHARGE  # Convert to eV
    
    def __repr__(self):
        return (f"LaserPulse(λ={self.wavelength*1e9:.1f} nm, "
                f"I={self.intensity:.2e} W/cm², "
                f"a0={self.a0:.2f}, "
                f"τ={self.pulse_duration*1e15:.1f} fs, "
                f"w0={self.spot_size*1e6:.1f} μm)")
