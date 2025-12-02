"""
Physical constants used in PIC simulations.

All values are in SI units unless otherwise specified.
Normalized units are commonly used in PIC codes where:
- c = 1 (speed of light)
- e = 1 (electron charge)
- m_e = 1 (electron mass)
- epsilon_0 = 1 (vacuum permittivity)
"""

import numpy as np

# Physical constants in SI units
C = 2.99792458e8  # Speed of light (m/s)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
E_MASS = 9.1093837015e-31  # Electron mass (kg)
PROTON_MASS = 1.67262192369e-27  # Proton mass (kg)
EPSILON_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
MU_0 = 1.25663706212e-6  # Vacuum permeability (H/m)
HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)

# Derived constants
OMEGA_P_COEFF = np.sqrt(E_CHARGE**2 / (EPSILON_0 * E_MASS))  # Plasma frequency coefficient


def plasma_frequency(n_e):
    """
    Calculate plasma frequency for given electron density.
    
    Parameters
    ----------
    n_e : float
        Electron density in m^-3
    
    Returns
    -------
    float
        Plasma frequency omega_p in rad/s
    """
    return OMEGA_P_COEFF * np.sqrt(n_e)


def critical_density(wavelength):
    """
    Calculate critical electron density for a given laser wavelength.
    
    Parameters
    ----------
    wavelength : float
        Laser wavelength in meters
    
    Returns
    -------
    float
        Critical density in m^-3
    """
    omega_laser = 2 * np.pi * C / wavelength
    return EPSILON_0 * E_MASS * omega_laser**2 / E_CHARGE**2


def debye_length(T_e, n_e):
    """
    Calculate Debye length for given electron temperature and density.
    
    Parameters
    ----------
    T_e : float
        Electron temperature in eV
    n_e : float
        Electron density in m^-3
    
    Returns
    -------
    float
        Debye length in meters
    """
    k_B_eV = 1.0  # In eV units, k_B*T is just T
    T_e_joules = T_e * E_CHARGE
    return np.sqrt(EPSILON_0 * T_e_joules / (n_e * E_CHARGE**2))
