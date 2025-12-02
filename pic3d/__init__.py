"""
PIC3D - A 3D Particle-In-Cell Simulation Framework for Laser-Plasma Interactions

This package provides tools for simulating laser-plasma interactions using the
Particle-In-Cell (PIC) method. It includes modules for:
- Grid management and field interpolation
- Particle dynamics and species handling
- Electromagnetic field solvers (FDTD for Maxwell's equations)
- Laser pulse initialization
- Diagnostics and output
"""

__version__ = "0.1.0"

from .grid import Grid
from .particles import ParticleSpecies
from .fields import Fields
from .laser import LaserPulse
from .simulation import Simulation

__all__ = [
    "Grid",
    "ParticleSpecies",
    "Fields",
    "LaserPulse",
    "Simulation",
]
