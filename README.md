# PIC3D

A 3D Particle-In-Cell (PIC) simulation framework for laser-plasma interactions.

## Overview

PIC3D is a Python-based simulation framework that implements the Particle-In-Cell method for studying laser-plasma interaction phenomena. It provides a complete toolkit for simulating relativistic laser pulses interacting with pre-ionized plasmas.

### Key Features

- **3D FDTD Field Solver**: Finite-Difference Time-Domain method for solving Maxwell's equations on a Yee lattice
- **Boris Particle Pusher**: Relativistic particle pusher with volume-preserving properties
- **Gaussian Laser Pulses**: Focused Gaussian beams with temporal envelope and paraxial propagation
- **Multiple Particle Species**: Support for electrons, protons, and arbitrary ion species
- **Flexible Boundary Conditions**: Periodic, absorbing, and reflecting boundaries
- **Comprehensive Diagnostics**: Energy tracking, field slicing, and density calculations

## Physics Background

### Laser-Plasma Interactions

When an intense laser pulse propagates through a plasma, several important phenomena occur:

1. **Plasma Oscillations**: Electrons oscillate in the laser's electric field
2. **Wake Field Generation**: The ponderomotive force of the laser creates electron density waves
3. **Electron Acceleration**: Trapped electrons can gain significant energy from the wakefield
4. **Self-Focusing**: At high intensities, the plasma acts as a lens

### Key Parameters

- **Normalized Vector Potential (a₀)**: `a₀ = eE₀/(m_e c ω)` - For a₀ > 1, electron motion becomes relativistic
- **Critical Density (n_c)**: The plasma density at which the laser frequency equals the plasma frequency
- **Debye Length**: The characteristic shielding length in a plasma

## Installation

```bash
# Clone the repository
git clone https://github.com/vvoidpointer/PIC3D.git
cd PIC3D

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
from pic3d import Simulation

# Create a pre-configured laser-plasma simulation
sim = Simulation.create_laser_plasma_simulation(
    nx=64, ny=32, nz=32,           # Grid dimensions
    plasma_density=1e25,           # Electron density (m⁻³)
    plasma_temperature=10.0,       # Temperature (eV)
    laser_wavelength=800e-9,       # 800 nm Ti:Sapphire
    laser_intensity=1e18,          # W/cm² (relativistic)
    laser_pulse_duration=30e-15,   # 30 fs FWHM
    laser_spot_size=5e-6,          # 5 µm spot
    particles_per_cell=8
)

# Run the simulation
sim.run(num_steps=1000, diagnostic_interval=100)

# Access diagnostics
diagnostics = sim.get_diagnostics_array()
```

## Custom Simulations

```python
from pic3d import Simulation, Grid, LaserPulse, ParticleSpecies
import pic3d.constants as const

# Create custom grid
grid = Grid(
    nx=100, ny=50, nz=50,
    x_range=(0, 50e-6),
    y_range=(-25e-6, 25e-6),
    z_range=(-25e-6, 25e-6)
)

# Create simulation
sim = Simulation(grid, boundary_conditions='absorbing')

# Add laser pulse
laser = LaserPulse.from_a0(
    wavelength=800e-9,
    a0=2.0,  # Relativistic intensity
    pulse_duration=30e-15,
    spot_size=10e-6,
    polarization='circular',
    focus_position=[25e-6, 0, 0]
)
sim.add_laser(laser)

# Add electron species
electrons = ParticleSpecies.create_electrons(num_particles=100000)
electrons.initialize_uniform(grid, density=1e25)
electrons.initialize_maxwellian(temperature_ev=100)
sim.add_species(electrons)

# Add ion species
ions = ParticleSpecies.create_protons(num_particles=100000)
ions.positions[:] = electrons.positions.copy()
ions.weight = electrons.weight
sim.add_species(ions)

# Run simulation
sim.run(num_steps=500)
```

## Running the Example

```bash
python -m pic3d.examples.laser_plasma_example
```

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
PIC3D/
├── pic3d/
│   ├── __init__.py          # Package initialization
│   ├── constants.py         # Physical constants
│   ├── grid.py              # Grid class for spatial discretization
│   ├── particles.py         # Particle species and Boris pusher
│   ├── fields.py            # Electromagnetic field solver (FDTD)
│   ├── laser.py             # Laser pulse initialization
│   ├── simulation.py        # Main simulation driver
│   └── examples/
│       └── laser_plasma_example.py
├── tests/
│   ├── test_grid.py
│   ├── test_particles.py
│   ├── test_fields.py
│   ├── test_laser.py
│   └── test_simulation.py
├── requirements.txt
├── setup.py
└── README.md
```

## Physical Algorithms

### FDTD Field Solver

The electromagnetic fields are updated using the Yee algorithm:

```
∂B/∂t = -∇×E       (Faraday's Law)
∂E/∂t = c²∇×B - J/ε₀  (Ampere's Law with current source)
```

### Boris Particle Pusher

Particles are advanced using the Boris algorithm, which is second-order accurate and preserves phase space volume:

1. Half acceleration from E-field
2. Rotation from B-field
3. Half acceleration from E-field
4. Position update

### Charge/Current Deposition

Cloud-In-Cell (CIC) interpolation is used for:
- Depositing particle charge/current onto the grid
- Interpolating fields from grid to particle positions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## References

1. Birdsall, C. K., & Langdon, A. B. (1991). *Plasma Physics via Computer Simulation*
2. Tajima, T., & Dawson, J. M. (1979). Laser Electron Accelerator. *Physical Review Letters*
3. Esarey, E., Schroeder, C. B., & Leemans, W. P. (2009). Physics of laser-driven plasma-based electron accelerators. *Reviews of Modern Physics*
