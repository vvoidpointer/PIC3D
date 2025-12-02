#!/usr/bin/env python3
"""
Example script demonstrating laser-plasma interaction simulation with PIC3D.

This example sets up a laser pulse interacting with a pre-ionized plasma
and demonstrates key phenomena such as:
- Laser propagation and focusing
- Plasma wave generation
- Electron acceleration
- Field energy evolution

Usage:
    python -m pic3d.examples.laser_plasma_example
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pic3d import Simulation, Grid, LaserPulse, ParticleSpecies
from pic3d import constants as const


def run_basic_laser_plasma_simulation():
    """
    Run a basic laser-plasma interaction simulation.
    
    This example demonstrates a relativistic laser pulse (a0 ~ 1)
    interacting with an underdense plasma.
    """
    print("=" * 60)
    print("PIC3D Laser-Plasma Interaction Simulation")
    print("=" * 60)
    
    # Simulation parameters
    laser_wavelength = 800e-9  # 800 nm (Ti:Sapphire)
    laser_intensity = 1e18    # W/cm^2 (relativistic)
    pulse_duration = 30e-15   # 30 fs FWHM
    spot_size = 5e-6          # 5 µm spot
    
    # Plasma parameters
    critical_density = const.critical_density(laser_wavelength)
    plasma_density = 0.1 * critical_density  # 10% of critical density
    plasma_temperature = 10.0  # 10 eV
    
    print(f"\nLaser parameters:")
    print(f"  Wavelength: {laser_wavelength*1e9:.1f} nm")
    print(f"  Intensity: {laser_intensity:.2e} W/cm²")
    print(f"  Pulse duration: {pulse_duration*1e15:.1f} fs")
    print(f"  Spot size: {spot_size*1e6:.1f} µm")
    
    # Calculate a0
    omega = 2 * np.pi * const.C / laser_wavelength
    intensity_si = laser_intensity * 1e4
    E0 = np.sqrt(2 * intensity_si / (const.C * const.EPSILON_0))
    a0 = const.E_CHARGE * E0 / (const.E_MASS * const.C * omega)
    print(f"  Normalized vector potential a0: {a0:.2f}")
    
    print(f"\nPlasma parameters:")
    print(f"  Density: {plasma_density:.2e} m⁻³ ({plasma_density/critical_density:.1%} n_c)")
    print(f"  Temperature: {plasma_temperature} eV")
    print(f"  Critical density: {critical_density:.2e} m⁻³")
    
    # Create simulation with smaller grid for demonstration
    sim = Simulation.create_laser_plasma_simulation(
        nx=32,           # Reduced for faster demo
        ny=16,
        nz=16,
        plasma_density=plasma_density,
        plasma_temperature=plasma_temperature,
        laser_wavelength=laser_wavelength,
        laser_intensity=laser_intensity,
        laser_pulse_duration=pulse_duration,
        laser_spot_size=spot_size,
        particles_per_cell=4  # Reduced for faster demo
    )
    
    # Run simulation
    num_steps = 100
    print(f"\nRunning {num_steps} time steps...")
    
    sim.run(
        num_steps=num_steps,
        diagnostic_interval=10,
        progress_interval=25
    )
    
    # Print final diagnostics
    print("\n" + "=" * 60)
    print("Simulation Results")
    print("=" * 60)
    
    diag = sim.get_diagnostics_array()
    
    print(f"\nEnergy evolution:")
    print(f"  Initial field energy: {diag['field_energy_E'][0] + diag['field_energy_B'][0]:.2e} J")
    print(f"  Final field energy: {diag['field_energy_E'][-1] + diag['field_energy_B'][-1]:.2e} J")
    print(f"  Initial kinetic energy: {diag['kinetic_energy'][0]:.2e} J")
    print(f"  Final kinetic energy: {diag['kinetic_energy'][-1]:.2e} J")
    
    # Check energy conservation (should be approximately conserved)
    energy_change = abs(diag['total_energy'][-1] - diag['total_energy'][0])
    if diag['total_energy'][0] > 0:
        relative_change = energy_change / diag['total_energy'][0]
        print(f"  Energy conservation: {relative_change:.1%} relative change")
    
    print("\nSimulation completed successfully!")
    
    return sim


def run_parametric_study():
    """
    Demonstrate parametric study capability with different laser intensities.
    """
    print("\n" + "=" * 60)
    print("Parametric Study: Effect of Laser Intensity")
    print("=" * 60)
    
    intensities = [1e17, 5e17, 1e18]  # W/cm^2
    results = []
    
    for intensity in intensities:
        print(f"\n--- Running with I = {intensity:.0e} W/cm² ---")
        
        sim = Simulation.create_laser_plasma_simulation(
            nx=32, ny=16, nz=16,
            laser_intensity=intensity,
            particles_per_cell=4
        )
        
        sim.run(num_steps=50, diagnostic_interval=10, progress_interval=50)
        
        diag = sim.get_diagnostics_array()
        final_kinetic = diag['kinetic_energy'][-1]
        
        results.append({
            'intensity': intensity,
            'final_kinetic_energy': final_kinetic,
            'a0': sim.lasers[0].a0
        })
    
    print("\n" + "=" * 60)
    print("Parametric Study Results")
    print("=" * 60)
    print(f"{'Intensity (W/cm²)':<20} {'a0':<10} {'Final KE (J)':<15}")
    print("-" * 45)
    for r in results:
        print(f"{r['intensity']:<20.0e} {r['a0']:<10.2f} {r['final_kinetic_energy']:<15.2e}")


def main():
    """Main entry point for the example script."""
    print("\n" + "=" * 70)
    print(" PIC3D - 3D Particle-In-Cell Simulation for Laser-Plasma Interactions")
    print("=" * 70)
    
    # Run basic simulation
    sim = run_basic_laser_plasma_simulation()
    
    # Optionally run parametric study
    run_parametric = False  # Set to True for extended demo
    if run_parametric:
        run_parametric_study()
    
    print("\n" + "=" * 70)
    print(" Example completed successfully!")
    print("=" * 70)
    
    return sim


if __name__ == "__main__":
    main()
