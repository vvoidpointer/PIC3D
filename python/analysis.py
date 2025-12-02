#!/usr/bin/env python3
"""
PIC3D Analysis Script

This script provides tools for analyzing and visualizing output from the PIC3D simulation.
Requires: numpy, h5py, matplotlib
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_metadata(filename):
    """Load simulation metadata from HDF5 file."""
    with h5py.File(filename, 'r') as f:
        metadata = {}
        if 'metadata' in f:
            meta_group = f['metadata']
            metadata['nx'] = meta_group['nx'][0]
            metadata['ny'] = meta_group['ny'][0]
            metadata['nz'] = meta_group['nz'][0]
            metadata['dx'] = meta_group['dx'][0]
            metadata['dy'] = meta_group['dy'][0]
            metadata['dz'] = meta_group['dz'][0]
            metadata['dt'] = meta_group['dt'][0]
        return metadata


def load_particles(filename, timestep):
    """Load particle data from HDF5 file at a specific timestep."""
    group_name = f'particles_{timestep:06d}'
    with h5py.File(filename, 'r') as f:
        if group_name in f:
            group = f[group_name]
            positions = np.array(group['position'])
            velocities = np.array(group['velocity'])
            return positions, velocities
        else:
            raise ValueError(f"Timestep {timestep} not found in file")


def load_fields(filename, timestep):
    """Load field data from HDF5 file at a specific timestep."""
    group_name = f'fields_{timestep:06d}'
    with h5py.File(filename, 'r') as f:
        if group_name in f:
            group = f[group_name]
            E = np.array(group['electric_field'])
            B = np.array(group['magnetic_field'])
            rho = np.array(group['charge_density'])
            return E, B, rho
        else:
            raise ValueError(f"Timestep {timestep} not found in file")


def get_timesteps(filename):
    """Get list of available timesteps in the HDF5 file."""
    timesteps = []
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            if key.startswith('particles_'):
                timesteps.append(int(key.split('_')[1]))
    return sorted(timesteps)


def plot_particles_3d(positions, title="Particle Distribution"):
    """Create 3D scatter plot of particle positions."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               s=1, alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    return fig, ax


def plot_velocity_distribution(velocities, component=0, bins=50):
    """Plot velocity distribution histogram."""
    labels = ['Vx', 'Vy', 'Vz']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(velocities[:, component], bins=bins, density=True, alpha=0.7)
    ax.set_xlabel(f'{labels[component]} (m/s)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'{labels[component]} Distribution')
    
    return fig, ax


def plot_field_slice(field, axis=2, slice_idx=None, component=0):
    """Plot 2D slice of a vector field component."""
    if slice_idx is None:
        slice_idx = field.shape[axis] // 2
    
    if axis == 0:
        data = field[slice_idx, :, :, component]
        xlabel, ylabel = 'Y', 'Z'
    elif axis == 1:
        data = field[:, slice_idx, :, component]
        xlabel, ylabel = 'X', 'Z'
    else:
        data = field[:, :, slice_idx, component]
        xlabel, ylabel = 'X', 'Y'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data.T, origin='lower', aspect='equal', cmap='RdBu_r')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    return fig, ax


def plot_charge_density_slice(rho, axis=2, slice_idx=None):
    """Plot 2D slice of charge density."""
    if slice_idx is None:
        slice_idx = rho.shape[axis] // 2
    
    if axis == 0:
        data = rho[slice_idx, :, :]
        xlabel, ylabel = 'Y', 'Z'
    elif axis == 1:
        data = rho[:, slice_idx, :]
        xlabel, ylabel = 'X', 'Z'
    else:
        data = rho[:, :, slice_idx]
        xlabel, ylabel = 'X', 'Y'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data.T, origin='lower', aspect='equal', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Charge Density')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    return fig, ax


def compute_kinetic_energy(velocities, mass):
    """Compute total kinetic energy of particles."""
    v_squared = np.sum(velocities**2, axis=1)
    return 0.5 * mass * np.sum(v_squared)


def compute_field_energy(E, B, dx, dy, dz, epsilon0, mu0):
    """Compute electromagnetic field energy."""
    E_squared = np.sum(E**2)
    B_squared = np.sum(B**2)
    
    volume = dx * dy * dz
    E_energy = 0.5 * epsilon0 * E_squared * volume
    B_energy = 0.5 * B_squared / mu0 * volume
    
    return E_energy, B_energy


def main():
    """Main analysis routine."""
    filename = 'output.h5'
    
    print("PIC3D Analysis")
    print("=" * 40)
    
    # Load metadata
    metadata = load_metadata(filename)
    print(f"Grid: {metadata.get('nx', 'N/A')} x {metadata.get('ny', 'N/A')} x {metadata.get('nz', 'N/A')}")
    print(f"Time step: {metadata.get('dt', 'N/A')}")
    
    # Get available timesteps
    timesteps = get_timesteps(filename)
    print(f"Available timesteps: {len(timesteps)}")
    
    if len(timesteps) > 0:
        # Load and plot final timestep
        final_step = timesteps[-1]
        print(f"\nAnalyzing timestep {final_step}...")
        
        positions, velocities = load_particles(filename, final_step)
        E, B, rho = load_fields(filename, final_step)
        
        print(f"Number of particles: {len(positions)}")
        print(f"Position range: [{positions.min():.3e}, {positions.max():.3e}]")
        print(f"Velocity range: [{velocities.min():.3e}, {velocities.max():.3e}]")
        print(f"Max |E|: {np.sqrt(np.max(np.sum(E**2, axis=-1))):.3e}")
        print(f"Max |B|: {np.sqrt(np.max(np.sum(B**2, axis=-1))):.3e}")
        print(f"Max |rho|: {np.max(np.abs(rho)):.3e}")
        
        # Create plots
        fig1, ax1 = plot_particles_3d(positions, f"Particles at t={final_step}")
        fig2, ax2 = plot_velocity_distribution(velocities, component=0)
        fig3, ax3 = plot_charge_density_slice(rho)
        
        plt.show()


if __name__ == '__main__':
    main()
