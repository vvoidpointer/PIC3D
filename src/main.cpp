#include <iostream>
#include <omp.h>
#include "particle.h"
#include "field.h"
#include "grid.h"
#include "io.h"

int main(int /* argc */, char* /* argv */[]) {
    std::cout << "PIC3D - Particle-in-Cell 3D Simulation" << std::endl;
    std::cout << "======================================" << std::endl;

    // Display OpenMP info
    #pragma omp parallel
    {
        #pragma omp single
        {
            std::cout << "OpenMP threads: " << omp_get_num_threads() << std::endl;
        }
    }

    // Simulation parameters
    const int nx = 32, ny = 32, nz = 32;
    const double Lx = 1.0, Ly = 1.0, Lz = 1.0;
    const double dt = 0.001;
    const int numParticles = 1000;
    const int numSteps = 100;
    const int outputInterval = 10;

    // Initialize grid
    Grid grid(nx, ny, nz, Lx, Ly, Lz);
    std::cout << "Grid: " << nx << " x " << ny << " x " << nz << std::endl;
    std::cout << "Domain: " << Lx << " x " << Ly << " x " << Lz << std::endl;

    // Initialize fields
    Field field(nx, ny, nz, grid.getDx(), grid.getDy(), grid.getDz());
    field.initialize();
    std::cout << "Fields initialized" << std::endl;

    // Initialize particles
    ParticleManager particleManager;
    particleManager.initializeParticles(numParticles, Lx);
    std::cout << "Particles: " << particleManager.getNumParticles() << std::endl;

    // Initialize HDF5 writer
    HDF5Writer writer("output.h5");
    writer.writeMetadata(nx, ny, nz, grid.getDx(), grid.getDy(), grid.getDz(), dt);

    // Main simulation loop
    std::cout << "\nStarting simulation..." << std::endl;
    for (int step = 0; step < numSteps; ++step) {
        // Update velocities using fields
        particleManager.updateVelocities(field.getElectricField(), 
                                         field.getMagneticField(), dt);
        
        // Update positions
        particleManager.updatePositions(dt);

        // Deposit charge onto grid
        std::vector<std::array<double, 3>> positions;
        std::vector<double> charges;
        for (const auto& p : particleManager.getParticles()) {
            positions.push_back(p.position);
            charges.push_back(p.charge);
        }
        field.depositCharge(positions, charges);

        // Update fields
        field.updateElectricField(dt);
        field.updateMagneticField(dt);

        // Output
        if (step % outputInterval == 0) {
            std::cout << "Step " << step << "/" << numSteps << std::endl;
            writer.writeParticles(particleManager.getParticles(), step);
            writer.writeField(field, step);
        }
    }

    std::cout << "\nSimulation complete!" << std::endl;
    return 0;
}
