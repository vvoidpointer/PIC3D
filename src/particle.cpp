#include "particle.h"
#include "constants.h"
#include <random>
#include <omp.h>
#include <cmath>

ParticleManager::ParticleManager() : gridNx(0), gridNy(0), gridNz(0), 
                                     gridDx(1.0), gridDy(1.0), gridDz(1.0) {}

ParticleManager::~ParticleManager() {}

void ParticleManager::setGridParameters(int nx, int ny, int nz, double dx, double dy, double dz) {
    gridNx = nx;
    gridNy = ny;
    gridNz = nz;
    gridDx = dx;
    gridDy = dy;
    gridDz = dz;
}

void ParticleManager::addParticle(const Particle& p) {
    particles.push_back(p);
}

void ParticleManager::initializeParticles(int numParticles, double boxSize) {
    particles.clear();
    particles.reserve(numParticles);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> posDist(0.0, boxSize);
    std::normal_distribution<> velDist(0.0, 1e5);  // thermal velocity

    for (int i = 0; i < numParticles; ++i) {
        Particle p;
        p.position = {posDist(gen), posDist(gen), posDist(gen)};
        p.velocity = {velDist(gen), velDist(gen), velDist(gen)};
        p.charge = -constants::e;  // electrons
        p.mass = constants::me;
        particles.push_back(p);
    }
}

void ParticleManager::updatePositions(double dt) {
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].position[0] += particles[i].velocity[0] * dt;
        particles[i].position[1] += particles[i].velocity[1] * dt;
        particles[i].position[2] += particles[i].velocity[2] * dt;
    }
}

size_t ParticleManager::getFieldIndex(double x, double y, double z) const {
    // Convert position to grid indices with proper handling of boundaries
    int i = static_cast<int>(std::floor(x / gridDx));
    int j = static_cast<int>(std::floor(y / gridDy));
    int k = static_cast<int>(std::floor(z / gridDz));
    
    // Apply periodic boundary conditions
    i = ((i % gridNx) + gridNx) % gridNx;
    j = ((j % gridNy) + gridNy) % gridNy;
    k = ((k % gridNz) + gridNz) % gridNz;
    
    return static_cast<size_t>(i + gridNx * (j + gridNy * k));
}

void ParticleManager::updateVelocities(
    const std::vector<std::array<double, 3>>& electricField,
    const std::vector<std::array<double, 3>>& magneticField,
    double dt) {
    
    // Boris pusher algorithm for particle velocity update
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        // Get field at particle position using nearest grid point interpolation
        size_t fieldIdx = getFieldIndex(particles[i].position[0], 
                                        particles[i].position[1], 
                                        particles[i].position[2]);
        
        double qm = particles[i].charge / particles[i].mass;
        double halfDt = 0.5 * dt;
        
        // Get field values
        double Ex = electricField[fieldIdx][0];
        double Ey = electricField[fieldIdx][1];
        double Ez = electricField[fieldIdx][2];
        double Bx = magneticField[fieldIdx][0];
        double By = magneticField[fieldIdx][1];
        double Bz = magneticField[fieldIdx][2];
        
        // Boris pusher: half electric field acceleration
        double vx_minus = particles[i].velocity[0] + qm * Ex * halfDt;
        double vy_minus = particles[i].velocity[1] + qm * Ey * halfDt;
        double vz_minus = particles[i].velocity[2] + qm * Ez * halfDt;
        
        // Boris pusher: magnetic rotation
        double tx = qm * Bx * halfDt;
        double ty = qm * By * halfDt;
        double tz = qm * Bz * halfDt;
        
        double t_mag_sq = tx * tx + ty * ty + tz * tz;
        double sx = 2.0 * tx / (1.0 + t_mag_sq);
        double sy = 2.0 * ty / (1.0 + t_mag_sq);
        double sz = 2.0 * tz / (1.0 + t_mag_sq);
        
        // v' = v- + v- x t
        double vx_prime = vx_minus + (vy_minus * tz - vz_minus * ty);
        double vy_prime = vy_minus + (vz_minus * tx - vx_minus * tz);
        double vz_prime = vz_minus + (vx_minus * ty - vy_minus * tx);
        
        // v+ = v- + v' x s
        double vx_plus = vx_minus + (vy_prime * sz - vz_prime * sy);
        double vy_plus = vy_minus + (vz_prime * sx - vx_prime * sz);
        double vz_plus = vz_minus + (vx_prime * sy - vy_prime * sx);
        
        // Boris pusher: second half electric field acceleration
        particles[i].velocity[0] = vx_plus + qm * Ex * halfDt;
        particles[i].velocity[1] = vy_plus + qm * Ey * halfDt;
        particles[i].velocity[2] = vz_plus + qm * Ez * halfDt;
    }
}
