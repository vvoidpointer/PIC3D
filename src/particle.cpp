#include "particle.h"
#include "constants.h"
#include <random>
#include <omp.h>

ParticleManager::ParticleManager() {}

ParticleManager::~ParticleManager() {}

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

void ParticleManager::updateVelocities(
    const std::vector<std::array<double, 3>>& electricField,
    const std::vector<std::array<double, 3>>& magneticField,
    double dt) {
    
    // Simple Boris pusher (simplified version)
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        // Get field at particle position (simplified: use nearest grid point)
        size_t fieldIdx = i % electricField.size();
        
        double qm = particles[i].charge / particles[i].mass;
        
        // Electric field acceleration
        particles[i].velocity[0] += qm * electricField[fieldIdx][0] * dt;
        particles[i].velocity[1] += qm * electricField[fieldIdx][1] * dt;
        particles[i].velocity[2] += qm * electricField[fieldIdx][2] * dt;
        
        // Magnetic field rotation (simplified)
        double bx = magneticField[fieldIdx][0];
        double by = magneticField[fieldIdx][1];
        double bz = magneticField[fieldIdx][2];
        
        double vx = particles[i].velocity[0];
        double vy = particles[i].velocity[1];
        double vz = particles[i].velocity[2];
        
        double t = qm * dt * 0.5;
        particles[i].velocity[0] += t * (vy * bz - vz * by);
        particles[i].velocity[1] += t * (vz * bx - vx * bz);
        particles[i].velocity[2] += t * (vx * by - vy * bx);
    }
}
