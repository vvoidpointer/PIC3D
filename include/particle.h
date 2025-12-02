#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>
#include <array>

struct Particle {
    std::array<double, 3> position;  // x, y, z
    std::array<double, 3> velocity;  // vx, vy, vz
    double charge;
    double mass;
};

class ParticleManager {
public:
    ParticleManager();
    ~ParticleManager();

    void setGridParameters(int nx, int ny, int nz, double dx, double dy, double dz);
    void addParticle(const Particle& p);
    void initializeParticles(int numParticles, double boxSize);
    void updatePositions(double dt);
    void updateVelocities(const std::vector<std::array<double, 3>>& electricField,
                          const std::vector<std::array<double, 3>>& magneticField,
                          double dt);
    
    const std::vector<Particle>& getParticles() const { return particles; }
    size_t getNumParticles() const { return particles.size(); }

private:
    std::vector<Particle> particles;
    int gridNx, gridNy, gridNz;
    double gridDx, gridDy, gridDz;
    
    size_t getFieldIndex(double x, double y, double z) const;
};

#endif // PARTICLE_H
