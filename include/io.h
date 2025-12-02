#ifndef IO_H
#define IO_H

#include <string>
#include <vector>
#include <array>
#include "particle.h"
#include "field.h"

class HDF5Writer {
public:
    HDF5Writer(const std::string& filename);
    ~HDF5Writer();

    void writeParticles(const std::vector<Particle>& particles, int timestep);
    void writeField(const Field& field, int timestep);
    void writeMetadata(int nx, int ny, int nz, double dx, double dy, double dz, double dt);

private:
    std::string filename;
};

#endif // IO_H
