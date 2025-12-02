#include "grid.h"

Grid::Grid(int nx, int ny, int nz, double Lx, double Ly, double Lz)
    : nx(nx), ny(ny), nz(nz), Lx(Lx), Ly(Ly), Lz(Lz) {
    dx = Lx / nx;
    dy = Ly / ny;
    dz = Lz / nz;
}

Grid::~Grid() {}

std::array<int, 3> Grid::getCellIndex(double x, double y, double z) const {
    int i = static_cast<int>(x / dx);
    int j = static_cast<int>(y / dy);
    int k = static_cast<int>(z / dz);
    
    // Clamp to grid bounds
    i = (i < 0) ? 0 : ((i >= nx) ? nx - 1 : i);
    j = (j < 0) ? 0 : ((j >= ny) ? ny - 1 : j);
    k = (k < 0) ? 0 : ((k >= nz) ? nz - 1 : k);
    
    return {i, j, k};
}

std::array<double, 3> Grid::getCellCenter(int i, int j, int k) const {
    return {(i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz};
}
