#include "field.h"
#include "constants.h"
#include <cmath>
#include <omp.h>

Field::Field(int nx, int ny, int nz, double dx, double dy, double dz)
    : nx(nx), ny(ny), nz(nz), dx(dx), dy(dy), dz(dz) {
    
    int totalCells = nx * ny * nz;
    E.resize(totalCells, {0.0, 0.0, 0.0});
    B.resize(totalCells, {0.0, 0.0, 0.0});
    rho.resize(totalCells, 0.0);
    phi.resize(totalCells, 0.0);
}

Field::~Field() {}

void Field::initialize() {
    // Initialize with zero fields
    #pragma omp parallel for
    for (int idx = 0; idx < nx * ny * nz; ++idx) {
        E[idx] = {0.0, 0.0, 0.0};
        B[idx] = {0.0, 0.0, 0.0};
        rho[idx] = 0.0;
        phi[idx] = 0.0;
    }
}

void Field::updateElectricField(double dt) {
    // Simplified FDTD update for electric field
    // E = E + dt * (curl(B) / mu0 - J / epsilon0)
    
    #pragma omp parallel for collapse(3)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                int idx = index(i, j, k);
                int idx_xp = index(i + 1, j, k);
                int idx_yp = index(i, j + 1, k);
                int idx_zp = index(i, j, k + 1);
                
                // dBz/dy - dBy/dz
                double curlBx = (B[idx_yp][2] - B[idx][2]) / dy - 
                               (B[idx_zp][1] - B[idx][1]) / dz;
                // dBx/dz - dBz/dx
                double curlBy = (B[idx_zp][0] - B[idx][0]) / dz - 
                               (B[idx_xp][2] - B[idx][2]) / dx;
                // dBy/dx - dBx/dy
                double curlBz = (B[idx_xp][1] - B[idx][1]) / dx - 
                               (B[idx_yp][0] - B[idx][0]) / dy;
                
                E[idx][0] += dt * curlBx / constants::mu0;
                E[idx][1] += dt * curlBy / constants::mu0;
                E[idx][2] += dt * curlBz / constants::mu0;
            }
        }
    }
}

void Field::updateMagneticField(double dt) {
    // Simplified FDTD update for magnetic field
    // B = B - dt * curl(E)
    
    #pragma omp parallel for collapse(3)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                int idx = index(i, j, k);
                int idx_xm = index(i - 1, j, k);
                int idx_ym = index(i, j - 1, k);
                int idx_zm = index(i, j, k - 1);
                
                // dEz/dy - dEy/dz
                double curlEx = (E[idx][2] - E[idx_ym][2]) / dy - 
                               (E[idx][1] - E[idx_zm][1]) / dz;
                // dEx/dz - dEz/dx
                double curlEy = (E[idx][0] - E[idx_zm][0]) / dz - 
                               (E[idx][2] - E[idx_xm][2]) / dx;
                // dEy/dx - dEx/dy
                double curlEz = (E[idx][1] - E[idx_xm][1]) / dx - 
                               (E[idx][0] - E[idx_ym][0]) / dy;
                
                B[idx][0] -= dt * curlEx;
                B[idx][1] -= dt * curlEy;
                B[idx][2] -= dt * curlEz;
            }
        }
    }
}

void Field::depositCharge(const std::vector<std::array<double, 3>>& positions,
                          const std::vector<double>& charges) {
    // Reset charge density
    #pragma omp parallel for
    for (int idx = 0; idx < nx * ny * nz; ++idx) {
        rho[idx] = 0.0;
    }
    
    // Simple nearest-grid-point charge deposition
    for (size_t p = 0; p < positions.size(); ++p) {
        int i = static_cast<int>(positions[p][0] / dx) % nx;
        int j = static_cast<int>(positions[p][1] / dy) % ny;
        int k = static_cast<int>(positions[p][2] / dz) % nz;
        
        if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz) {
            #pragma omp atomic
            rho[index(i, j, k)] += charges[p] / (dx * dy * dz);
        }
    }
}
