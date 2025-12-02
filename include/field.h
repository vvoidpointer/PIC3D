#ifndef FIELD_H
#define FIELD_H

#include <vector>
#include <array>

class Field {
public:
    Field(int nx, int ny, int nz, double dx, double dy, double dz);
    ~Field();

    void initialize();
    void updateElectricField(double dt);
    void updateMagneticField(double dt);
    void depositCharge(const std::vector<std::array<double, 3>>& positions,
                       const std::vector<double>& charges);
    
    const std::vector<std::array<double, 3>>& getElectricField() const { return E; }
    const std::vector<std::array<double, 3>>& getMagneticField() const { return B; }
    const std::vector<double>& getChargeDensity() const { return rho; }

    int getNx() const { return nx; }
    int getNy() const { return ny; }
    int getNz() const { return nz; }

private:
    int nx, ny, nz;
    double dx, dy, dz;
    std::vector<std::array<double, 3>> E;  // Electric field
    std::vector<std::array<double, 3>> B;  // Magnetic field
    std::vector<double> rho;               // Charge density
    std::vector<double> phi;               // Electric potential

    int index(int i, int j, int k) const { return i + nx * (j + ny * k); }
};

#endif // FIELD_H
