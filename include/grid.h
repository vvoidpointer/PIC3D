#ifndef GRID_H
#define GRID_H

#include <vector>
#include <array>

class Grid {
public:
    Grid(int nx, int ny, int nz, double Lx, double Ly, double Lz);
    ~Grid();

    double getDx() const { return dx; }
    double getDy() const { return dy; }
    double getDz() const { return dz; }
    
    int getNx() const { return nx; }
    int getNy() const { return ny; }
    int getNz() const { return nz; }
    
    double getLx() const { return Lx; }
    double getLy() const { return Ly; }
    double getLz() const { return Lz; }

    std::array<int, 3> getCellIndex(double x, double y, double z) const;
    std::array<double, 3> getCellCenter(int i, int j, int k) const;

private:
    int nx, ny, nz;
    double Lx, Ly, Lz;
    double dx, dy, dz;
};

#endif // GRID_H
