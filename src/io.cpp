#include "io.h"
#include <H5Cpp.h>
#include <sstream>
#include <iomanip>

HDF5Writer::HDF5Writer(const std::string& filename) : filename(filename) {
    // Create or truncate the HDF5 file
    H5::H5File file(filename, H5F_ACC_TRUNC);
    file.close();
}

HDF5Writer::~HDF5Writer() {}

void HDF5Writer::writeParticles(const std::vector<Particle>& particles, int timestep) {
    H5::H5File file(filename, H5F_ACC_RDWR);
    
    std::stringstream ss;
    ss << "particles_" << std::setw(6) << std::setfill('0') << timestep;
    std::string groupName = ss.str();
    
    H5::Group group = file.createGroup(groupName);
    
    size_t numParticles = particles.size();
    
    // Write positions
    std::vector<double> posData(numParticles * 3);
    for (size_t i = 0; i < numParticles; ++i) {
        posData[i * 3 + 0] = particles[i].position[0];
        posData[i * 3 + 1] = particles[i].position[1];
        posData[i * 3 + 2] = particles[i].position[2];
    }
    
    hsize_t posDims[2] = {numParticles, 3};
    H5::DataSpace posSpace(2, posDims);
    H5::DataSet posDataset = group.createDataSet("position", H5::PredType::NATIVE_DOUBLE, posSpace);
    posDataset.write(posData.data(), H5::PredType::NATIVE_DOUBLE);
    
    // Write velocities
    std::vector<double> velData(numParticles * 3);
    for (size_t i = 0; i < numParticles; ++i) {
        velData[i * 3 + 0] = particles[i].velocity[0];
        velData[i * 3 + 1] = particles[i].velocity[1];
        velData[i * 3 + 2] = particles[i].velocity[2];
    }
    
    hsize_t velDims[2] = {numParticles, 3};
    H5::DataSpace velSpace(2, velDims);
    H5::DataSet velDataset = group.createDataSet("velocity", H5::PredType::NATIVE_DOUBLE, velSpace);
    velDataset.write(velData.data(), H5::PredType::NATIVE_DOUBLE);
    
    group.close();
    file.close();
}

void HDF5Writer::writeField(const Field& field, int timestep) {
    H5::H5File file(filename, H5F_ACC_RDWR);
    
    std::stringstream ss;
    ss << "fields_" << std::setw(6) << std::setfill('0') << timestep;
    std::string groupName = ss.str();
    
    H5::Group group = file.createGroup(groupName);
    
    int nx = field.getNx();
    int ny = field.getNy();
    int nz = field.getNz();
    size_t totalCells = static_cast<size_t>(nx) * ny * nz;
    
    // Write electric field
    const auto& E = field.getElectricField();
    std::vector<double> eData(totalCells * 3);
    for (size_t i = 0; i < totalCells; ++i) {
        eData[i * 3 + 0] = E[i][0];
        eData[i * 3 + 1] = E[i][1];
        eData[i * 3 + 2] = E[i][2];
    }
    
    hsize_t eDims[4] = {static_cast<hsize_t>(nx), static_cast<hsize_t>(ny), 
                        static_cast<hsize_t>(nz), 3};
    H5::DataSpace eSpace(4, eDims);
    H5::DataSet eDataset = group.createDataSet("electric_field", H5::PredType::NATIVE_DOUBLE, eSpace);
    eDataset.write(eData.data(), H5::PredType::NATIVE_DOUBLE);
    
    // Write magnetic field
    const auto& B = field.getMagneticField();
    std::vector<double> bData(totalCells * 3);
    for (size_t i = 0; i < totalCells; ++i) {
        bData[i * 3 + 0] = B[i][0];
        bData[i * 3 + 1] = B[i][1];
        bData[i * 3 + 2] = B[i][2];
    }
    
    hsize_t bDims[4] = {static_cast<hsize_t>(nx), static_cast<hsize_t>(ny), 
                        static_cast<hsize_t>(nz), 3};
    H5::DataSpace bSpace(4, bDims);
    H5::DataSet bDataset = group.createDataSet("magnetic_field", H5::PredType::NATIVE_DOUBLE, bSpace);
    bDataset.write(bData.data(), H5::PredType::NATIVE_DOUBLE);
    
    // Write charge density
    const auto& rho = field.getChargeDensity();
    
    hsize_t rhoDims[3] = {static_cast<hsize_t>(nx), static_cast<hsize_t>(ny), 
                          static_cast<hsize_t>(nz)};
    H5::DataSpace rhoSpace(3, rhoDims);
    H5::DataSet rhoDataset = group.createDataSet("charge_density", H5::PredType::NATIVE_DOUBLE, rhoSpace);
    rhoDataset.write(rho.data(), H5::PredType::NATIVE_DOUBLE);
    
    group.close();
    file.close();
}

void HDF5Writer::writeMetadata(int nx, int ny, int nz, double dx, double dy, double dz, double dt) {
    H5::H5File file(filename, H5F_ACC_RDWR);
    
    H5::Group group = file.createGroup("metadata");
    
    hsize_t dims[1] = {1};
    H5::DataSpace scalarSpace(1, dims);
    
    // Write grid dimensions
    H5::DataSet nxDataset = group.createDataSet("nx", H5::PredType::NATIVE_INT, scalarSpace);
    nxDataset.write(&nx, H5::PredType::NATIVE_INT);
    
    H5::DataSet nyDataset = group.createDataSet("ny", H5::PredType::NATIVE_INT, scalarSpace);
    nyDataset.write(&ny, H5::PredType::NATIVE_INT);
    
    H5::DataSet nzDataset = group.createDataSet("nz", H5::PredType::NATIVE_INT, scalarSpace);
    nzDataset.write(&nz, H5::PredType::NATIVE_INT);
    
    // Write grid spacing
    H5::DataSet dxDataset = group.createDataSet("dx", H5::PredType::NATIVE_DOUBLE, scalarSpace);
    dxDataset.write(&dx, H5::PredType::NATIVE_DOUBLE);
    
    H5::DataSet dyDataset = group.createDataSet("dy", H5::PredType::NATIVE_DOUBLE, scalarSpace);
    dyDataset.write(&dy, H5::PredType::NATIVE_DOUBLE);
    
    H5::DataSet dzDataset = group.createDataSet("dz", H5::PredType::NATIVE_DOUBLE, scalarSpace);
    dzDataset.write(&dz, H5::PredType::NATIVE_DOUBLE);
    
    // Write time step
    H5::DataSet dtDataset = group.createDataSet("dt", H5::PredType::NATIVE_DOUBLE, scalarSpace);
    dtDataset.write(&dt, H5::PredType::NATIVE_DOUBLE);
    
    group.close();
    file.close();
}
