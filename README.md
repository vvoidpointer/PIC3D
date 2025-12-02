# PIC3D

A 3D Particle-in-Cell (PIC) simulation program for plasma physics.

## Features

- **C++17** - Modern C++ implementation
- **HDF5** - Efficient data storage and output
- **OpenMP** - Parallel computing support for multi-core systems
- **Python** - Analysis and visualization tools

## Dependencies

- CMake >= 3.10
- C++17 compatible compiler (GCC >= 7, Clang >= 5)
- HDF5 library with C++ bindings
- OpenMP support
- Python 3 with numpy, h5py, matplotlib (for analysis)

### Installing Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get install cmake g++ libhdf5-dev libhdf5-cpp-103 python3-numpy python3-h5py python3-matplotlib
```

#### macOS (Homebrew)
```bash
brew install cmake hdf5 libomp
pip3 install numpy h5py matplotlib
```

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Running

```bash
./pic3d
```

The simulation outputs data to `output.h5` in HDF5 format.

## Analysis

Python scripts for analysis are provided in the `python/` directory:

```bash
cd python
pip install -r requirements.txt
python analysis.py
```

## Project Structure

```
PIC3D/
├── CMakeLists.txt      # Build configuration
├── include/            # Header files
│   ├── constants.h     # Physical constants
│   ├── particle.h      # Particle management
│   ├── field.h         # Electromagnetic fields
│   ├── grid.h          # Computational grid
│   └── io.h            # HDF5 I/O
├── src/                # Source files
│   ├── main.cpp        # Main simulation loop
│   ├── particle.cpp    # Particle implementation
│   ├── field.cpp       # Field solver
│   ├── grid.cpp        # Grid implementation
│   └── io.cpp          # HDF5 output
├── python/             # Python analysis tools
│   ├── analysis.py     # Analysis script
│   └── requirements.txt
└── README.md
```

## License

MIT License
