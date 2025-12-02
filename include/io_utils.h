#ifndef IO_UTILS_H
#define IO_UTILS_H

#include <hdf5.h>
#include <vector>
#include <mpi.h>
#include "particle.h"

// 修改宏名称避免与HDF5内部冲突
#define H5_SAFE_CALL(status)                                                    \
    if (status < 0)                                                             \
    {                                                                           \
        fprintf(stderr, "Rank %d: HDF5 error at line %d\n", my_rank, __LINE__); \
        MPI_Abort(MPI_COMM_WORLD, 1);                                           \
    }

// IO相关函数声明
void output_data_hdf5(int step, int my_rank, int num_procs, MPI_Comm my_comm, hid_t plist_id);

// 粒子并行输出函数声明
void output_particles_hdf5(int step, int my_rank, int num_procs, MPI_Comm my_comm,
                           const std::vector<std::vector<particle>> &particles,
                           const int *num_particles);

#endif // IO_UTILS_H
