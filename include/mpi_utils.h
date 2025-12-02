#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include <mpi.h>
#include <vector>
#include "constants.h"
#include "field.h"

// 前向声明
struct particle;

// MPI通信相关函数声明
void exchange_boundary_data(double ***data, MPI_Comm my_comm);
void exchange_particle_data(std::vector<particle> &particles, int &num_particles, int my_rank, MPI_Comm my_comm);
void exchange_boundary_current(double ***data, MPI_Comm my_comm);
void exchange_boundary_current_fixed(double ***data, MPI_Comm my_comm);
#endif // MPI_UTILS_H
