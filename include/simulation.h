#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <hdf5.h>
#include <mpi.h>
#include "particle.h"

// 主要模拟函数声明
void initialize_simulation(int my_rank, int num_procs);
void initialize_domain(int my_rank);
void run_main_time_loop(std::vector<std::vector<particle>> &particles,
                        int *num_particles, int my_rank, int num_procs, hid_t plist_id, MPI_Comm my_comm);

// 矩阵内存管理函数声明
double **dmatrix(long nrl, long nrh, long ncl, long nch);
void free_dmatrix(double **m, long nrl, long ncl);
void free_dmatrix(double **m, long nrl);
double ***dtensor3(long nzl, long nzh, long nyl, long nyh, long nxl, long nxh);
double ****dtensor4(long n4l, long n4h, long n3l, long n3h, long n2l, long n2h, long n1l, long n1h);
void free_dtensor3(double ***m, long nzl, long nyl, long nxl);
void free_dtensor4(double ****m, long n4l, long n3l, long n2l, long n1l);
void free_memory();

#endif // SIMULATION_H
