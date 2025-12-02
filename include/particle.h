#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>
#include <string>
#include "constants.h"

// 粒子结构体
struct particle
{
    int id;
    double x;
    double y;
    double z;
    double x_old;
    double y_old;
    double z_old;
    double px;
    double py;
    double pz;
    double sx;
    double sy;
    double sz;
    double gamma;
    double charge;
    double mass;
    double weight;
    int rank; // 用于存储粒子所在的MPI进程rank
};

// 粒子相关函数声明
void initialize_particles(std::vector<particle> &particles, int &num_particles,
                          double density_func(double, double, double),
                          double charge, double mass,
                          double initial_px, double initial_py, double initial_pz,
                          double initial_sx, double initial_sy, double initial_sz,
                          int my_rank);

void update_particle_position_momentum(std::vector<particle> &particles, int num_particles, int my_rank, int kind);
void currents_particle(const particle &particles, int kind);
void binomial_filter_currents();
// 形状函数
void shape_function_2(double x, double *Interpolation);
void shape_function_4(double x, double *Interpolation);
void clear_currents();
void exchange_currents(MPI_Comm my_comm);
double g_factor(double x);

#endif // PARTICLE_H
