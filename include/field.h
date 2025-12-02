#ifndef FIELD_H
#define FIELD_H
#include <mpi.h>
#include <vector>
#include "constants.h"

// 前向声明
struct particle;

// 场变量声明
extern double ***rho_local;     // 电荷密度
extern double ****rho_particle; // 三类粒子的电荷密度 (大小固定为3，必须匹配num_kind)
extern double ***ex;
extern double ***phi;
extern double ***phi_new;
extern double ***ey;
extern double ***ez;
extern double ***bx;
extern double ***by;
extern double ***bz;
extern double ***jx;
extern double ***jy;
extern double ***jz;
extern double ***current_recv_xminus, ***current_recv_xplus, ***current_recv_yminus, ***current_recv_yplus, ***current_recv_zminus, ***current_recv_zplus;
extern double max_error;

// 场相关函数声明

void update_electromagnetic_fields(int my_rank, int num_procs, MPI_Comm my_comm);
void update_electromagnetic_fields_first(int my_rank, int num_procs, MPI_Comm my_comm);
void update_electromagnetic_fields_second(int my_rank, int num_procs, MPI_Comm my_comm);

#endif // FIELD_H
