#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <math.h>
#include <stdbool.h> // for bool type
#include <mpi.h>     // for MPI types

// 全局参数
extern const double Pi;

// 物理常数 (归一化单位制)
extern const double c1;        // 光速 (归一化)
extern const double q_e;       // 电子电荷 (归一化)
extern const double m_e;       // 电子质量 (归一化)
extern const double omega;     // 角频率 作为其他单位的参考
extern const double epsilon_0; // 真空介电常数 (归一化单位制)

// 计算常数
extern const double l0; // 波长 (无量纲)
extern const double t0; // 周期 (无量纲)
extern const double Lx; // 模拟域长度 (无量纲)
extern const double Ly; // 模拟域长度 (无量纲)
extern const double Lz; // 模拟域长度 (无量纲)

// 网格和并行参数
extern const int resx;
extern const int resy;
extern const int resz;
extern const int rest;
extern const int rank_len[3];
extern const int rank_period[3];
extern int rank_reorder;
extern int rank_coords[3];
extern int rank_xminus, rank_xplus, rank_yminus, rank_yplus, rank_zminus, rank_zplus;
extern const int local_xcells;
extern const int local_ycells;
extern const int local_zcells;
extern const int global_nx;
extern const int global_ny;
extern const int global_nz;
extern const int cell_per_particles;
extern const double density;
extern const double dx;
extern const double dy;
extern const double dz;
extern const double dt;
extern const double dt_half;
extern const double simulation_time;
extern const int timesteps;
extern const int output_interval;
extern const int num_steps;

// 控制参数
extern bool is_periodic;
extern bool is_relativistic_poission;

// 边界余量
extern const int mem;

// MPI 自定义数据类型
extern MPI_Datatype type_field_Xdir, type_field_Ydir, type_current_Xdir, type_current_Ydir, type_current_Zdir, type_field_Zdir;

// 激光参数
extern const double laser_amplitude;
extern const double laser_pulse_duration;
extern const double laser_start_time;
extern const int laser_polarization;
extern const double waist;
extern const double xfocus;
extern const double phase;
extern const double yfocus;
extern const double zfocus;

// 网格坐标变量
extern int nc_xst, nc_xend;
extern int nc_xst_new, nc_xend_new;
extern int nc_xst_out, nc_xend_out;
extern double xst, xend;
extern int nc_yst, nc_yend;
extern int nc_yst_new, nc_yend_new;
extern int nc_yst_out, nc_yend_out;
extern double yst, yend;
extern int nc_zst, nc_zend;
extern int nc_zst_new, nc_zend_new;
extern int nc_zst_out, nc_zend_out;
extern double zst, zend;

extern double E_schwinger;
extern const double alpha0, alpha1, alpha2, alpha3,
                  beta0, beta2;
// 等离子体密度函数
double plasma_density(double x, double y, double z);
double plasma_density1(double x, double y, double z);
extern const int num_kind; // 三类粒子

// g.txt 数据
extern const int g_data_size;
extern double g_data[];
bool load_g_data();

#define TAG_FIELD_XMINUS 1000
#define TAG_FIELD_XPLUS 1001
#define TAG_FIELD_YMINUS 1002
#define TAG_FIELD_YPLUS 1003
#define TAG_FIELD_ZMINUS 1004
#define TAG_FIELD_ZPLUS 1005
#define TAG_CURRENT_XMINUS_ACCUMULATE 2000
#define TAG_CURRENT_XPLUS_ACCUMULATE 2001
#define TAG_CURRENT_YMINUS_ACCUMULATE 2002
#define TAG_CURRENT_YPLUS_ACCUMULATE 2003
#define TAG_CURRENT_ZMINUS_ACCUMULATE 2004
#define TAG_CURRENT_ZPLUS_ACCUMULATE 2005
#define TAG_CURRENT_XMINUS_DATA 2100
#define TAG_CURRENT_XPLUS_DATA 2101
#define TAG_CURRENT_YMINUS_DATA 2102
#define TAG_CURRENT_YPLUS_DATA 2103
#define TAG_CURRENT_ZMINUS_DATA 2104
#define TAG_CURRENT_ZPLUS_DATA 2105
#define TAG_PARTICLE_XMINUS_COUNT 3000
#define TAG_PARTICLE_XPLUS_COUNT 3001
#define TAG_PARTICLE_YMINUS_COUNT 3002
#define TAG_PARTICLE_YPLUS_COUNT 3003
#define TAG_PARTICLE_ZMINUS_COUNT 3004
#define TAG_PARTICLE_ZPLUS_COUNT 3005
#define TAG_PARTICLE_XMINUS_DATA 3100
#define TAG_PARTICLE_XPLUS_DATA 3101
#define TAG_PARTICLE_YMINUS_DATA 3102
#define TAG_PARTICLE_YPLUS_DATA 3103
#define TAG_PARTICLE_ZMINUS_DATA 3104
#define TAG_PARTICLE_ZPLUS_DATA 3105

#endif // CONSTANTS_H
