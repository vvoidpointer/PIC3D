#include "constants.h"
#include <stdio.h>
#include <cstdio>

// 全局参数定义
const double Pi = 3.1415926536;

// 物理常数 (归一化单位制)
const double c1 = 1.0;        // 光速 (归一化)
const double q_e = -1.0;      // 电子电荷 (归一化)
const double m_e = 1.0;       // 电子质量 (归一化)
const double omega = 1.0;     // 角频率 作为其他单位的参考
const double epsilon_0 = 1.0; // 真空介电常数 (归一化单位制)

const double l0 = 2.0 * Pi * c1 / omega; // 波长 (无量纲)
const double t0 = 2.0 * Pi / omega;      // 周期 (无量纲)
const double Lx = 20.0 * l0;             // 模拟域长度 (无量纲)
const double Ly = 20.0 * l0;             // 模拟域长度 (无量纲)
const double Lz = 20.0 * l0;             // 模拟域长度 (无量纲)
const int resx = 20;
const int resy = 20;
const int resz = 20;
const int rest = 40;
const int rank_len[3] = {5,5,5};
const int rank_period[3] = {0, 0, 0}; // 周期性边界条件
int rank_reorder = 1;              // 进程重排 (允许MPI重新排序进程)
int rank_coords[3] = {0, 0, 0};
int rank_xminus, rank_xplus, rank_yminus, rank_yplus, rank_zminus, rank_zplus;

const int local_xcells = Lx / l0 / rank_len[0] * resx;
const int local_ycells = Ly / l0 / rank_len[1] * resy;
const int local_zcells = Lz / l0 / rank_len[2] * resz;
const int global_nx = rank_len[0] * local_xcells;
const int global_ny = rank_len[1] * local_ycells;
const int global_nz = rank_len[2] * local_zcells;
const int cell_per_particles = 0;
const double density = 0.5;                    // 标准化密度
const double dx = l0 / resx;                       // 网格间距
const double dy = l0 / resy;                       // 网格间距
const double dz = l0 / resz;                       // 网格间距
const double dt = t0 / rest;                       // 更小的时间步长确保数值稳定性
const double dt_half = dt / 2.0;                   // 半时间步长
const double simulation_time = 15.0 * t0;          // 总模拟时间
const int timesteps = (int)(simulation_time / dt); // 总时间步数
const int output_interval = (int)(1 * t0 / dt);    // 更频繁的输出间隔
const int num_steps = 0;
bool is_periodic = false;              // 是否使用周期性边界条件
bool is_relativistic_poission = false; // 是否使用相对论泊松方程
const int num_kind = 2;                // 两类粒子

// 边界余量
const int mem = 4;
// 创建自定义数据类型
MPI_Datatype type_field_Xdir, type_field_Ydir, type_field_Zdir, type_current_Xdir, type_current_Ydir, type_current_Zdir;

// 激光参数
const double laser_amplitude = 1.0 * m_e * c1 * omega / (-q_e); // 激光电场振幅
const double laser_pulse_duration = 10.0 * t0;                  // 激光脉宽 (时间)
const double laser_start_time = 0.0;                            // 激光开始时间
const int laser_polarization = 1;                               // 1: S偏振, 2: P偏振, 3: 圆偏振
const double waist = 3.0 * l0;
const double xfocus = 10.0 * l0;
const double phase = 0.0;
const double yfocus = 10.0 * l0;
const double zfocus = 10.0 * l0;

// 全局变量
int nc_xst, nc_xend;
int nc_xst_new, nc_xend_new;
int nc_xst_out, nc_xend_out;
double xst, xend;

int nc_yst, nc_yend;
int nc_yst_new, nc_yend_new;
int nc_yst_out, nc_yend_out;
double yst, yend;

int nc_zst, nc_zend;
int nc_zst_new, nc_zend_new;
int nc_zst_out, nc_zend_out;
double zst, zend;

double E_schwinger = 4.1215e5;
const double alpha0 = 55.0 / 96.0, alpha1 = 5.0 / 24.0, alpha2 = 5.0 / 4.0, alpha3 = 5.0 / 6.0,
                  beta0 = 115.0 / 192.0, beta2 = 5.0 / 8.0;

double plasma_density(double x, double y, double z)
{
    double x_phys = x * dx; // 转换为物理坐标
    double y_phys = y * dy;
    double z_phys = z * dz;

    // 创建两个高斯分布的等离子体云
    double result = 0.0;

    if (x_phys > 5.00 * l0 && x_phys <= 15.00 * l0 && y_phys > 5.0 * l0 && y_phys <= 15.0 * l0 && z_phys > 5.0 * l0 && z_phys <= 15.0 * l0)
    {
        // result = density * exp(-pow((x_phys - 5.0 * l0) / (2.0 * l0), 2.0));
        result = density;
    }

    return result;
}

double plasma_density1(double x, double y, double z)
{
    return 0.0;
}

// g.txt 数据数组定义
const int g_data_size = 2849;
double g_data[g_data_size];

bool load_g_data()
{
    FILE *file = fopen("g.txt", "r");
    if (file == NULL)
    {
        printf("Error: Unable to open g.txt file\n");
        return false;
    }

    for (int i = 0; i < g_data_size; i++)
    {
        if (fscanf(file, "%lf", &g_data[i]) != 1)
        {
            printf("Error: Failed to read data at line %d\n", i + 1);
            fclose(file);
            return false;
        }
    }

    fclose(file);
    printf("Successfully loaded %d values from g.txt\n", g_data_size);
    return true;
}
