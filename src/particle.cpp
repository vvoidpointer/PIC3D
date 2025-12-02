#include "particle.h"
#include "field.h"
#include "mpi_utils.h"
#include <cmath>
#include <cstdio>
#include <algorithm>

void shape_function_2(double x, double *Interpolation)
{
    // 参考Smilei的二阶形状函数实现
    // 计算到最近网格整数点的距离
    int i_center = (int)(x + 0.5);
    double delta = x - (double)i_center;

    // 二阶B样条形状函数（3点插值）
    Interpolation[0] = 0.5 * (0.5 - delta) * (0.5 - delta); // 左节点
    Interpolation[1] = 0.75 - delta * delta;                // 中心节点
    Interpolation[2] = 0.5 * (0.5 + delta) * (0.5 + delta); // 右节点
}

void shape_function_4(double x, double *Interpolation)
{
    int ii;
    double dx, xx, x_square;

    ii = (int)(x + 0.5);
    dx = x - (ii - 2);

    xx = dx - 2.5;
    x_square = xx * xx; /*W_i-2*/
    Interpolation[0] = x_square * x_square / 24.0;

    xx = dx - 1.0;
    x_square = xx * xx; /*W_i-1*/
    Interpolation[1] = -x_square * x_square / 6.0 + x_square * xx * alpha3 - x_square * alpha2 + xx * alpha1 + alpha0;

    xx = dx - 2.0;
    x_square = xx * xx; /*W_i*/
    Interpolation[2] = x_square * x_square / 4.0 - x_square * beta2 + beta0;

    xx = dx - 3.0;
    x_square = xx * xx; /*W_i+1*/
    Interpolation[3] = -x_square * x_square / 6.0 - x_square * xx * alpha3 - x_square * alpha2 - xx * alpha1 + alpha0;

    xx = dx - 1.5;
    x_square = xx * xx; /*W_i+2*/
    Interpolation[4] = x_square * x_square / 24.0;
}

void initialize_particles(std::vector<particle> &particles, int &num_particles,
                          double density_func(double, double, double),
                          double charge, double mass,
                          double initial_px, double initial_py, double initial_pz,
                          double initial_sx, double initial_sy, double initial_sz,
                          int my_rank)
{
    num_particles = local_xcells * local_ycells * local_zcells * cell_per_particles;
    particles.resize(num_particles);
    int num_x = local_xcells * std::cbrt(cell_per_particles);
    int num_y = local_ycells * std::cbrt(cell_per_particles);
    int num_z = local_zcells * std::cbrt(cell_per_particles);
    double norm_x = (xend - xst) / (num_x);
    double norm_y = (yend - yst) / (num_y);
    double norm_z = (zend - zst) / (num_z);

    int particle_index = 0; // 粒子索引计数器
    for(int k = 0; k < num_z; k++)
    {
        for (int j = 0; j < num_y; j++)
        {
            for (int i = 0; i < num_x; i++)
            {
                if (particle_index >= num_particles)
                    break; // 防止数组越界

                double x_pos = xst + (i)*norm_x + norm_x / 2.0; // 网格索引空间位置
                double y_pos = yst + (j)*norm_y + norm_y / 2.0;
                double z_pos = zst + (k)*norm_z + norm_z / 2.0;
                double temp_weight = density_func(x_pos, y_pos, z_pos) / cell_per_particles;
                if (temp_weight <= 0.0)
                {
                    continue; // 跳过权重为0的粒子
                }

                particles[particle_index].id = my_rank * num_particles + particle_index; // 给粒子分配唯一ID
                particles[particle_index].x = x_pos;                                     // 网格索引空间位置
                particles[particle_index].y = y_pos;                                     // 网格索引空间位置
                particles[particle_index].z = z_pos;
                particles[particle_index].px = initial_px; // 初始动量
                particles[particle_index].py = initial_py;
                particles[particle_index].pz = initial_pz;
                particles[particle_index].sx = initial_sx; // 初始化自旋
                particles[particle_index].sy = initial_sy;
                particles[particle_index].sz = initial_sz;
                particles[particle_index].gamma = sqrt(1 + (particles[particle_index].px * particles[particle_index].px +
                                                            particles[particle_index].py * particles[particle_index].py +
                                                            particles[particle_index].pz * particles[particle_index].pz)); // 初始伽马因子
                particles[particle_index].charge = charge;                                                                 // 电荷
                particles[particle_index].mass = mass;                                                                     // 质量
                particles[particle_index].weight = temp_weight;                                                              // 粒子权重
                particles[particle_index].rank = my_rank; // 记录粒子所在的MPI进程rank
                

                particle_index++;
            }
        }
    }
    //printf("Rank %d: Initialized %d particles.\n", my_rank, particle_index);
    particles.resize(particle_index);
    num_particles = particle_index; // 更新实际粒子数量
}

void update_particle_position_momentum(std::vector<particle> &particles, int num_particles, int my_rank, int kind)
{
    double ex_p, ey_p, ez_p, bx_p, by_p, bz_p; // 用于存储粒子电场和磁场

    // 更新粒子位置和动量
    for (int i = 0; i < num_particles; i++)
    {
        particles[i].rank = my_rank; // 记录粒子所在的MPI进程rank


        if (particles[i].x < xst || particles[i].x > xend || particles[i].y < yst || particles[i].y > yend || particles[i].z < zst || particles[i].z > zend)
        {
            if (rank_period[0] == 0 && rank_period[1] == 0 && rank_period[2] == 0)
            {
                printf("Error: Particle %d out of bounds! Particle position: (%.2f, %.2f, %.2f)\n",
                       kind, particles[i].x, particles[i].y, particles[i].z);
                // exit(1);
            }
            else
            {
                if (rank_period[0] == 1)
                {
                    if (particles[i].x < 0.0)
                    {
                        particles[i].x += global_nx;
                    }
                    if (particles[i].x > global_nx)
                    {
                        particles[i].x -= global_nx;
                    }
                }
                if (rank_period[1] == 1)
                {
                    if (particles[i].y < 0.0)
                    {
                        particles[i].y += global_ny;
                    }
                    if (particles[i].y > global_ny)
                    {
                        particles[i].y -= global_ny;
                    }
                }
                if (rank_period[2] == 1)
                {
                    if (particles[i].z < 0.0)
                    {
                        particles[i].z += global_nz;
                    }
                    if (particles[i].z > global_nz)
                    {
                        particles[i].z -= global_nz;
                    }
                }
            }
        }

        ex_p = ey_p = ez_p = 0.0;
        bx_p = by_p = bz_p = 0.0;

        // 获取粒子位置电场和磁场
        double x_cell = particles[i].x - nc_xst_new; // 在本进程 buffer 中的位置
        double y_cell = particles[i].y - nc_yst_new;
        double z_cell = particles[i].z - nc_zst_new;
        int i1, i_st, i_end, j1, j_st, j_end, k1, k_st, k_end;

        // 声明形状函数权重数组
        double Wx_p[5], Wy_p[5], Wz_p[5];
        double Wx_d[5], Wy_d[5], Wz_d[5];
        int i_p, j_p, k_p;
        int i_d, j_d, k_d;

        if (mem == 2)
        {
            shape_function_2(x_cell, Wx_p); i_p = (int)(x_cell + 0.5);
            shape_function_2(y_cell, Wy_p); j_p = (int)(y_cell + 0.5);
            shape_function_2(z_cell, Wz_p); k_p = (int)(z_cell + 0.5);

            shape_function_2(x_cell - 0.5, Wx_d); i_d = (int)(x_cell);
            shape_function_2(y_cell - 0.5, Wy_d); j_d = (int)(y_cell);
            shape_function_2(z_cell - 0.5, Wz_d); k_d = (int)(z_cell);
        }
        else if (mem == 4)
        {
            shape_function_4(x_cell, Wx_p); i_p = (int)(x_cell + 0.5);
            shape_function_4(y_cell, Wy_p); j_p = (int)(y_cell + 0.5);
            shape_function_4(z_cell, Wz_p); k_p = (int)(z_cell + 0.5);

            shape_function_4(x_cell - 0.5, Wx_d); i_d = (int)(x_cell);
            shape_function_4(y_cell - 0.5, Wy_d); j_d = (int)(y_cell);
            shape_function_4(z_cell - 0.5, Wz_d); k_d = (int)(z_cell);
        }

        int half_mem = mem / 2;

        // Ex: (i+1/2, j, k) -> Dual X, Primal Y, Primal Z
        for (int k = -half_mem; k <= half_mem; k++) {
            for (int j = -half_mem; j <= half_mem; j++) {
                for (int i = -half_mem; i <= half_mem; i++) {
                    ex_p += Wx_d[i + half_mem] * Wy_p[j + half_mem] * Wz_p[k + half_mem] * ex[k_p + k][j_p + j][i_d + i];
                }
            }
        }

        // Ey: (i, j+1/2, k) -> Primal X, Dual Y, Primal Z
        for (int k = -half_mem; k <= half_mem; k++) {
            for (int j = -half_mem; j <= half_mem; j++) {
                for (int i = -half_mem; i <= half_mem; i++) {
                    ey_p += Wx_p[i + half_mem] * Wy_d[j + half_mem] * Wz_p[k + half_mem] * ey[k_p + k][j_d + j][i_p + i];
                }
            }
        }

        // Ez: (i, j, k+1/2) -> Primal X, Primal Y, Dual Z
        for (int k = -half_mem; k <= half_mem; k++) {
            for (int j = -half_mem; j <= half_mem; j++) {
                for (int i = -half_mem; i <= half_mem; i++) {
                    ez_p += Wx_p[i + half_mem] * Wy_p[j + half_mem] * Wz_d[k + half_mem] * ez[k_d + k][j_p + j][i_p + i];
                }
            }
        }

        // Bx: (i, j+1/2, k+1/2) -> Primal X, Dual Y, Dual Z
        for (int k = -half_mem; k <= half_mem; k++) {
            for (int j = -half_mem; j <= half_mem; j++) {
                for (int i = -half_mem; i <= half_mem; i++) {
                    bx_p += Wx_p[i + half_mem] * Wy_d[j + half_mem] * Wz_d[k + half_mem] * bx[k_d + k][j_d + j][i_p + i];
                }
            }
        }

        // By: (i+1/2, j, k+1/2) -> Dual X, Primal Y, Dual Z
        for (int k = -half_mem; k <= half_mem; k++) {
            for (int j = -half_mem; j <= half_mem; j++) {
                for (int i = -half_mem; i <= half_mem; i++) {
                    by_p += Wx_d[i + half_mem] * Wy_p[j + half_mem] * Wz_d[k + half_mem] * by[k_d + k][j_p + j][i_d + i];
                }
            }
        }

        // Bz: (i+1/2, j+1/2, k) -> Dual X, Dual Y, Primal Z
        for (int k = -half_mem; k <= half_mem; k++) {
            for (int j = -half_mem; j <= half_mem; j++) {
                for (int i = -half_mem; i <= half_mem; i++) {
                    bz_p += Wx_d[i + half_mem] * Wy_d[j + half_mem] * Wz_p[k + half_mem] * bz[k_p + k][j_d + j][i_d + i];
                }
            }
        }


        // 保存旧的动量用于位置更新
        double px_old = particles[i].px;
        double py_old = particles[i].py;
        double pz_old = particles[i].pz;

        particles[i].x_old = particles[i].x; // 保存旧位置用于位置更新
        particles[i].y_old = particles[i].y;
        particles[i].z_old = particles[i].z;

        double q_over_mc = particles[i].charge / (particles[i].mass) * dt_half;
        ex_p *= q_over_mc;
        ey_p *= q_over_mc;
        ez_p *= q_over_mc;
        bx_p *= q_over_mc;
        by_p *= q_over_mc;
        bz_p *= q_over_mc;

        // Boris粒子推进器 - 第一步：半电场推进
        double px1 = px_old + ex_p;
        double py1 = py_old + ey_p;
        double pz1 = pz_old + ez_p;

        // 更新伽马因子
        double gm1 = sqrt(1.0 + (px1 * px1 + py1 * py1 + pz1 * pz1));

        double tx = bx_p / gm1;
        double ty = by_p / gm1;
        double tz = bz_p / gm1;
        double tt = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
        double sx = tx * tt;
        double sy = ty * tt;
        double sz = tz * tt;

        double px2 = px1 + (py1 * tz - pz1 * ty);
        double py2 = py1 + (pz1 * tx - px1 * tz);
        double pz2 = pz1 + (px1 * ty - py1 * tx);

        double px3 = px1 + (py2 * sz - pz2 * sy);
        double py3 = py1 + (pz2 * sx - px2 * sz);
        double pz3 = pz1 + (px2 * sy - py2 * sx);

        // 第二步：半电场推进
        particles[i].px = px3 + ex_p;
        particles[i].py = py3 + ey_p;
        particles[i].pz = pz3 + ez_p;

        // 更新最终伽马因子
        particles[i].gamma = sqrt(1.0 + (particles[i].px * particles[i].px +
                                         particles[i].py * particles[i].py +
                                         particles[i].pz * particles[i].pz));

        particles[i].x = particles[i].x_old + particles[i].px / particles[i].gamma * dt / dx;
        particles[i].y = particles[i].y_old + particles[i].py / particles[i].gamma * dt / dy;
        particles[i].z = particles[i].z_old + particles[i].pz / particles[i].gamma * dt / dz;
        currents_particle(particles[i], kind);

        // 计算电子自旋进动
        if (particles[i].mass == m_e)
        {
            ex_p /= q_over_mc;
            ey_p /= q_over_mc;
            ez_p /= q_over_mc;
            bx_p /= q_over_mc;
            by_p /= q_over_mc;
            bz_p /= q_over_mc;

            double gamma_0 = particles[i].gamma;
            double vx_0 = particles[i].px / gamma_0;
            double vy_0 = particles[i].py / gamma_0;
            double vz_0 = particles[i].pz / gamma_0;
            double px_0 = vx_0 * ex_p;
            double py_0 = vy_0 * ey_p;
            double pz_0 = vz_0 * ez_p;
            double fLx = (ex_p + vy_0 * bz_p - vz_0 * by_p);
            double fLy = (ey_p + vz_0 * bx_p - vx_0 * bz_p);
            double fLz = (ez_p + vx_0 * by_p - vy_0 * bx_p);
            double chi_0 = std::max(0.0, fLx * fLx + fLy * fLy + fLz * fLz - (px_0 + py_0 + pz_0) * (px_0 + py_0 + pz_0));
            chi_0 = sqrt(chi_0) / E_schwinger * gamma_0;

            double alpha = -(g_factor(chi_0) / 2.0 - 1.0) * (gamma_0) / (1.0 + gamma_0);
            double beta = (g_factor(chi_0) / 2.0 - 1.0) + 1.0 / gamma_0;
            double gamma = g_factor(chi_0) / 2.0 - gamma_0 / (1.0 + gamma_0);

            double VB = vx_0 * bx_p + vy_0 * by_p + vz_0 * bz_p;
            double omega_x = alpha * VB * vx_0 + beta * bx_p + gamma * (vy_0 * ez_p - vz_0 * ey_p);
            double omega_y = alpha * VB * vy_0 + beta * by_p + gamma * (vz_0 * ex_p - vx_0 * ez_p);
            double omega_z = alpha * VB * vz_0 + beta * bz_p + gamma * (vx_0 * ey_p - vy_0 * ex_p);

            // 修正：自旋进动频率需要乘以 q/m
            double q_m_dt2 = particles[i].charge / particles[i].mass * dt_half;
            tx = omega_x * q_m_dt2;
            ty = omega_y * q_m_dt2;
            tz = omega_z * q_m_dt2;
            tt = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
            
            double sx_r = particles[i].sx + (particles[i].sy * tz - particles[i].sz * ty);
            double sy_r = particles[i].sy + (particles[i].sz * tx - particles[i].sx * tz);
            double sz_r = particles[i].sz + (particles[i].sx * ty - particles[i].sy * tx);

            double ox = particles[i].sx + (sy_r * tz - sz_r * ty) * tt;
            double oy = particles[i].sy + (sz_r * tx - sx_r * tz) * tt;
            double oz = particles[i].sz + (sx_r * ty - sy_r * tx) * tt;

            particles[i].sx = ox;
            particles[i].sy = oy;
            particles[i].sz = oz;
        }
    }
}

void currents_particle(const particle &particles, int kind)
{
    // 使用Esirkepov算法计算三维电流密度
    // Grid definition based on standard Yee Lattice (matching user image):
    // Jx: (i+1/2, j, k)     -> X:Dual, Y:Primal, Z:Primal
    // Jy: (i, j+1/2, k)     -> X:Primal, Y:Dual, Z:Primal
    // Jz: (i, j, k+1/2)     -> X:Primal, Y:Primal, Z:Dual
    // Rho: (i, j, k)        -> All Primal
    
    double x1 = particles.x_old - nc_xst_new; // 旧位置（本进程buffer中的相对位置）
    double x2 = particles.x - nc_xst_new;     // 新位置（本进程buffer中的相对位置）
    double y1 = particles.y_old - nc_yst_new; // 旧位置Y方向
    double y2 = particles.y - nc_yst_new;     // 新位置Y方向
    double z1 = particles.z_old - nc_zst_new; // Z方向位置
    double z2 = particles.z - nc_zst_new;
    
    int i1, i2, j1, j2, k1, k2;
    int i_min, i_max, j_min, j_max, k_min, k_max;
    double Sx0[mem + 1], Sx1[mem + 1], Sy0[mem + 1], Sy1[mem + 1], Sz0[mem + 1], Sz1[mem + 1];

    // 计算旧位置和新位置的形状函数
    if (mem == 2)
    {
        shape_function_2(x1, Sx0); shape_function_2(x2, Sx1);
        shape_function_2(y1, Sy0); shape_function_2(y2, Sy1);
        shape_function_2(z1, Sz0); shape_function_2(z2, Sz1);

        i1 = (int)(x1 + 0.5); i2 = (int)(x2 + 0.5);
        j1 = (int)(y1 + 0.5); j2 = (int)(y2 + 0.5);
        k1 = (int)(z1 + 0.5); k2 = (int)(z2 + 0.5);

        i_min = std::min(i1, i2) - 1; i_max = std::max(i1, i2) + 1;
        j_min = std::min(j1, j2) - 1; j_max = std::max(j1, j2) + 1;
        k_min = std::min(k1, k2) - 1; k_max = std::max(k1, k2) + 1;
    }
    else if (mem == 4)
    {
        shape_function_4(x1, Sx0); shape_function_4(x2, Sx1);
        shape_function_4(y1, Sy0); shape_function_4(y2, Sy1);
        shape_function_4(z1, Sz0); shape_function_4(z2, Sz1);

        i1 = (int)(x1 + 0.5); i2 = (int)(x2 + 0.5);
        j1 = (int)(y1 + 0.5); j2 = (int)(y2 + 0.5);
        k1 = (int)(z1 + 0.5); k2 = (int)(z2 + 0.5);

        i_min = std::min(i1, i2) - 2; i_max = std::max(i1, i2) + 2;
        j_min = std::min(j1, j2) - 2; j_max = std::max(j1, j2) + 2;
        k_min = std::min(k1, k2) - 2; k_max = std::max(k1, k2) + 2;
    }

    double q_w = particles.charge * particles.weight;
    double c_x = q_w * dx / dt;
    double c_y = q_w * dy / dt;
    double c_z = q_w * dz / dt;

    // Jx
    for (int k = k_min; k <= k_max; k++) {
        double Sz0_k = (k >= k1 - mem/2 && k <= k1 + mem/2) ? Sz0[k - k1 + mem/2] : 0.0;
        double Sz1_k = (k >= k2 - mem/2 && k <= k2 + mem/2) ? Sz1[k - k2 + mem/2] : 0.0;
        double DSz = Sz1_k - Sz0_k;
        
        for (int j = j_min; j <= j_max; j++) {
            double Sy0_j = (j >= j1 - mem/2 && j <= j1 + mem/2) ? Sy0[j - j1 + mem/2] : 0.0;
            double Sy1_j = (j >= j2 - mem/2 && j <= j2 + mem/2) ? Sy1[j - j2 + mem/2] : 0.0;
            double DSy = Sy1_j - Sy0_j;

            // Esirkepov weight for Jx: S0(j)*S0(k) + 0.5*DS(j)*S0(k) + 0.5*S0(j)*DS(k) + (1/3)*DS(j)*DS(k)
            double W_yz = Sy0_j * Sz0_k + 0.5 * DSy * Sz0_k + 0.5 * Sy0_j * DSz + (1.0/3.0) * DSy * DSz;

            double Wx = 0.0;
            for (int i = i_min; i <= i_max; i++) {
                double Sx0_i = (i >= i1 - mem/2 && i <= i1 + mem/2) ? Sx0[i - i1 + mem/2] : 0.0;
                double Sx1_i = (i >= i2 - mem/2 && i <= i2 + mem/2) ? Sx1[i - i2 + mem/2] : 0.0;
                double DSx = Sx1_i - Sx0_i;
                Wx += DSx;
                jx[k][j][i] -= c_x * Wx * W_yz;
            }
        }
    }

    // Jy
    for (int k = k_min; k <= k_max; k++) {
        double Sz0_k = (k >= k1 - mem/2 && k <= k1 + mem/2) ? Sz0[k - k1 + mem/2] : 0.0;
        double Sz1_k = (k >= k2 - mem/2 && k <= k2 + mem/2) ? Sz1[k - k2 + mem/2] : 0.0;
        double DSz = Sz1_k - Sz0_k;

        for (int i = i_min; i <= i_max; i++) {
            double Sx0_i = (i >= i1 - mem/2 && i <= i1 + mem/2) ? Sx0[i - i1 + mem/2] : 0.0;
            double Sx1_i = (i >= i2 - mem/2 && i <= i2 + mem/2) ? Sx1[i - i2 + mem/2] : 0.0;
            double DSx = Sx1_i - Sx0_i;

            // Esirkepov weight for Jy: S0(i)*S0(k) + 0.5*DS(i)*S0(k) + 0.5*S0(i)*DS(k) + (1/3)*DS(i)*DS(k)
            double W_xz = Sx0_i * Sz0_k + 0.5 * DSx * Sz0_k + 0.5 * Sx0_i * DSz + (1.0/3.0) * DSx * DSz;

            double Wy = 0.0;
            for (int j = j_min; j <= j_max; j++) {
                double Sy0_j = (j >= j1 - mem/2 && j <= j1 + mem/2) ? Sy0[j - j1 + mem/2] : 0.0;
                double Sy1_j = (j >= j2 - mem/2 && j <= j2 + mem/2) ? Sy1[j - j2 + mem/2] : 0.0;
                double DSy = Sy1_j - Sy0_j;
                Wy += DSy;
                jy[k][j][i] -= c_y * Wy * W_xz;
            }
        }
    }

    // Jz
    for (int j = j_min; j <= j_max; j++) {
        double Sy0_j = (j >= j1 - mem/2 && j <= j1 + mem/2) ? Sy0[j - j1 + mem/2] : 0.0;
        double Sy1_j = (j >= j2 - mem/2 && j <= j2 + mem/2) ? Sy1[j - j2 + mem/2] : 0.0;
        double DSy = Sy1_j - Sy0_j;

        for (int i = i_min; i <= i_max; i++) {
            double Sx0_i = (i >= i1 - mem/2 && i <= i1 + mem/2) ? Sx0[i - i1 + mem/2] : 0.0;
            double Sx1_i = (i >= i2 - mem/2 && i <= i2 + mem/2) ? Sx1[i - i2 + mem/2] : 0.0;
            double DSx = Sx1_i - Sx0_i;

            // Esirkepov weight for Jz: S0(i)*S0(j) + 0.5*DS(i)*S0(j) + 0.5*S0(i)*DS(j) + (1/3)*DS(i)*DS(j)
            double W_xy = Sx0_i * Sy0_j + 0.5 * DSx * Sy0_j + 0.5 * Sx0_i * DSy + (1.0/3.0) * DSx * DSy;

            double Wz = 0.0;
            for (int k = k_min; k <= k_max; k++) {
                double Sz0_k = (k >= k1 - mem/2 && k <= k1 + mem/2) ? Sz0[k - k1 + mem/2] : 0.0;
                double Sz1_k = (k >= k2 - mem/2 && k <= k2 + mem/2) ? Sz1[k - k2 + mem/2] : 0.0;
                double DSz = Sz1_k - Sz0_k;
                Wz += DSz;
                jz[k][j][i] -= c_z * Wz * W_xy;
            }
        }
    }

    // Rho
    for (int k = k_min; k <= k_max; k++) {
        double Sz = (k >= k2 - mem/2 && k <= k2 + mem/2) ? Sz1[k - k2 + mem/2] : 0.0;
        if (Sz == 0.0) continue;
        for (int j = j_min; j <= j_max; j++) {
            double Sy = (j >= j2 - mem/2 && j <= j2 + mem/2) ? Sy1[j - j2 + mem/2] : 0.0;
            if (Sy == 0.0) continue;
            for (int i = i_min; i <= i_max; i++) {
                double Sx = (i >= i2 - mem/2 && i <= i2 + mem/2) ? Sx1[i - i2 + mem/2] : 0.0;
                rho_particle[kind][k][j][i] += q_w * Sx * Sy * Sz;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Apply a single pass binomial filter on currents
// ---------------------------------------------------------------------------------------------------------------------
void filter_field(double ***data)
{
    std::vector<double> buffer;

    // X-pass
    int nx = nc_xend - nc_xst + 1;
    buffer.resize(nx + 2); 
    for (int k = nc_zst; k <= nc_zend; k++) {
        for (int j = nc_yst; j <= nc_yend; j++) {
            // Copy to buffer (including ghost cells)
            for (int i = -1; i <= nx; i++) {
                buffer[i + 1] = data[k][j][nc_xst + i];
            }
            // Filter
            for (int i = 0; i < nx; i++) {
                data[k][j][nc_xst + i] = 0.25 * buffer[i] + 0.5 * buffer[i + 1] + 0.25 * buffer[i + 2];
            }
        }
    }

    // Y-pass
    int ny = nc_yend - nc_yst + 1;
    buffer.resize(ny + 2);
    for (int k = nc_zst; k <= nc_zend; k++) {
        for (int i = nc_xst; i <= nc_xend; i++) {
            for (int j = -1; j <= ny; j++) {
                buffer[j + 1] = data[k][nc_yst + j][i];
            }
            for (int j = 0; j < ny; j++) {
                data[k][nc_yst + j][i] = 0.25 * buffer[j] + 0.5 * buffer[j + 1] + 0.25 * buffer[j + 2];
            }
        }
    }

    // Z-pass
    int nz = nc_zend - nc_zst + 1;
    buffer.resize(nz + 2);
    for (int j = nc_yst; j <= nc_yend; j++) {
        for (int i = nc_xst; i <= nc_xend; i++) {
            for (int k = -1; k <= nz; k++) {
                buffer[k + 1] = data[nc_zst + k][j][i];
            }
            for (int k = 0; k < nz; k++) {
                data[nc_zst + k][j][i] = 0.25 * buffer[k] + 0.5 * buffer[k + 1] + 0.25 * buffer[k + 2];
            }
        }
    }
}

void binomial_filter_currents()
{
    filter_field(jx);
    filter_field(jy);
    filter_field(jz);
}

void clear_currents()
{
    for (int k = nc_zst_out - 1; k <= nc_zend_out + 1; k++)
    {
        for (int j = nc_yst_out - 1; j <= nc_yend_out + 1; j++)
        {
            for (int i = nc_xst_out - 1; i <= nc_xend_out + 1; i++)
            {
                jx[k][j][i] = 0.0;
                jy[k][j][i] = 0.0;
                jz[k][j][i] = 0.0;
                rho_particle[0][k][j][i] = 0.0;
                rho_particle[1][k][j][i] = 0.0;
            }
        }
    }
}

void exchange_currents(MPI_Comm my_comm)
{
    exchange_boundary_current_fixed(jx, my_comm);
    exchange_boundary_current_fixed(jy, my_comm);
    exchange_boundary_current_fixed(jz, my_comm);
    exchange_boundary_current_fixed(rho_particle[0], my_comm);
    exchange_boundary_current_fixed(rho_particle[1], my_comm);
}

double g_factor(double x)
{
    if (x < 0.005 && x >= 0.0)
        return 2.003219;
    else if (x >= 0.005 && x < 0.01)
    {
        int index = (int)((x - 0.005) / 0.0001);
        return g_data[index];
    }
    else if (x >= 0.01 && x < 1)
    {
        int index = (int)((x - 0.01) / 0.01) + 50;
        return g_data[index];
    }
    else if (x >= 1 && x < 10)
    {
        int index = (int)((x - 1.0) / 0.01) + 150;
        return g_data[index];
    }
    else if (x >= 10 && x < 20)
    {
        int index = (int)((x - 10.0) / 0.01) + 1050;
        return g_data[index];
    }
    else if (x >= 20 && x <= 100)
    {
        int index = (int)((x - 20.0) / 0.1) + 2050;
        return g_data[index];
    }
    else
    {
        printf("Error: x value out of g_data range! x = %.2f\n", x);
        exit(1);
    }
}