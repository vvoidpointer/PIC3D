#include "simulation.h"
#include "constants.h"
#include "field.h"
#include "mpi_utils.h"
#include "boundary.h"
#include "io_utils.h"
#include <cstring>
#include <cstdio>
#include <stdexcept> // 用于异常处理

void initialize_simulation(int my_rank, int num_procs)
{
    if (my_rank == 0)
    {
        printf("Starting PIC simulation:\n");
        printf("  Total timesteps: %d\n", timesteps);
        printf("  Output interval: every %d steps\n", output_interval);
        printf("  Domain: Lx=%.2f, dx=%.6f, dt=%.6f\n", Lx, dx, dt);
        printf("  Expected output files: %d\n", (timesteps / output_interval) + 1);
    }
}

void initialize_domain(int my_rank)
{
    // 基于笛卡尔坐标系的正确域分解
    int coord_x = rank_coords[0]; // X方向的坐标 (0, 1, 2)
    int coord_y = rank_coords[1]; // Y方向的坐标 (0, 1, 2)
    int coord_z = rank_coords[2]; // Z方向的坐标 (0, 1, 2)

    // X方向域分解：基于笛卡尔坐标
    nc_xst = coord_x * local_xcells + 1;
    nc_xend = nc_xst + local_xcells - 1;
    xst = (double)(nc_xst - 1); // 网格索引空间位置
    xend = (double)(nc_xend);

    nc_xst_new = nc_xst - mem;
    nc_xst = nc_xst - nc_xst_new;
    nc_xend = nc_xend - nc_xst_new;
    nc_xst_out = nc_xst - mem;
    nc_xend_out = nc_xend + mem;

    // Y方向域分解：基于笛卡尔坐标
    nc_yst = coord_y * local_ycells + 1;
    nc_yend = nc_yst + local_ycells - 1;
    yst = (double)(nc_yst - 1); // 网格索引空间位置
    yend = (double)(nc_yend);

    nc_yst_new = nc_yst - mem;
    nc_yst = nc_yst - nc_yst_new;
    nc_yend = nc_yend - nc_yst_new;
    nc_yst_out = nc_yst - mem;
    nc_yend_out = nc_yend + mem;

    // Z方向域分解：基于笛卡尔坐标
    nc_zst = coord_z * local_zcells + 1;
    nc_zend = nc_zst + local_zcells - 1;
    zst = (double)(nc_zst - 1); // 网格索引
    zend = (double)(nc_zend);
    nc_zst_new = nc_zst - mem;
    nc_zst = nc_zst - nc_zst_new;
    nc_zend = nc_zend - nc_zst_new;
    nc_zst_out = nc_zst - mem;
    nc_zend_out = nc_zend + mem;

    // 计算当前进程的域边界（网格索引空间）
    double domain_left = nc_xst;   // 本进程的左边界（网格索引）
    double domain_right = nc_xend; // 本进程的右边界（网格索引）
    double domain_bottom = nc_zst; // 本进程的下边界（网格索引）
    double domain_top = nc_zend;   // 本进程的上边界（网格索引）
    double domain_front = nc_yst;  // 本进程的前边界（网格索引）
    double domain_back = nc_yend;  // 本进程的后边界（网格索引）

    // 每个进程输出其域信息和坐标
    //printf("Rank %d (coord_x=%d, coord_y=%d): domain_left=%.2f, domain_right=%.2f, domain_bottom=%.2f, domain_top=%.2f\n",
    //       my_rank, coord_x, coord_y, domain_left, domain_right, domain_bottom, domain_top);
    //printf("Rank %d: xst=%.2f, xend=%.2f, yst=%.2f, yend=%.2f\n",
    //     my_rank, xst, xend, yst, yend);

    int ncy_exchg_field = nc_yend_out - nc_yst_out + 1;
    int ncx_exchg_field = nc_xend_out - nc_xst_out + 1;
    int ncz_exchg_field = nc_zend_out - nc_zst_out + 1;

    // 为场数据交换定义MPI派生数据类型

    // 1. Z方向交换 (交换XY平面)
    // 发送 mem 个连续的XY平面
    MPI_Type_contiguous(mem * ncx_exchg_field * ncy_exchg_field, MPI_DOUBLE, &type_field_Zdir);
    MPI_Type_commit(&type_field_Zdir);

    // 2. Y方向交换 (交换XZ切片)
    // 需要发送 ncz_exchg_field 个 [mem * ncx_exchg_field] 的数据块
    // 块间距为一个XY平面的大小
    MPI_Datatype type_temp_y;
    MPI_Type_vector(ncz_exchg_field, mem * ncx_exchg_field, ncy_exchg_field * ncx_exchg_field, MPI_DOUBLE, &type_temp_y);
    MPI_Type_create_resized(type_temp_y, 0, sizeof(double), &type_field_Ydir);
    MPI_Type_commit(&type_field_Ydir);
    MPI_Type_free(&type_temp_y);

    // 3. X方向交换 (交换YZ切片)
    // 需要发送 ncz_exchg_field * ncy_exchg_field 个长度为 mem 的小数据块
    // 块间距为 ncx_exchg_field
    MPI_Datatype type_temp_x;
    MPI_Type_vector(ncz_exchg_field * ncy_exchg_field, mem, ncx_exchg_field, MPI_DOUBLE, &type_temp_x);
    MPI_Type_create_resized(type_temp_x, 0, sizeof(double), &type_field_Xdir);
    MPI_Type_commit(&type_field_Xdir);
    MPI_Type_free(&type_temp_x);

    int ncx_exchg_current = ncx_exchg_field + 2;
    int ncy_exchg_current = ncy_exchg_field + 2;
    int ncz_exchg_current = ncz_exchg_field + 2;

    // 为电流数据交换定义MPI派生数据类型 (逻辑同上)

    // 1. Z方向交换
    MPI_Type_contiguous((mem + 1) * ncx_exchg_current * ncy_exchg_current, MPI_DOUBLE, &type_current_Zdir);
    MPI_Type_commit(&type_current_Zdir);

    // 2. Y方向交换
    MPI_Datatype type_temp_curr_y;
    MPI_Type_vector(ncz_exchg_current, (mem + 1) * ncx_exchg_current, ncy_exchg_current * ncx_exchg_current, MPI_DOUBLE, &type_temp_curr_y);
    MPI_Type_create_resized(type_temp_curr_y, 0, sizeof(double), &type_current_Ydir);
    MPI_Type_commit(&type_current_Ydir);
    MPI_Type_free(&type_temp_curr_y);

    // 3. X方向交换
    MPI_Datatype type_temp_curr_x;
    MPI_Type_vector(ncz_exchg_current * ncy_exchg_current, mem + 1, ncx_exchg_current, MPI_DOUBLE, &type_temp_curr_x);
    MPI_Type_create_resized(type_temp_curr_x, 0, sizeof(double), &type_current_Xdir);
    MPI_Type_commit(&type_current_Xdir);
    MPI_Type_free(&type_temp_curr_x);

    ex = dtensor3(nc_zst_out, nc_zend_out, nc_yst_out, nc_yend_out, nc_xst_out, nc_xend_out);
    ey = dtensor3(nc_zst_out, nc_zend_out, nc_yst_out, nc_yend_out, nc_xst_out, nc_xend_out);
    ez = dtensor3(nc_zst_out, nc_zend_out, nc_yst_out, nc_yend_out, nc_xst_out, nc_xend_out);
    bx = dtensor3(nc_zst_out, nc_zend_out, nc_yst_out, nc_yend_out, nc_xst_out, nc_xend_out);
    by = dtensor3(nc_zst_out, nc_zend_out, nc_yst_out, nc_yend_out, nc_xst_out, nc_xend_out);
    bz = dtensor3(nc_zst_out, nc_zend_out, nc_yst_out, nc_yend_out, nc_xst_out, nc_xend_out);

    jx = dtensor3(nc_zst_out - 1, nc_zend_out + 1, nc_yst_out - 1, nc_yend_out + 1, nc_xst_out - 1, nc_xend_out + 1);
    jy = dtensor3(nc_zst_out - 1, nc_zend_out + 1, nc_yst_out - 1, nc_yend_out + 1, nc_xst_out - 1, nc_xend_out + 1);
    jz = dtensor3(nc_zst_out - 1, nc_zend_out + 1, nc_yst_out - 1, nc_yend_out + 1, nc_xst_out - 1, nc_xend_out + 1);

    rho_local = dtensor3(nc_zst_out - 1, nc_zend_out + 1, nc_yst_out - 1, nc_yend_out + 1, nc_xst_out - 1, nc_xend_out + 1);
    rho_particle = dtensor4(0, num_kind - 1, nc_zst_out - 1, nc_zend_out + 1, nc_yst_out - 1, nc_yend_out + 1, nc_xst_out - 1, nc_xend_out + 1);

    current_recv_xminus = dtensor3(nc_zst_out - 1, nc_zend_out + 1, nc_yst_out - 1, nc_yend_out + 1, nc_xst_out - 1, nc_xend_out + 1);
    current_recv_xplus = dtensor3(nc_zst_out - 1, nc_zend_out + 1, nc_yst_out - 1, nc_yend_out + 1, nc_xst_out - 1, nc_xend_out + 1);
    current_recv_yminus = dtensor3(nc_zst_out - 1, nc_zend_out + 1, nc_yst_out - 1, nc_yend_out + 1, nc_xst_out - 1, nc_xend_out + 1);
    current_recv_yplus = dtensor3(nc_zst_out - 1, nc_zend_out + 1, nc_yst_out - 1, nc_yend_out + 1, nc_xst_out - 1, nc_xend_out + 1);
    current_recv_zminus = dtensor3(nc_zst_out - 1, nc_zend_out + 1, nc_yst_out - 1, nc_yend_out + 1, nc_xst_out - 1, nc_xend_out + 1);
    current_recv_zplus = dtensor3(nc_zst_out - 1, nc_zend_out + 1, nc_yst_out - 1, nc_yend_out + 1, nc_xst_out - 1, nc_xend_out + 1);
}

void run_main_time_loop(std::vector<std::vector<particle>> &particles,
                        int *num_particles, int my_rank, int num_procs, hid_t plist_id, MPI_Comm my_comm)
{
    // 主时间循环
    int step = 0;
    while (step < timesteps)
    {

        MPI_Barrier(MPI_COMM_WORLD); // 确保所有进程同步
        step++;                      // 移到循环开始，这样步数从1开始

        // 更新电场后施加激光边界条件
        radiating_boundary(my_rank, 1, 0, num_procs, step);
        // 更新电磁场 - 第一部分
        update_electromagnetic_fields_first(my_rank, num_procs, my_comm);

        // 清除电流
        clear_currents();
        // 粒子运动
        update_particle_position_momentum(particles[0], num_particles[0], my_rank, 0);
        update_particle_position_momentum(particles[1], num_particles[1], my_rank, 1);
        //  交换粒子数据
        exchange_particle_data(particles[0], num_particles[0], my_rank, my_comm);
        exchange_particle_data(particles[1], num_particles[1], my_rank, my_comm);

        exchange_currents(my_comm);

        // 更新电磁场 - 第二部分
        update_electromagnetic_fields_second(my_rank, num_procs, my_comm);

        // 定期输出数据
        if (step % output_interval == 0)
        {
            if (my_rank == 5)
            {
                printf("Rank %d: Outputting data for step %d\n", my_rank, step);
            }
            // 输出场数据
            output_data_hdf5(step, my_rank, num_procs, my_comm, plist_id);
            // 输出粒子数据
            output_particles_hdf5(step, my_rank, num_procs, my_comm, particles, num_particles);
        }
    }

    // 模拟完成信息
    if (my_rank == 0)
    {
        printf("PIC simulation completed! Total %d timesteps executed.\n", timesteps);
        printf("Final output files: output/output_step_0000.h5 to output/output_step_%04d.h5\n",
               (timesteps / output_interval) * output_interval);
    }
}

// 定义预留空间大小（防止越界访问）
const int NR_END = 0;

/*
 * 分配一个支持自定义下标范围的二维double矩阵
 * @param nrl 行下标下界
 * @param nrh 行下标上界
 * @param ncl 列下标下界
 * @param nch 列下标上界
 * @return 指向矩阵的指针，可使用m[i][j]访问（i在[nrl..nrh]，j在[ncl..nch]）
 */
double **dmatrix(long nrl, long nrh, long ncl, long nch)
{
    long i, nrow = nrh - nrl + 1; // 实际行数
    long ncol = nch - ncl + 1;    // 实际列数
    double **m;

    try
    {
        // 1. 分配行指针数组（额外预留NR_END空间）
        m = new double *[nrow + NR_END];
        if (!m)
        {
            throw std::bad_alloc();
        }

        // 调整指针，使m[nrl]成为第一个行指针
        m -= nrl;

        // 2. 分配整个矩阵的数据块（连续内存）
        m[nrl] = new double[nrow * ncol + NR_END];
        if (!m[nrl])
        {
            throw std::bad_alloc();
        }

        // 调整列指针：跳过预留空间，并使m[nrl][ncl]成为第一个元素
        m[nrl] += NR_END;
        m[nrl] -= ncl;

        // 3. 设置各行的指针（每行起始地址 = 上一行起始 + 列数）
        for (i = nrl + 1; i <= nrh; ++i)
        {
            m[i] = m[i - 1] + ncol;
        }
    }
    catch (const std::bad_alloc &e)
    {
        // 内存分配失败时抛出异常（包含错误位置信息）
        throw std::runtime_error("内存分配失败 in dmatrix(): " + std::string(e.what()));
    }

    return m;
}

/*
 * 释放由dmatrix分配的二维矩阵
 * @param m 矩阵指针
 * @param nrl 行下标下界（必须与分配时的参数一致）
 * @param ncl 列下标下界（必须与分配时的参数一致）
 */
void free_dmatrix(double **m, long nrl, long ncl)
{
    if (m == nullptr)
        return;

    // 释放数据块（先恢复到实际分配的起始地址）
    if (m[nrl] != nullptr)
    {
        m[nrl] += ncl;    // 恢复列指针的偏移
        m[nrl] -= NR_END; // 恢复NR_END偏移
        delete[] m[nrl];
    }

    // 释放行指针数组（恢复指针位置）
    m += nrl;
    delete[] m;
}

void free_dmatrix(double **m, long nrl)
{
    if (m == nullptr)
        return;

    // 释放数据块（假设ncl=0的默认情况）
    if (m[nrl] != nullptr)
    {
        m[nrl] -= NR_END; // 恢复NR_END偏移
        delete[] m[nrl];
    }

    // 释放行指针数组（恢复指针位置）
    m += nrl;
    delete[] m;
}

double ***dtensor3(long nzl, long nzh, long nyl, long nyh, long nxl, long nxh)
{
    long nx = nxh - nxl + 1; // x维度大小
    long ny = nyh - nyl + 1; // y维度大小
    long nz = nzh - nzl + 1; // z维度大小
    double ***m;

    try
    {
        // 1. 分配z方向指针数组
        m = new double **[nz + NR_END];
        if (!m)
            throw std::bad_alloc();

        // 调整z指针，使m[nzl]为第一个z指针
        m += NR_END;
        m -= nzl;

        // 2. 分配y方向指针数组（所有z共享一个连续块）
        m[nzl] = new double *[nz * ny + NR_END];
        if (!m[nzl])
            throw std::bad_alloc();

        // 调整y指针，使m[nzl][nyl]为第一个y指针
        m[nzl] += NR_END;
        m[nzl] -= nyl;

        // 3. 设置各z对应的y指针起始地址
        for (long i = nzl + 1; i <= nzh; ++i)
        {
            m[i] = m[i - 1] + ny;
        }

        // 4. 分配x方向数据块（连续内存）
        m[nzl][nyl] = new double[nz * ny * nx + NR_END];
        if (!m[nzl][nyl])
            throw std::bad_alloc();

        // 调整x指针，使m[nzl][nyl][nxl]为第一个x元素
        m[nzl][nyl] += NR_END;
        m[nzl][nyl] -= nxl;

        // 5. 设置所有(y,z)对应的x数据起始地址
        for (long i = nzl; i <= nzh; ++i)
        {
            for (long j = nyl; j <= nyh; ++j)
            {
                m[i][j] = m[nzl][nyl] + ((i - nzl) * ny + (j - nyl)) * nx;
            }
        }
    }
    catch (const std::bad_alloc &e)
    {
        throw std::runtime_error("内存分配失败 in dtensor3(): " + std::string(e.what()));
    }

    return m;
}

double ****dtensor4(long n4l, long n4h, long n3l, long n3h, long n2l, long n2h, long n1l, long n1h)
{
    long n1 = n1h - n1l + 1;
    long n2 = n2h - n2l + 1;
    long n3 = n3h - n3l + 1;
    long n4 = n4h - n4l + 1;
    double ****m;

    try
    {
        // 1. 分配第四维指针数组
        m = new double ***[n4 + NR_END];
        m += NR_END;
        m -= n4l;

        // 2. 分配第三维指针数组
        m[n4l] = new double **[n4 * n3 + NR_END];
        m[n4l] += NR_END;
        m[n4l] -= n3l;

        // 3. 设置第四维的各个指针
        for (long i = n4l + 1; i <= n4h; i++)
        {
            m[i] = m[i - 1] + n3;
        }

        // 4. 分配第二维指针数组
        m[n4l][n3l] = new double *[n4 * n3 * n2 + NR_END];
        m[n4l][n3l] += NR_END;
        m[n4l][n3l] -= n2l;

        // 5. 设置第三维的各个指针
        for (long i = n4l; i <= n4h; i++)
        {
            for (long j = n3l; j <= n3h; j++)
            {
                if (i == n4l && j == n3l)
                    continue;
                m[i][j] = m[n4l][n3l] + ((i - n4l) * n3 + (j - n3l)) * n2;
            }
        }

        // 6. 分配第一维数据块（连续内存）
        m[n4l][n3l][n2l] = new double[n4 * n3 * n2 * n1 + NR_END];
        m[n4l][n3l][n2l] += NR_END;
        m[n4l][n3l][n2l] -= n1l;

        // 7. 设置第二维的各个指针
        for (long i = n4l; i <= n4h; i++)
        {
            for (long j = n3l; j <= n3h; j++)
            {
                for (long k = n2l; k <= n2h; k++)
                {
                     if (i == n4l && j == n3l && k == n2l)
                        continue;
                    m[i][j][k] = m[n4l][n3l][n2l] + (((i - n4l) * n3 + (j - n3l)) * n2 + (k - n2l)) * n1;
                }
            }
        }
    }
    catch (const std::bad_alloc &e)
    {
        throw std::runtime_error("内存分配失败 in dtensor4(): " + std::string(e.what()));
    }

    return m;
}

void free_dtensor4(double ****m, long n4l, long n3l, long n2l, long n1l)
{
    if (!m)
        return;

    // 释放第一维数据块
    if (m[n4l] && m[n4l][n3l] && m[n4l][n3l][n2l])
    {
        m[n4l][n3l][n2l] += n1l;
        m[n4l][n3l][n2l] -= NR_END;
        delete[] m[n4l][n3l][n2l];
    }

    // 释放第二维指针数组
    if (m[n4l] && m[n4l][n3l])
    {
        m[n4l][n3l] += n2l;
        m[n4l][n3l] -= NR_END;
        delete[] m[n4l][n3l];
    }

    // 释放第三维指针数组
    if (m[n4l])
    {
        m[n4l] += n3l;
        m[n4l] -= NR_END;
        delete[] m[n4l];
    }

    // 释放第四维指针数组
    m += n4l;
    m -= NR_END;
    delete[] m;
}

void free_dtensor3(double ***m, long nzl, long nyl, long nxl)
{
    if (!m)
        return;

    // 释放x数据块（恢复原始指针位置）
    if (m[nzl] && m[nzl][nyl])
    {
        m[nzl][nyl] += nxl; // 恢复x指针的原始位置
        m[nzl][nyl] -= NR_END;
        delete[] m[nzl][nyl];
    }

    // 释放y指针数组
    if (m[nzl])
    {
        m[nzl] += nyl;
        m[nzl] -= NR_END;
        delete[] m[nzl];
    }

    // 释放z指针数组
    m += nzl;
    m -= NR_END;
    delete[] m;
}

void free_memory()
{
    // 检查指针并释放内存 - 使用与分配时相同的参数
    if (rho_local != nullptr)
    {
        free_dtensor3(rho_local, nc_zst_out - 1, nc_yst_out - 1, nc_xst_out - 1);
        rho_local = nullptr;
    }

    if (rho_particle != nullptr)
    {
        free_dtensor4(rho_particle, 0, nc_zst_out - 1, nc_yst_out - 1, nc_xst_out - 1);
        rho_particle = nullptr;
    }

    // 这些数组使用标准索引从0开始
    if (ex != nullptr)
    {
        free_dtensor3(ex, nc_zst_out, nc_yst_out, nc_xst_out);
        ex = nullptr;
    }
    if (ey != nullptr)
    {
        free_dtensor3(ey, nc_zst_out, nc_yst_out, nc_xst_out);
        ey = nullptr;
    }
    if (ez != nullptr)
    {
        free_dtensor3(ez, nc_zst_out, nc_yst_out, nc_xst_out);
        ez = nullptr;
    }
    if (bx != nullptr)
    {
        free_dtensor3(bx, nc_zst_out, nc_yst_out, nc_xst_out);
        bx = nullptr;
    }
    if (by != nullptr)
    {
        free_dtensor3(by, nc_zst_out, nc_yst_out, nc_xst_out);
        by = nullptr;
    }
    if (bz != nullptr)
    {
        free_dtensor3(bz, nc_zst_out, nc_yst_out, nc_xst_out);
        bz = nullptr;
    }

    // 电流数组使用扩展的索引范围
    if (jx != nullptr)
    {
        free_dtensor3(jx, nc_zst_out - 1, nc_yst_out - 1, nc_xst_out - 1);
        jx = nullptr;
    }
    if (jy != nullptr)
    {
        free_dtensor3(jy, nc_zst_out - 1, nc_yst_out - 1, nc_xst_out - 1);
        jy = nullptr;
    }
    if (jz != nullptr)
    {
        free_dtensor3(jz, nc_zst_out - 1, nc_yst_out - 1, nc_xst_out - 1);
        jz = nullptr;
    }

    // 释放MPI边界交换缓冲区
    if (current_recv_xminus != nullptr)
    {
        free_dtensor3(current_recv_xminus, nc_zst_out - 1, nc_yst_out - 1, nc_xst_out - 1);
        current_recv_xminus = nullptr;
    }
    if (current_recv_xplus != nullptr)
    {
        free_dtensor3(current_recv_xplus, nc_zst_out - 1, nc_yst_out - 1, nc_xst_out - 1);
        current_recv_xplus = nullptr;
    }
    if (current_recv_yminus != nullptr)
    {
        free_dtensor3(current_recv_yminus, nc_zst_out - 1, nc_yst_out - 1, nc_xst_out - 1);
        current_recv_yminus = nullptr;
    }
    if (current_recv_yplus != nullptr)
    {
        free_dtensor3(current_recv_yplus, nc_zst_out - 1, nc_yst_out - 1, nc_xst_out - 1);
        current_recv_yplus = nullptr;
    }
    if (current_recv_zminus != nullptr)
    {
        free_dtensor3(current_recv_zminus, nc_zst_out - 1, nc_yst_out - 1, nc_xst_out - 1);
        current_recv_zminus = nullptr;
    }
    if (current_recv_zplus != nullptr)
    {
        free_dtensor3(current_recv_zplus, nc_zst_out - 1, nc_yst_out - 1, nc_xst_out - 1);
        current_recv_zplus = nullptr;
    }
}
