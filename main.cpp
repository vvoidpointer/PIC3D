#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <iostream>
#include <vector>

// 包含所有模块头文件
#include "constants.h"
#include "simulation.h"
#include "io_utils.h"
#include "field.h"
#include "particle.h"
#include "mpi_utils.h"

int main(int argc, char **argv)
{
    // 初始化MPI
    MPI_Init(&argc, &argv);

    int my_rank, num_procs;
    MPI_Comm my_comm;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Cart_create(MPI_COMM_WORLD, 3, rank_len, rank_period, rank_reorder, &my_comm);
    MPI_Comm_rank(my_comm, &my_rank);
    MPI_Cart_coords(my_comm, my_rank, 3, rank_coords);
    MPI_Cart_shift(my_comm, 0, +1, &rank_xminus, &rank_xplus);
    MPI_Cart_shift(my_comm, 1, +1, &rank_yminus, &rank_yplus);
    MPI_Cart_shift(my_comm, 2, +1, &rank_zminus, &rank_zplus);

    // 检查进程数量
    if (num_procs != rank_len[0] * rank_len[1] * rank_len[2])
    {
        if (my_rank == 0)
        {
            fprintf(stderr, "请使用%d个进程运行：mpirun -n %d ./a.out\n",
                    rank_len[0] * rank_len[1] * rank_len[2], rank_len[0] * rank_len[1] * rank_len[2]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    printf("进程 %d/%d 正在运行...\n, rank_xminus = %d, rank_xplus = %d, rank_yminus = %d, rank_yplus = %d, rank_zminus = %d, rank_zplus = %d\n",
         my_rank, num_procs, rank_xminus, rank_xplus, rank_yminus, rank_yplus, rank_zminus, rank_zplus);

    // 初始化HDF5
    H5open();
    hid_t plist_id = H5P_DEFAULT;

    // 初始化域划分
    initialize_domain(my_rank);

    // 加载g.txt数据（只在主进程执行）
    if (my_rank == 0)
    {
        if (!load_g_data())
        {
            fprintf(stderr, "Failed to load g.txt data\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    // 将数据广播到所有进程
    MPI_Bcast(g_data, g_data_size, MPI_DOUBLE, 0, my_comm);

    // 初始化模拟参数并输出信息
    initialize_simulation(my_rank, num_procs);

    // 初始化粒子
    int num_particles[num_kind];
    std::vector<std::vector<particle>> particles(num_kind);

    // 初始化电子 (类型0)
    initialize_particles(particles[0], num_particles[0], plasma_density, q_e, m_e, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, my_rank);
    // 初始化质子 (类型1)
    initialize_particles(particles[1], num_particles[1], plasma_density, -q_e, 1836.0 * m_e, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, my_rank);

    update_particle_position_momentum(particles[0], num_particles[0], my_rank, 0);
    update_particle_position_momentum(particles[1], num_particles[1], my_rank, 1);

    exchange_particle_data(particles[0], num_particles[0], my_rank, my_comm);
    exchange_particle_data(particles[1], num_particles[1], my_rank, my_comm);
    

    // 输出初始状态
    output_data_hdf5(0, my_rank, num_procs, my_comm, plist_id);
    output_particles_hdf5(0, my_rank, num_procs, my_comm, particles, num_particles);

    // 运行主时间循环
    run_main_time_loop(particles, num_particles, my_rank, num_procs, plist_id, my_comm);

    free_memory();

    // 清理和退出
    H5close();
    MPI_Finalize();
    return 0;
}
