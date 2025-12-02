#include "mpi_utils.h"
#include "constants.h"
#include "particle.h"
#include <cstdio>

// 二维数组边界数据交换（使用MPI笛卡尔拓扑和自定义数据类型）
// data为double**，有效数据为data[1..rows][1..cols]
// 使用rank_xminus, rank_xplus, rank_yminus, rank_yplus进行通信
void exchange_boundary_data(double ***data, MPI_Comm my_comm)
{
    MPI_Request requests[12];
    MPI_Status statuses[12];
    int req_idx = 0;

    // 左右边界交换 (X方向)
    if (rank_xminus != -2)
    {
        // 发送左边界，接收左ghost
        MPI_Isend(&data[nc_zst_out][nc_yst_out][nc_xst], 1, type_field_Xdir, rank_xminus, TAG_FIELD_XMINUS, my_comm, &requests[req_idx++]);
        MPI_Irecv(&data[nc_zst_out][nc_yst_out][nc_xst_out], 1, type_field_Xdir, rank_xminus, TAG_FIELD_XPLUS, my_comm, &requests[req_idx++]);
    }
    if (rank_xplus != -2)
    {
        // 发送右边界，接收右ghost
        MPI_Isend(&data[nc_zst_out][nc_yst_out][nc_xend - mem + 1], 1, type_field_Xdir, rank_xplus, TAG_FIELD_XPLUS, my_comm, &requests[req_idx++]);
        MPI_Irecv(&data[nc_zst_out][nc_yst_out][nc_xend + 1], 1, type_field_Xdir, rank_xplus, TAG_FIELD_XMINUS, my_comm, &requests[req_idx++]);
    }

    // 前后边界交换 (Y方向)
    if (rank_yplus != -2)
    {
        // 发送前边界，接收前ghost
        MPI_Isend(&data[nc_zst_out][nc_yend - mem + 1][nc_xst_out], 1, type_field_Ydir, rank_yplus, TAG_FIELD_YPLUS, my_comm, &requests[req_idx++]);
        MPI_Irecv(&data[nc_zst_out][nc_yend + 1][nc_xst_out], 1, type_field_Ydir, rank_yplus, TAG_FIELD_YMINUS, my_comm, &requests[req_idx++]);
    }
    if (rank_yminus != -2)
    {
        // 发送后边界，接收后ghost
        MPI_Isend(&data[nc_zst_out][nc_yst][nc_xst_out], 1, type_field_Ydir, rank_yminus, TAG_FIELD_YMINUS, my_comm, &requests[req_idx++]);
        MPI_Irecv(&data[nc_zst_out][nc_yst_out][nc_xst_out], 1, type_field_Ydir, rank_yminus, TAG_FIELD_YPLUS, my_comm, &requests[req_idx++]);
    }

    // 上下边界交换 (Z方向)
    if (rank_zminus != -2)
    {
        // 发送下边界，接收下ghost
        MPI_Isend(&data[nc_zst][nc_yst_out][nc_xst_out], 1, type_field_Zdir, rank_zminus, TAG_FIELD_ZMINUS, my_comm, &requests[req_idx++]);
        MPI_Irecv(&data[nc_zst_out][nc_yst_out][nc_xst_out], 1, type_field_Zdir, rank_zminus, TAG_FIELD_ZPLUS, my_comm, &requests[req_idx++]);
    }
    if (rank_zplus != -2)
    {
        // 发送上边界，接收上ghost
        MPI_Isend(&data[nc_zend - mem + 1][nc_yst_out][nc_xst_out], 1, type_field_Zdir, rank_zplus, TAG_FIELD_ZPLUS, my_comm, &requests[req_idx++]);
        MPI_Irecv(&data[nc_zend + 1][nc_yst_out][nc_xst_out], 1, type_field_Zdir, rank_zplus, TAG_FIELD_ZMINUS, my_comm, &requests[req_idx++]);
    }

    // 等待所有通信完成
    if (req_idx > 0)
        MPI_Waitall(req_idx, requests, statuses);
}

void exchange_particle_data(std::vector<particle> &particles, int &num_particles, int my_rank, MPI_Comm my_comm)
{
    // 修正后的三维粒子交换：分三个阶段（X -> Y -> Z）进行
    // 这样可以正确处理对角线移动的粒子（例如：先移动到左边邻居，再由左边邻居移动到下边邻居）

    MPI_Request requests[8];
    MPI_Status statuses[8];
    int req_count = 0;

    // --- 阶段 1: X方向交换 ---
    {
        std::vector<particle> send_xminus, send_xplus;
        std::vector<particle> recv_xminus, recv_xplus;

        auto it = particles.begin();
        while (it != particles.end())
        {
            bool remove = false;
            if (it->x < xst && rank_xminus != -2) {
                send_xminus.push_back(*it);
                remove = true;
            } else if (it->x >= xend && rank_xplus != -2) {
                send_xplus.push_back(*it);
                remove = true;
            } else if ((it->x < 0.0 && rank_xminus == -2) || (it->x >= global_nx && rank_xplus == -2)) {
                remove = true; // 全局边界移除
            }

            if (remove) it = particles.erase(it);
            else ++it;
        }

        // 交换数量
        int send_counts[2] = {(int)send_xminus.size(), (int)send_xplus.size()};
        int recv_counts[2] = {0, 0};
        req_count = 0;

        if (rank_xminus != -2) {
            MPI_Isend(&send_counts[0], 1, MPI_INT, rank_xminus, TAG_PARTICLE_XMINUS_COUNT, my_comm, &requests[req_count++]);
            MPI_Irecv(&recv_counts[0], 1, MPI_INT, rank_xminus, TAG_PARTICLE_XPLUS_COUNT, my_comm, &requests[req_count++]);
        }
        if (rank_xplus != -2) {
            MPI_Isend(&send_counts[1], 1, MPI_INT, rank_xplus, TAG_PARTICLE_XPLUS_COUNT, my_comm, &requests[req_count++]);
            MPI_Irecv(&recv_counts[1], 1, MPI_INT, rank_xplus, TAG_PARTICLE_XMINUS_COUNT, my_comm, &requests[req_count++]);
        }
        if (req_count > 0) MPI_Waitall(req_count, requests, statuses);

        // 调整缓冲区
        if (recv_counts[0] > 0) recv_xminus.resize(recv_counts[0]);
        if (recv_counts[1] > 0) recv_xplus.resize(recv_counts[1]);

        // 交换数据
        req_count = 0;
        if (rank_xminus != -2 && send_counts[0] > 0) MPI_Isend(send_xminus.data(), send_counts[0] * sizeof(particle), MPI_BYTE, rank_xminus, TAG_PARTICLE_XMINUS_DATA, my_comm, &requests[req_count++]);
        if (rank_xminus != -2 && recv_counts[0] > 0) MPI_Irecv(recv_xminus.data(), recv_counts[0] * sizeof(particle), MPI_BYTE, rank_xminus, TAG_PARTICLE_XPLUS_DATA, my_comm, &requests[req_count++]);
        if (rank_xplus != -2 && send_counts[1] > 0) MPI_Isend(send_xplus.data(), send_counts[1] * sizeof(particle), MPI_BYTE, rank_xplus, TAG_PARTICLE_XPLUS_DATA, my_comm, &requests[req_count++]);
        if (rank_xplus != -2 && recv_counts[1] > 0) MPI_Irecv(recv_xplus.data(), recv_counts[1] * sizeof(particle), MPI_BYTE, rank_xplus, TAG_PARTICLE_XMINUS_DATA, my_comm, &requests[req_count++]);
        if (req_count > 0) MPI_Waitall(req_count, requests, statuses);

        // 合并粒子 (关键：立即合并以便进行下一阶段检查)
        if (recv_counts[0] > 0) particles.insert(particles.end(), recv_xminus.begin(), recv_xminus.end());
        if (recv_counts[1] > 0) particles.insert(particles.end(), recv_xplus.begin(), recv_xplus.end());
    }

    // --- 阶段 2: Y方向交换 ---
    {
        std::vector<particle> send_yminus, send_yplus;
        std::vector<particle> recv_yminus, recv_yplus;

        auto it = particles.begin();
        while (it != particles.end())
        {
            bool remove = false;
            if (it->y < yst && rank_yminus != -2) {
                send_yminus.push_back(*it);
                remove = true;
            } else if (it->y >= yend && rank_yplus != -2) {
                send_yplus.push_back(*it);
                remove = true;
            } else if ((it->y < 0.0 && rank_yminus == -2) || (it->y >= global_ny && rank_yplus == -2)) {
                remove = true;
            }

            if (remove) it = particles.erase(it);
            else ++it;
        }

        int send_counts[2] = {(int)send_yminus.size(), (int)send_yplus.size()};
        int recv_counts[2] = {0, 0};
        req_count = 0;

        if (rank_yminus != -2) {
            MPI_Isend(&send_counts[0], 1, MPI_INT, rank_yminus, TAG_PARTICLE_YMINUS_COUNT, my_comm, &requests[req_count++]);
            MPI_Irecv(&recv_counts[0], 1, MPI_INT, rank_yminus, TAG_PARTICLE_YPLUS_COUNT, my_comm, &requests[req_count++]);
        }
        if (rank_yplus != -2) {
            MPI_Isend(&send_counts[1], 1, MPI_INT, rank_yplus, TAG_PARTICLE_YPLUS_COUNT, my_comm, &requests[req_count++]);
            MPI_Irecv(&recv_counts[1], 1, MPI_INT, rank_yplus, TAG_PARTICLE_YMINUS_COUNT, my_comm, &requests[req_count++]);
        }
        if (req_count > 0) MPI_Waitall(req_count, requests, statuses);

        if (recv_counts[0] > 0) recv_yminus.resize(recv_counts[0]);
        if (recv_counts[1] > 0) recv_yplus.resize(recv_counts[1]);

        req_count = 0;
        if (rank_yminus != -2 && send_counts[0] > 0) MPI_Isend(send_yminus.data(), send_counts[0] * sizeof(particle), MPI_BYTE, rank_yminus, TAG_PARTICLE_YMINUS_DATA, my_comm, &requests[req_count++]);
        if (rank_yminus != -2 && recv_counts[0] > 0) MPI_Irecv(recv_yminus.data(), recv_counts[0] * sizeof(particle), MPI_BYTE, rank_yminus, TAG_PARTICLE_YPLUS_DATA, my_comm, &requests[req_count++]);
        if (rank_yplus != -2 && send_counts[1] > 0) MPI_Isend(send_yplus.data(), send_counts[1] * sizeof(particle), MPI_BYTE, rank_yplus, TAG_PARTICLE_YPLUS_DATA, my_comm, &requests[req_count++]);
        if (rank_yplus != -2 && recv_counts[1] > 0) MPI_Irecv(recv_yplus.data(), recv_counts[1] * sizeof(particle), MPI_BYTE, rank_yplus, TAG_PARTICLE_YMINUS_DATA, my_comm, &requests[req_count++]);
        if (req_count > 0) MPI_Waitall(req_count, requests, statuses);

        if (recv_counts[0] > 0) particles.insert(particles.end(), recv_yminus.begin(), recv_yminus.end());
        if (recv_counts[1] > 0) particles.insert(particles.end(), recv_yplus.begin(), recv_yplus.end());
    }

    // --- 阶段 3: Z方向交换 ---
    {
        std::vector<particle> send_zminus, send_zplus;
        std::vector<particle> recv_zminus, recv_zplus;

        auto it = particles.begin();
        while (it != particles.end())
        {
            bool remove = false;
            if (it->z < zst && rank_zminus != -2) {
                send_zminus.push_back(*it);
                remove = true;
            } else if (it->z >= zend && rank_zplus != -2) {
                send_zplus.push_back(*it);
                remove = true;
            } else if ((it->z < 0.0 && rank_zminus == -2) || (it->z >= global_nz && rank_zplus == -2)) {
                remove = true;
            }

            if (remove) it = particles.erase(it);
            else ++it;
        }

        int send_counts[2] = {(int)send_zminus.size(), (int)send_zplus.size()};
        int recv_counts[2] = {0, 0};
        req_count = 0;

        if (rank_zminus != -2) {
            MPI_Isend(&send_counts[0], 1, MPI_INT, rank_zminus, TAG_PARTICLE_ZMINUS_COUNT, my_comm, &requests[req_count++]);
            MPI_Irecv(&recv_counts[0], 1, MPI_INT, rank_zminus, TAG_PARTICLE_ZPLUS_COUNT, my_comm, &requests[req_count++]);
        }
        if (rank_zplus != -2) {
            MPI_Isend(&send_counts[1], 1, MPI_INT, rank_zplus, TAG_PARTICLE_ZPLUS_COUNT, my_comm, &requests[req_count++]);
            MPI_Irecv(&recv_counts[1], 1, MPI_INT, rank_zplus, TAG_PARTICLE_ZMINUS_COUNT, my_comm, &requests[req_count++]);
        }
        if (req_count > 0) MPI_Waitall(req_count, requests, statuses);

        if (recv_counts[0] > 0) recv_zminus.resize(recv_counts[0]);
        if (recv_counts[1] > 0) recv_zplus.resize(recv_counts[1]);

        req_count = 0;
        if (rank_zminus != -2 && send_counts[0] > 0) MPI_Isend(send_zminus.data(), send_counts[0] * sizeof(particle), MPI_BYTE, rank_zminus, TAG_PARTICLE_ZMINUS_DATA, my_comm, &requests[req_count++]);
        if (rank_zminus != -2 && recv_counts[0] > 0) MPI_Irecv(recv_zminus.data(), recv_counts[0] * sizeof(particle), MPI_BYTE, rank_zminus, TAG_PARTICLE_ZPLUS_DATA, my_comm, &requests[req_count++]);
        if (rank_zplus != -2 && send_counts[1] > 0) MPI_Isend(send_zplus.data(), send_counts[1] * sizeof(particle), MPI_BYTE, rank_zplus, TAG_PARTICLE_ZPLUS_DATA, my_comm, &requests[req_count++]);
        if (rank_zplus != -2 && recv_counts[1] > 0) MPI_Irecv(recv_zplus.data(), recv_counts[1] * sizeof(particle), MPI_BYTE, rank_zplus, TAG_PARTICLE_ZMINUS_DATA, my_comm, &requests[req_count++]);
        if (req_count > 0) MPI_Waitall(req_count, requests, statuses);

        if (recv_counts[0] > 0) particles.insert(particles.end(), recv_zminus.begin(), recv_zminus.end());
        if (recv_counts[1] > 0) particles.insert(particles.end(), recv_zplus.begin(), recv_zplus.end());
    }

    // 更新粒子数量
    num_particles = particles.size();
}

void exchange_boundary_current_fixed(double ***data, MPI_Comm my_comm)
{
    for (int k = nc_zst_out - 1; k <= nc_zend_out + 1; k++)
    {
        for (int j = nc_yst_out - 1; j <= nc_yend_out + 1; j++)
        {
            for (int i = nc_xst_out - 1; i <= nc_xend_out + 1; i++)
            {
                current_recv_xminus[k][j][i] = 0.0;
                current_recv_xplus[k][j][i] = 0.0;
                current_recv_yminus[k][j][i] = 0.0;
                current_recv_yplus[k][j][i] = 0.0;
                current_recv_zminus[k][j][i] = 0.0;
                current_recv_zplus[k][j][i] = 0.0;
            }
        }
    }
    // 修正后的电流边界交换：必须分阶段顺序执行 (X -> Y -> Z)
    // 1. 累积阶段 (Accumulate): 将Ghost区的电流沉积发送给邻居的物理区
    // 2. 同步阶段 (Sync): 将物理区的总电流同步回邻居的Ghost区
    // 顺序执行是为了正确处理角点 (Corner) 和棱边 (Edge) 的数据传递

    MPI_Request requests[4];
    MPI_Status statuses[4];
    
    // ==========================================
    // 第一阶段：Ghost数据累积 (Ghost -> Physical)
    // 顺序：X -> Y -> Z
    // ==========================================

    // --- 1. X方向累积 ---
    {
        int req_count = 0;
        // 发送左Ghost -> 左邻居 (加到右物理)
        // 接收左邻居的右Ghost -> 加到左物理
        if (rank_xminus != MPI_PROC_NULL) {
            MPI_Isend(&data[nc_zst_out - 1][nc_yst_out - 1][nc_xst_out - 1], 1, type_current_Xdir, rank_xminus, TAG_CURRENT_XMINUS_ACCUMULATE, my_comm, &requests[req_count++]);
            MPI_Irecv(&current_recv_xminus[nc_zst_out - 1][nc_yst_out - 1][nc_xst_out - 1], 1, type_current_Xdir, rank_xminus, TAG_CURRENT_XPLUS_ACCUMULATE, my_comm, &requests[req_count++]);
        }
        // 发送右Ghost -> 右邻居 (加到左物理)
        // 接收右邻居的左Ghost -> 加到右物理
        if (rank_xplus != MPI_PROC_NULL) {
            MPI_Isend(&data[nc_zst_out - 1][nc_yst_out - 1][nc_xend + 1], 1, type_current_Xdir, rank_xplus, TAG_CURRENT_XPLUS_ACCUMULATE, my_comm, &requests[req_count++]);
            MPI_Irecv(&current_recv_xplus[nc_zst_out - 1][nc_yst_out - 1][nc_xend + 1], 1, type_current_Xdir, rank_xplus, TAG_CURRENT_XMINUS_ACCUMULATE, my_comm, &requests[req_count++]);
        }
        
        if (req_count > 0) MPI_Waitall(req_count, requests, statuses);
        
        // 应用X方向累积
        if (rank_xminus != MPI_PROC_NULL) {
            for(int k = nc_zst_out - 1; k <= nc_zend_out + 1; k++) {
                for (int j = nc_yst_out - 1; j <= nc_yend_out + 1; j++) {
                    for (int i = 0; i <= mem; i++) {
                        data[k][j][nc_xst + i] += current_recv_xminus[k][j][nc_xst_out - 1 + i];
                    }
                }
            }
        }
        if (rank_xplus != MPI_PROC_NULL) {
            for(int k = nc_zst_out - 1; k <= nc_zend_out + 1; k++) {
                for (int j = nc_yst_out - 1; j <= nc_yend_out + 1; j++) {
                    for (int i = 0; i <= mem; i++) {
                        data[k][j][nc_xend - mem + i] += current_recv_xplus[k][j][nc_xend + 1 + i];
                    }
                }
            }
        }
    }

    // --- 2. Y方向累积 (包含已更新的X方向贡献) ---
    {
        int req_count = 0;
        if (rank_yminus != MPI_PROC_NULL) {
            MPI_Isend(&data[nc_zst_out - 1][nc_yst_out - 1][nc_xst_out - 1], 1, type_current_Ydir, rank_yminus, TAG_CURRENT_YMINUS_ACCUMULATE, my_comm, &requests[req_count++]);
            MPI_Irecv(&current_recv_yminus[nc_zst_out - 1][nc_yst_out - 1][nc_xst_out - 1], 1, type_current_Ydir, rank_yminus, TAG_CURRENT_YPLUS_ACCUMULATE, my_comm, &requests[req_count++]);
        }
        if (rank_yplus != MPI_PROC_NULL) {
            MPI_Isend(&data[nc_zst_out - 1][nc_yend + 1][nc_xst_out - 1], 1, type_current_Ydir, rank_yplus, TAG_CURRENT_YPLUS_ACCUMULATE, my_comm, &requests[req_count++]);
            MPI_Irecv(&current_recv_yplus[nc_zst_out - 1][nc_yend + 1][nc_xst_out - 1], 1, type_current_Ydir, rank_yplus, TAG_CURRENT_YMINUS_ACCUMULATE, my_comm, &requests[req_count++]);
        }
        
        if (req_count > 0) MPI_Waitall(req_count, requests, statuses);
        
        // 应用Y方向累积
        if (rank_yminus != MPI_PROC_NULL) {
            for(int k = nc_zst_out - 1; k <= nc_zend_out + 1; k++) {
                for (int j = 0; j <= mem; j++) {
                    for (int i = nc_xst_out - 1; i <= nc_xend_out + 1; i++) {
                        data[k][nc_yst + j][i] += current_recv_yminus[k][nc_yst_out - 1 + j][i];
                    }
                }
            }
        }
        if (rank_yplus != MPI_PROC_NULL) {
            for(int k = nc_zst_out - 1; k <= nc_zend_out + 1; k++) {
                for (int j = 0; j <= mem; j++) {
                    for (int i = nc_xst_out - 1; i <= nc_xend_out + 1; i++) {
                        data[k][nc_yend - mem + j][i] += current_recv_yplus[k][nc_yend + 1 + j][i];
                    }
                }
            }
        }
    }

    // --- 3. Z方向累积 (包含已更新的X, Y方向贡献) ---
    {
        int req_count = 0;
        if (rank_zminus != MPI_PROC_NULL) {
            MPI_Isend(&data[nc_zst_out - 1][nc_yst_out - 1][nc_xst_out - 1], 1, type_current_Zdir, rank_zminus, TAG_CURRENT_ZMINUS_ACCUMULATE, my_comm, &requests[req_count++]);
            MPI_Irecv(&current_recv_zminus[nc_zst_out - 1][nc_yst_out - 1][nc_xst_out - 1], 1, type_current_Zdir, rank_zminus, TAG_CURRENT_ZPLUS_ACCUMULATE, my_comm, &requests[req_count++]);
        }
        if (rank_zplus != MPI_PROC_NULL) {
            MPI_Isend(&data[nc_zend + 1][nc_yst_out - 1][nc_xst_out - 1], 1, type_current_Zdir, rank_zplus, TAG_CURRENT_ZPLUS_ACCUMULATE, my_comm, &requests[req_count++]);
            MPI_Irecv(&current_recv_zplus[nc_zend + 1][nc_yst_out - 1][nc_xst_out - 1], 1, type_current_Zdir, rank_zplus, TAG_CURRENT_ZMINUS_ACCUMULATE, my_comm, &requests[req_count++]);
        }
        
        if (req_count > 0) MPI_Waitall(req_count, requests, statuses);
        
        // 应用Z方向累积
        if (rank_zminus != MPI_PROC_NULL) {
            for(int k = 0; k <= mem; k++) {
                for (int j = nc_yst_out - 1; j <= nc_yend_out + 1; j++) {
                    for (int i = nc_xst_out - 1; i <= nc_xend_out + 1; i++) {
                        data[nc_zst + k][j][i] += current_recv_zminus[nc_zst_out - 1 + k][j][i];
                    }
                }
            }
        }
        if (rank_zplus != MPI_PROC_NULL) {
            for(int k = 0; k <= mem; k++) {
                for (int j = nc_yst_out - 1; j <= nc_yend_out + 1; j++) {
                    for (int i = nc_xst_out - 1; i <= nc_xend_out + 1; i++) {
                        data[nc_zend - mem + k][j][i] += current_recv_zplus[nc_zend + 1 + k][j][i];
                    }
                }
            }
        }
    }

    // ==========================================
    // 第二阶段：Ghost数据同步 (Physical -> Ghost)
    // 顺序：X -> Y -> Z (确保角点数据正确传播)
    // ==========================================
    
    // 1. X方向同步
    {
        int req_count = 0;
        if (rank_xminus != MPI_PROC_NULL) {
            MPI_Isend(&data[nc_zst_out - 1][nc_yst_out - 1][nc_xst], 1, type_current_Xdir, rank_xminus, TAG_CURRENT_XMINUS_DATA, my_comm, &requests[req_count++]);
            MPI_Irecv(&data[nc_zst_out - 1][nc_yst_out - 1][nc_xst_out - 1], 1, type_current_Xdir, rank_xminus, TAG_CURRENT_XPLUS_DATA, my_comm, &requests[req_count++]);
        }
        if (rank_xplus != MPI_PROC_NULL) {
            MPI_Isend(&data[nc_zst_out - 1][nc_yst_out - 1][nc_xend - mem], 1, type_current_Xdir, rank_xplus, TAG_CURRENT_XPLUS_DATA, my_comm, &requests[req_count++]);
            MPI_Irecv(&data[nc_zst_out - 1][nc_yst_out - 1][nc_xend + 1], 1, type_current_Xdir, rank_xplus, TAG_CURRENT_XMINUS_DATA, my_comm, &requests[req_count++]);
        }
        if (req_count > 0) MPI_Waitall(req_count, requests, statuses);
    }

    // 2. Y方向同步
    {
        int req_count = 0;
        if (rank_yplus != MPI_PROC_NULL) {
            MPI_Isend(&data[nc_zst_out - 1][nc_yend - mem][nc_xst_out - 1], 1, type_current_Ydir, rank_yplus, TAG_CURRENT_YPLUS_DATA, my_comm, &requests[req_count++]);
            MPI_Irecv(&data[nc_zst_out - 1][nc_yend + 1][nc_xst_out - 1], 1, type_current_Ydir, rank_yplus, TAG_CURRENT_YMINUS_DATA, my_comm, &requests[req_count++]);
        }
        if (rank_yminus != MPI_PROC_NULL) {
            MPI_Isend(&data[nc_zst_out - 1][nc_yst][nc_xst_out - 1], 1, type_current_Ydir, rank_yminus, TAG_CURRENT_YMINUS_DATA, my_comm, &requests[req_count++]);
            MPI_Irecv(&data[nc_zst_out - 1][nc_yst_out - 1][nc_xst_out - 1], 1, type_current_Ydir, rank_yminus, TAG_CURRENT_YPLUS_DATA, my_comm, &requests[req_count++]);
        }
        if (req_count > 0) MPI_Waitall(req_count, requests, statuses);
    }

    // 3. Z方向同步
    {
        int req_count = 0;
        if (rank_zminus != MPI_PROC_NULL) {
            MPI_Isend(&data[nc_zst][nc_yst_out - 1][nc_xst_out - 1], 1, type_current_Zdir, rank_zminus, TAG_CURRENT_ZMINUS_DATA, my_comm, &requests[req_count++]);
            MPI_Irecv(&data[nc_zst_out - 1][nc_yst_out - 1][nc_xst_out - 1], 1, type_current_Zdir, rank_zminus, TAG_CURRENT_ZPLUS_DATA, my_comm, &requests[req_count++]);
        }
        if (rank_zplus != MPI_PROC_NULL) {
            MPI_Isend(&data[nc_zend - mem][nc_yst_out - 1][nc_xst_out - 1], 1, type_current_Zdir, rank_zplus, TAG_CURRENT_ZPLUS_DATA, my_comm, &requests[req_count++]);
            MPI_Irecv(&data[nc_zend + 1][nc_yst_out - 1][nc_xst_out - 1], 1, type_current_Zdir, rank_zplus, TAG_CURRENT_ZMINUS_DATA, my_comm, &requests[req_count++]);
        }
        if (req_count > 0) MPI_Waitall(req_count, requests, statuses);
    }
}