#include "io_utils.h"
#include "constants.h"
#include "field.h"
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <mpi.h>

// 并行HDF5输出函数 - 每个进程直接写入自己的数据部分
void output_data_hdf5(int step, int my_rank, int num_procs, MPI_Comm my_comm, hid_t plist_id)
{
    // 创建输出目录（只有rank 0创建）
    if (my_rank == 0)
    {
        struct stat st = {0};
        if (stat("output", &st) == -1)
        {
            mkdir("output", 0755);
        }
    }

    // 同步确保目录创建完成
    MPI_Barrier(my_comm);

    char filename[256];
    sprintf(filename, "output/output_step_%04d.h5", step);

    // 设置并行HDF5访问
    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, my_comm, MPI_INFO_NULL);

    // 所有进程协同创建/打开文件
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5_SAFE_CALL(file_id);
    H5Pclose(fapl_id);

    // 计算全局和本地网格尺寸 - 使用constants.h中定义的全局尺寸
    int local_nx = local_xcells;
    int local_ny = local_ycells;
    int local_nz = local_zcells;

    // 获取当前进程的坐标
    int coords[3];
    MPI_Cart_coords(my_comm, my_rank, 3, coords);

    // 计算当前进程在全局数组中的起始位置
    int start_x = coords[0] * local_nx;
    int start_y = coords[1] * local_ny;
    int start_z = coords[2] * local_nz;

    // 定义全局数据空间 (z, y, x)
    hsize_t global_dims[3] = {(hsize_t)global_nz, (hsize_t)global_ny, (hsize_t)global_nx};
    hid_t global_space = H5Screate_simple(3, global_dims, NULL);
    H5_SAFE_CALL(global_space);

    // 定义本地数据空间 (z, y, x)
    hsize_t local_dims[3] = {(hsize_t)local_nz, (hsize_t)local_ny, (hsize_t)local_nx};
    hid_t local_space = H5Screate_simple(3, local_dims, NULL);
    H5_SAFE_CALL(local_space);

    // 定义内存中的hyperslab选择（本地数据布局）
    hsize_t mem_start[3] = {0, 0, 0};
    hsize_t mem_count[3] = {(hsize_t)local_nz, (hsize_t)local_ny, (hsize_t)local_nx};
    H5Sselect_hyperslab(local_space, H5S_SELECT_SET, mem_start, NULL, mem_count, NULL);

    // 定义文件中的hyperslab选择（全局数据中的位置）
    hsize_t file_start[3] = {(hsize_t)start_z, (hsize_t)start_y, (hsize_t)start_x};
    hsize_t file_count[3] = {(hsize_t)local_nz, (hsize_t)local_ny, (hsize_t)local_nx};
    H5Sselect_hyperslab(global_space, H5S_SELECT_SET, file_start, NULL, file_count, NULL);

    // 设置并行写入属性
    hid_t dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);

    // 写入三维场数据的lambda函数
    auto write_field_data = [&](double ***field, const char *dataset_name)
    {
        // 准备本地数据
        std::vector<double> local_data(local_nx * local_ny * local_nz);
        for (int k = 0; k < local_nz; k++)
        {
            for (int j = 0; j < local_ny; j++)
            {
                for (int i = 0; i < local_nx; i++)
                {
                    int idx = (k * local_ny + j) * local_nx + i;
                    local_data[idx] = field[nc_zst + k][nc_yst + j][nc_xst + i];
                }
            }
        }

        // 创建数据集
        hid_t dataset = H5Dcreate(file_id, dataset_name, H5T_NATIVE_DOUBLE, global_space,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5_SAFE_CALL(dataset);

        // 并行写入数据
        H5Dwrite(dataset, H5T_NATIVE_DOUBLE, local_space, global_space, dxpl_id, local_data.data());

        H5Dclose(dataset);
    };

    // 写入四维粒子密度数据的lambda函数
    auto write_particle_data = [&](double ****field, int particle_type, const char *dataset_name)
    {
        // 准备本地数据
        std::vector<double> local_data(local_nx * local_ny * local_nz);
        for (int k = 0; k < local_nz; k++)
        {
            for (int j = 0; j < local_ny; j++)
            {
                for (int i = 0; i < local_nx; i++)
                {
                    int idx = (k * local_ny + j) * local_nx + i;
                    local_data[idx] = field[particle_type][nc_zst + k][nc_yst + j][nc_xst + i];
                }
            }
        }

        // 创建数据集
        hid_t dataset = H5Dcreate(file_id, dataset_name, H5T_NATIVE_DOUBLE, global_space,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5_SAFE_CALL(dataset);

        // 并行写入数据
        H5Dwrite(dataset, H5T_NATIVE_DOUBLE, local_space, global_space, dxpl_id, local_data.data());

        H5Dclose(dataset);
    };

    // 写入所有场数据
    write_field_data(rho_local, "/rho");
    write_particle_data(rho_particle, 0, "/rho_electron");
    write_particle_data(rho_particle, 1, "/rho_proton");
    write_field_data(ex, "/ex");
    write_field_data(ey, "/ey");
    write_field_data(ez, "/ez");
    write_field_data(bx, "/bx");
    write_field_data(by, "/by");
    write_field_data(bz, "/bz");
    write_field_data(jx, "/jx");
    write_field_data(jy, "/jy");
    write_field_data(jz, "/jz");

    // 清理资源
    H5Pclose(dxpl_id);
    H5Sclose(local_space);
    H5Sclose(global_space);
    H5Fclose(file_id);

    // 同步所有进程
    MPI_Barrier(my_comm);
}

// 并行粒子输出函数
void output_particles_hdf5(int step, int my_rank, int num_procs, MPI_Comm my_comm,
                           const std::vector<std::vector<particle>> &particles,
                           const int *num_particles)
{
    // 创建输出目录（只有rank 0创建）
    if (my_rank == 0)
    {
        struct stat st = {0};
        if (stat("output", &st) == -1)
        {
            mkdir("output", 0755);
        }
    }

    // 同步确保目录创建完成
    MPI_Barrier(my_comm);

    char filename[256];
    sprintf(filename, "output/particles_step_%04d.h5", step);

    // 设置并行HDF5访问
    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, my_comm, MPI_INFO_NULL);

    // 所有进程协同创建/打开文件
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5_SAFE_CALL(file_id);
    H5Pclose(fapl_id);

    // 计算每个进程的粒子总数
    int total_particles_local = 0;
    for (int species = 0; species < num_kind; species++)
    {
        total_particles_local += num_particles[species];
    }

    // 使用MPI_Allgather收集所有进程的粒子数量
    std::vector<int> particles_per_rank(num_procs);
    MPI_Allgather(&total_particles_local, 1, MPI_INT, particles_per_rank.data(), 1, MPI_INT, my_comm);

    // 计算全局粒子总数和每个进程的起始偏移
    int total_particles_global = 0;
    int offset = 0;
    for (int i = 0; i < num_procs; i++)
    {
        if (i == my_rank)
        {
            offset = total_particles_global;
        }
        total_particles_global += particles_per_rank[i];
    }

    if (total_particles_global == 0)
    {
        // 如果没有粒子，关闭文件并返回
        H5Fclose(file_id);
        MPI_Barrier(my_comm);
        return;
    }

    // 定义粒子数据的HDF5复合数据类型
    hid_t particle_type = H5Tcreate(H5T_COMPOUND, sizeof(particle));
    H5Tinsert(particle_type, "id", HOFFSET(particle, id), H5T_NATIVE_INT);
    H5Tinsert(particle_type, "x", HOFFSET(particle, x), H5T_NATIVE_DOUBLE);
    H5Tinsert(particle_type, "y", HOFFSET(particle, y), H5T_NATIVE_DOUBLE);
    H5Tinsert(particle_type, "z", HOFFSET(particle, z), H5T_NATIVE_DOUBLE);
    H5Tinsert(particle_type, "x_old", HOFFSET(particle, x_old), H5T_NATIVE_DOUBLE);
    H5Tinsert(particle_type, "y_old", HOFFSET(particle, y_old), H5T_NATIVE_DOUBLE);
    H5Tinsert(particle_type, "z_old", HOFFSET(particle, z_old), H5T_NATIVE_DOUBLE);
    H5Tinsert(particle_type, "px", HOFFSET(particle, px), H5T_NATIVE_DOUBLE);
    H5Tinsert(particle_type, "py", HOFFSET(particle, py), H5T_NATIVE_DOUBLE);
    H5Tinsert(particle_type, "pz", HOFFSET(particle, pz), H5T_NATIVE_DOUBLE);
    H5Tinsert(particle_type, "sx", HOFFSET(particle, sx), H5T_NATIVE_DOUBLE);
    H5Tinsert(particle_type, "sy", HOFFSET(particle, sy), H5T_NATIVE_DOUBLE);
    H5Tinsert(particle_type, "sz", HOFFSET(particle, sz), H5T_NATIVE_DOUBLE);
    H5Tinsert(particle_type, "gamma", HOFFSET(particle, gamma), H5T_NATIVE_DOUBLE);
    H5Tinsert(particle_type, "charge", HOFFSET(particle, charge), H5T_NATIVE_DOUBLE);
    H5Tinsert(particle_type, "mass", HOFFSET(particle, mass), H5T_NATIVE_DOUBLE);
    H5Tinsert(particle_type, "weight", HOFFSET(particle, weight), H5T_NATIVE_DOUBLE);
    H5Tinsert(particle_type, "rank", HOFFSET(particle, rank), H5T_NATIVE_INT);

    // 创建全局数据空间
    hsize_t global_dims[1] = {(hsize_t)total_particles_global};
    hid_t global_space = H5Screate_simple(1, global_dims, NULL);
    H5_SAFE_CALL(global_space);

    // 创建本地数据空间
    hsize_t local_dims[1] = {(hsize_t)total_particles_local};
    hid_t local_space = H5Screate_simple(1, local_dims, NULL);
    H5_SAFE_CALL(local_space);

    // 在全局空间中选择本进程要写入的hyperslab
    hsize_t file_start[1] = {(hsize_t)offset};
    hsize_t file_count[1] = {(hsize_t)total_particles_local};
    if (total_particles_local > 0)
    {
        H5Sselect_hyperslab(global_space, H5S_SELECT_SET, file_start, NULL, file_count, NULL);
    }
    else
    {
        H5Sselect_none(global_space);
    }

    // 创建数据集
    hid_t dataset = H5Dcreate(file_id, "/particles", particle_type, global_space,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5_SAFE_CALL(dataset);

    // 设置并行写入属性
    hid_t dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);

    // 准备本地粒子数据
    std::vector<particle> local_particles;
    if (total_particles_local > 0)
    {
        local_particles.reserve(total_particles_local);

        // 将所有种类的粒子合并到一个向量中
        for (int species = 0; species < num_kind; species++)
        {
            for (int i = 0; i < num_particles[species]; i++)
            {
                local_particles.push_back(particles[species][i]);
            }
        }
    }

    // 创建合适的内存空间
    hid_t mem_space;
    if (total_particles_local > 0)
    {
        mem_space = local_space;
    }
    else
    {
        mem_space = H5Screate(H5S_NULL);
        H5_SAFE_CALL(mem_space);
    }

    // 并行写入粒子数据 - 所有进程都需要参与集合操作
    herr_t write_status = H5Dwrite(dataset, particle_type, mem_space, global_space, dxpl_id,
                                   total_particles_local > 0 ? local_particles.data() : nullptr);

    // 清理临时创建的空间
    if (total_particles_local == 0)
    {
        H5Sclose(mem_space);
    }

    // 添加属性信息
    if (my_rank == 0)
    {
        // 添加时间步属性
        hid_t attr_space = H5Screate(H5S_SCALAR);
        hid_t attr = H5Acreate(dataset, "timestep", H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_INT, &step);
        H5Aclose(attr);
        H5Sclose(attr_space);

        // 添加总粒子数属性
        attr_space = H5Screate(H5S_SCALAR);
        attr = H5Acreate(dataset, "total_particles", H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_INT, &total_particles_global);
        H5Aclose(attr);
        H5Sclose(attr_space);

        // 添加进程数属性
        attr_space = H5Screate(H5S_SCALAR);
        attr = H5Acreate(dataset, "num_processes", H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_INT, &num_procs);
        H5Aclose(attr);
        H5Sclose(attr_space);
    }

    // 清理资源
    H5Pclose(dxpl_id);
    H5Dclose(dataset);
    H5Sclose(local_space);
    H5Sclose(global_space);
    H5Tclose(particle_type);
    H5Fclose(file_id);

    // 同步所有进程
    MPI_Barrier(my_comm);

    if (my_rank == 0)
    {
        printf("Particle data written to %s (total particles: %d)\n", filename, total_particles_global);
    }
}
