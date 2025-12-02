#include "field.h"
#include "mpi_utils.h"
#include <cstring>
#include <cstdio>
#include <vector>

// 场变量定义
double max_error = 1.0;
double ***rho_local;
double ****rho_particle; // 三类粒子的电荷密度 (大小固定为3，循环使用num_kind)
double ***ex;
double ***phi;
double ***phi_new;
double ***ey;
double ***ez;
double ***bx;
double ***by;
double ***bz;
double ***jx;
double ***jy;
double ***jz;
double ***current_recv_xminus, ***current_recv_xplus, ***current_recv_yminus, ***current_recv_yplus, ***current_recv_zminus, ***current_recv_zplus;

void update_electromagnetic_fields(int my_rank, int num_procs, MPI_Comm my_comm)
{
    // 更新E(n+1/2) - Maxwell方程: ∂E/∂t = c²(∇×B - μ₀J)
    for(int k = nc_zst;k<=nc_zend;k++){
        for(int j=nc_yst;j<=nc_yend;j++){
            for(int i=nc_xst;i<=nc_xend;i++){
                ex[k][j][i] += dt * ((bz[k][j][i] - bz[k][j-1][i]) / dy - (by[k][j][i] - by[k-1][j][i]) / dz - jx[k][j][i]); // 更新Ex
                ey[k][j][i] += dt * ((bx[k][j][i] - bx[k-1][j][i]) / dz - (bz[k][j][i] - bz[k][j][i-1]) / dx - jy[k][j][i]); // 更新Ey
                ez[k][j][i] += dt * ((by[k][j][i] - by[k][j][i-1]) / dx - (bx[k][j][i] - bx[k][j-1][i]) / dy - jz[k][j][i]); // 更新Ez
            }
        }
    }
    exchange_boundary_data(ex, my_comm);
    exchange_boundary_data(ey, my_comm);
    exchange_boundary_data(ez, my_comm);

    // 更新B(n+1/2) - Maxwell方程: ∂B/∂t = -∇×E
    
    for(int k = nc_zst;k<=nc_zend;k++){
        for(int j=nc_yst;j<=nc_yend;j++){
            for(int i=nc_xst;i<=nc_xend;i++){
                bx[k][j][i] -= dt * ((ez[k][j+1][i] - ez[k][j][i]) / dy - (ey[k+1][j][i] - ey[k][j][i]) / dz); // 更新Bx
                by[k][j][i] -= dt * ((ex[k+1][j][i] - ex[k][j][i]) / dz - (ez[k][j][i+1] - ez[k][j][i]) / dx); // 更新By
                bz[k][j][i] -= dt * ((ey[k][j][i+1] - ey[k][j][i]) / dx - (ex[k][j+1][i] - ex[k][j][i]) / dy); // 更新Bz
            }
        }
    }
    exchange_boundary_data(bx, my_comm);
    exchange_boundary_data(by, my_comm);
    exchange_boundary_data(bz, my_comm);
}

void update_electromagnetic_fields_first(int my_rank, int num_procs, MPI_Comm my_comm)
{
    // 更新E(n+1/2) - Maxwell方程: ∂E/∂t = c²(∇×B - μ₀J)
    for(int k = nc_zst;k<=nc_zend;k++){
        for(int j=nc_yst;j<=nc_yend;j++){
            for(int i=nc_xst;i<=nc_xend;i++){
                ex[k][j][i] += dt_half * ((bz[k][j][i] - bz[k][j-1][i]) / dy - (by[k][j][i] - by[k-1][j][i]) / dz - jx[k][j][i]); // 更新Ex
                ey[k][j][i] += dt_half * ((bx[k][j][i] - bx[k-1][j][i]) / dz - (bz[k][j][i] - bz[k][j][i-1]) / dx - jy[k][j][i]); // 更新Ey
                ez[k][j][i] += dt_half * ((by[k][j][i] - by[k][j][i-1]) / dx - (bx[k][j][i] - bx[k][j-1][i]) / dy - jz[k][j][i]); // 更新Ez
            }
        }
    }
    exchange_boundary_data(ex, my_comm);
    exchange_boundary_data(ey, my_comm);
    exchange_boundary_data(ez, my_comm);

    // 更新B(n+1/2) - Maxwell方程: ∂B/∂t = -∇×E
    
    for(int k = nc_zst;k<=nc_zend;k++){
        for(int j=nc_yst;j<=nc_yend;j++){
            for(int i=nc_xst;i<=nc_xend;i++){
                bx[k][j][i] -= dt_half * ((ez[k][j+1][i] - ez[k][j][i]) / dy - (ey[k+1][j][i] - ey[k][j][i]) / dz); // 更新Bx
                by[k][j][i] -= dt_half * ((ex[k+1][j][i] - ex[k][j][i]) / dz - (ez[k][j][i+1] - ez[k][j][i]) / dx); // 更新By
                bz[k][j][i] -= dt_half * ((ey[k][j][i+1] - ey[k][j][i]) / dx - (ex[k][j+1][i] - ex[k][j][i]) / dy); // 更新Bz
            }
        }
    }
    exchange_boundary_data(bx, my_comm);
    exchange_boundary_data(by, my_comm);
    exchange_boundary_data(bz, my_comm);
}

void update_electromagnetic_fields_second(int my_rank, int num_procs, MPI_Comm my_comm)
{
    // 更新B(n+1/2) - Maxwell方程: ∂B/∂t = -∇×E
    
    for(int k = nc_zst;k<=nc_zend;k++){
        for(int j=nc_yst;j<=nc_yend;j++){
            for(int i=nc_xst;i<=nc_xend;i++){
                bx[k][j][i] -= dt_half * ((ez[k][j+1][i] - ez[k][j][i]) / dy - (ey[k+1][j][i] - ey[k][j][i]) / dz); // 更新Bx
                by[k][j][i] -= dt_half * ((ex[k+1][j][i] - ex[k][j][i]) / dz - (ez[k][j][i+1] - ez[k][j][i]) / dx); // 更新By
                bz[k][j][i] -= dt_half * ((ey[k][j][i+1] - ey[k][j][i]) / dx - (ex[k][j+1][i] - ex[k][j][i]) / dy); // 更新Bz
            }
        }
    }
    exchange_boundary_data(bx, my_comm);
    exchange_boundary_data(by, my_comm);
    exchange_boundary_data(bz, my_comm);

    // 更新E(n+1/2) - Maxwell方程: ∂E/∂t = c²(∇×B - μ₀J)
    for(int k = nc_zst;k<=nc_zend;k++){
        for(int j=nc_yst;j<=nc_yend;j++){
            for(int i=nc_xst;i<=nc_xend;i++){
                ex[k][j][i] += dt_half * ((bz[k][j][i] - bz[k][j-1][i]) / dy - (by[k][j][i] - by[k-1][j][i]) / dz - jx[k][j][i]); // 更新Ex
                ey[k][j][i] += dt_half * ((bx[k][j][i] - bx[k-1][j][i]) / dz - (bz[k][j][i] - bz[k][j][i-1]) / dx - jy[k][j][i]); // 更新Ey
                ez[k][j][i] += dt_half * ((by[k][j][i] - by[k][j][i-1]) / dx - (bx[k][j][i] - bx[k][j-1][i]) / dy - jz[k][j][i]); // 更新Ez
            }
        }
    }
    exchange_boundary_data(ex, my_comm);
    exchange_boundary_data(ey, my_comm);
    exchange_boundary_data(ez, my_comm);
}