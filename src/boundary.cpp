#include "boundary.h"
#include "constants.h"
#include "field.h"
#include <cmath>
#include <cstdio>

// 生成激光脉冲电场
double generate_laser_pulse(double time, double position)
{
    double t = time - position - laser_start_time;

    double envelope;
    if (t >= 0.0 && t <= 1.0 * laser_pulse_duration)
    {
        envelope = sin(Pi * t / laser_pulse_duration); // 正弦包络
        //envelope = exp(-pow((t - 0.5 * laser_pulse_duration) / (0.25 * laser_pulse_duration), 2)); // 高斯包络
        //envelope = 1.0;
    }
    else
    {
        envelope = 0.0; // 脉冲结束后电场为0
    }
    // 振荡项
    double oscillation = sin(omega * t);

    return laser_amplitude * envelope; // 返回激光电场强度
}

double Gpulse(double time, double x, double y, double z)
{
    double k = 2.0 * Pi / l0;                   // 波数
    double RayleighL = 0.5 * k * waist * waist; // 瑞利长度
    double x_eff = x - xfocus;
    double phi = atan(x_eff / RayleighL); // 相位
    double w = waist * sqrt(1 + pow(x_eff / RayleighL, 2));
    double PHI = (x_eff / RayleighL) * (y * y + z * z) / w / w - phi + k * x_eff - omega * time;
    double tau = laser_pulse_duration / (2.0 * sqrt(log(2.0)));
    double E = -1.0 * generate_laser_pulse(time, x) * waist / w * exp(-(y * y + z * z) / (w * w)) * cos(PHI + phase);
    return E;
}

void radiating_boundary(int my_rank, int is_left, int is_right, int num_procs, int step)
{

    double time = step * dt; // 当前时间步长
    // 辐射边界条件处理
    // 这里可以实现更复杂的边界处理逻辑
    double S_l = 0.0; // 辐射源项
    double P_l = 0.0;

    double S_r = 0.0; // 辐射源项
    double P_r = 0.0;

    double alpha = 1 / (1 + dt / dx);
    double beta = 1 - dt / dx;
    for(int j = nc_zst; j <= nc_zend; j++)
    {
        for (int i = nc_yst; i <= nc_yend; i++)
      {

        double y = (i + yst) * dy - yfocus;
        double z = (j + zst) * dz - zfocus;

        if (is_left)
        {
            if (laser_polarization == 1 || laser_polarization == 3)
            {
                S_l = Gpulse(time, 0.0, y, z);
            }
            if (laser_polarization == 2 || laser_polarization == 3)
            {
                P_l = Gpulse(time, 0.0, y, z);
            }
        }
        if (is_right)
        {
            if (laser_polarization == 1 || laser_polarization == 3)
            {
                S_r = Gpulse(time, Lx, y, z);
            }
            if (laser_polarization == 2 || laser_polarization == 3)
            {
                P_r = Gpulse(time, Lx, y, z);
            }
        }

        if (rank_xminus == -2)
        {
            bz[j][i][mem - 1] = alpha * (4 * S_l - 2 * ey[j][i][mem] - dt / dz * (bx[j][i][mem] - bx[j - 1][i][mem]) - beta * bz[j][i][mem] - dt * jy[j][i][mem]);
            by[j][i][mem - 1] = alpha * (-4 * P_l + 2 * ez[j][i][mem] + dt / dy * (bx[j][i][mem] - bx[j][i - 1][mem]) - beta * by[j][i][mem] + dt * jz[j][i][mem]);
        }
        if (rank_xplus == -2)
        {
            bz[j][i][local_xcells + 1] = alpha * (-4 * S_r + 2 * ey[j][i][local_xcells] - dt / dz * (bx[j][i][local_xcells] - bx[j - 1][i][local_xcells])- beta * bz[j][i][local_xcells] + dt * jy[j][i][local_xcells]);
            by[j][i][local_xcells + 1] = alpha * (4 * P_r - 2 * ez[j][i][local_xcells] + dt / dy * (bx[j][i][local_xcells] - bx[j][i - 1][local_xcells]) - beta * by[j][i][local_xcells] - dt * jz[j][i][local_xcells]);
        }
      }
    }
}
