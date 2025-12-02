#ifndef BOUNDARY_H
#define BOUNDARY_H

double generate_laser_pulse(double time, double position);
double Gpulse(double time, double x, double y,double z);
void radiating_boundary(int my_rank, int is_left, int is_right, int num_procs, int step);

#endif // BOUNDARY_H