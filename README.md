# PIC3D
This is a simulation program of particle-in-cell.
这是一个三维PIC模拟程序，用于模拟超强激光与等离子体相互作用过程
编译环境需要C++编译器，OpenMp库，HDF5库
该项目使用cmake管理项目，编译时需更改Makefile里的路径
该程序参数更改主要在constant.cpp
其他的.py文件是用于处理数据，可自行更改删除，也可借鉴
run.sh作为SLURM服务器任务系统提交脚本
