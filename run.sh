#!/bin/bash
#SBATCH --job-name=PIC3D
#SBATCH --nodes=1
#SBATCH --nodelist=node003
#SBATCH --ntasks-per-node=125
#SBATCH --partition=large
#SBATCH --cpus-per-task=1
#SBATCH --output=PIC3D_correct_output_%j.txt
#SBATCH --error=PIC3D_correct_error_%j.txt

module load autotools prun/2.2 gnu12/12.2.0 cmake/3.24.2 ohpc  ucx/1.11.2 smilei/python nvhpc smilei/avx2 hdf5/hdf5-avx2

Cur_Dir=$(pwd)
echo $Cur_Dir
export PATH=$(pwd):$PATH
cd $Cur_Dir
rm -rf output
mkdir -p ./output
time mpirun -np 125 --use-hwthread-cpus  ./pic3d > ./output/run.log 
mv PIC3D_correct_output_*.txt ./output/
mv PIC3D_correct_error_*.txt ./output/