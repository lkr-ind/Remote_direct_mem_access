#!/bin/bash
#PBS -l walltime=00:02:00
#PBS -l select=2:ncpus=1:mem=10gb:ngpus=1:gpu_type=A100
#PBS -l place=scatter
#PBS -e error.txt
#PBS -o output.txt

cd $PBS_O_WORKDIR
cat $PBS_NODEFILE

module purge
module load OpenMPI/4.1.4-NVHPC-22.7-CUDA-11.7.0

nvcc -x cu -ccbin=mpicxx -arch=sm_80 RDMA_mnodes_cuda_mpi.cpp -o rdma_mnodes_cuda_mpi
mpirun -np 2 --mca pml ucx ./rdma_mnodes_cuda_mpi
