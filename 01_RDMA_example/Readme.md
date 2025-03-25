# GPUDirectRDMA

This directory consists a simple source code to demonstrate how you can use **GPUDirectRDMA** to transfer data between two GPUs on different nodes without involving the CPU. This is a simple add on to the **GPUDirectP2P** already present in the course.

In this case, the network card like "Infiniband" is used to transfer data between two GPUs.

## How to compile and run the code?

On HX1 cluster, you can compile the code using the following command:

```bash
module purge
module load OpenMPI/4.1.4-NVHPC-22.7-CUDA-11.7.0

nvcc -x cu -ccbin=mpicxx -arch=sm_80 RDMA_mnodes_cuda_mpi.cpp -o rdma_mnodes_cuda_mpi
```

To run the code, you can use the following command:

```bash
mpirun -np 2 --mca pml ucx ./rdma_mnodes_cuda_mpi
```

Both these details are mentioned in [Job_scipt](./submit.sh) file.

## Known issues

We have recently started testing **GPUDirectRDMA** on HX1 cluster. At present, we observed that code only works with the above module. Moreover, the code was tested on a special queue and needs to be tested on a genaral queue.

Please feel free to test the code and provide feedback.
