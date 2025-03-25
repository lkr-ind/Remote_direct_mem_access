# How to get local rank of the process in a node?

This code explains how you can get the local rank of the MPI process relative to that node. In order to explain, why it is necessary, wei will consider a simple example.

Let us assume that you are running a job with 4 MPI processes with 4 GPUs. For our example, we also assume that each node has 4 GPUs. 

In this case, the mapping of the MPI processes to the GPUs will look like the following:

| Node number | MPI Rank | GPU ID |
|-------------|----------|--------|
| 0           | 0        | 0      |
| 0           | 1        | 1      |
| 0           | 2        | 2      |
| 0           | 3        | 3      |

In this case, the mapping looks easy and straight forward because each MPI process mapped to the GPU with the same ID as the MPI rank. But, this is not always the case.

Now, let us assume that you are running a job with 4 MPI processes on 2 nodes. On each node, you are using 2 GPUs. The GPU ID start from `0` on each node. When you submit a job to the scheduler, the scheduler will assign you 2 Gpus on both nodes with IDs `0` and `1` on each node. Thus, it becomes important to find the local MPI rank relative to the node so that you can map your MPI process to a GPU.

In this case, the mapping of the MPI processes to the GPUs should look like the following:

| Node number | Global MPI Rank | Local MPI rank relative to node | GPU ID |
|-------------|-----------------|----------------------------------|--------|
| 0           | 0               | 0                                | 0      |
| 0           | 1               | 1                                | 1      |
| 1           | 2               | 0                                | 0      |
| 1           | 3               | 1                                | 1      |

The program in the code snippet [Local_rank_node.cpp](./Local_rank_node.cpp) allows you to find the local rank of the MPI process relative to the node. 

The code does this by creating a new communicator for each node and then finding the rank of the process in that communicator. This code is generic and should work with both **OpenMPI** and **MPICH**.

OpenMPI has also some environment variables that can help you to achieve the same. You can find more details at [MPI environment variable for OpenMPI](https://www.open-mpi.org/faq/?category=running#mpi-environmental-variables).

## Compile and Run the code

You can compile the code with any OpenMPI module by the following command

```bash
mpic++ Local_rank_node.cpp -o Local_rank_node
```

When you run the code, you will see the output like the following:

```bash
mpirun -np 4 ./Local_rank_node

Global Rank: 1, Local Rank: 1, Node: hx1-d12-gpu-05
Global Rank: 0, Local Rank: 0, Node: hx1-d12-gpu-05
Global Rank: 2, Local Rank: 0, Node: hx1-d12-gpu-02
Global Rank: 3, Local Rank: 1, Node: hx1-d12-gpu-02
```
