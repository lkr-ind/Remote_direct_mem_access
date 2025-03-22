#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>

#define N 1024  // Number of integers

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        std::cerr << "Error: This program requires at least two MPI processes.\n";
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Allocate GPU memory
    int *d_send, *d_recv;
    checkCudaError(cudaMalloc((void**)&d_send, N * sizeof(int)), "cudaMalloc d_send");
    checkCudaError(cudaMalloc((void**)&d_recv, N * sizeof(int)), "cudaMalloc d_recv");

    if (rank == 0) {
        // Initialize data on GPU
        int h_data[N];
        for (int i = 0; i < N; i++) h_data[i] = i;
        checkCudaError(cudaMemcpy(d_send, h_data, N * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy H->D");

        // Send GPU data to Rank 1
        std::cout << "Rank 0 sending data to Rank 1...\n";
        MPI_Send(d_send, N, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } 
    else if (rank == 1) {
        // Receive GPU data from Rank 0
        std::cout << "Rank 1 receiving data from Rank 0...\n";
        MPI_Recv(d_recv, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Copy received data back to host for verification
        int h_data[N];
        checkCudaError(cudaMemcpy(h_data, d_recv, N * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy D->H");

        // Check correctness
        bool success = true;
        for (int i = 0; i < N; i++) {
            if (h_data[i] != i) {
                success = false;
                break;
            }
        }
        std::cout << (success ? "Data received correctly!\n" : "Data transfer error!\n");
    }

    // Cleanup
    cudaFree(d_send);
    cudaFree(d_recv);
    MPI_Finalize();

    return 0;
}

