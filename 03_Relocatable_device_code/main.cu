#include <iostream>
#include "device_functions.cuh"

__global__ void compute(float *d_result) {
    *d_result = multiply(3.0f, 4.0f);
}

int main() {
    float *d_result, h_result;
    cudaMalloc(&d_result, sizeof(float));
    
    compute<<<1, 1>>>(d_result);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Result: " << h_result << std::endl;
    cudaFree(d_result);
    return 0;
}

