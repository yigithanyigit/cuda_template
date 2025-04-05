#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "benchmark.cuh"

// Sample kernels to demonstrate the template use

// Example kernel for reduction (sum)
template<unsigned int blockSize>
__global__ void reduceKernel(const float* input, float* output, size_t n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize * 2 + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    
    // Load and perform first level of reduction
    float sum = 0.0f;
    while (i < n) {
        sum += input[i];
        if (i + blockSize < n)
            sum += input[i + blockSize];
        i += gridSize;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction within the block
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    
    // Last 64 threads handled differently (to avoid unnecessary __syncthreads)
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
    }
    
    // Write result for this block
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Example kernel using shuffle instructions (warp-level reduction)
__global__ void reduceShuffleKernel(const float* input, float* output, size_t n) {
    const unsigned int blockSize = blockDim.x;
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize + tid;
    unsigned int lane = tid % 32;
    unsigned int warpId = tid / 32;
    unsigned int warpsPerBlock = blockSize / 32;
    
    // Load data
    float sum = (i < n) ? input[i] : 0.0f;
    
    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread in each warp writes result to shared memory
    if (lane == 0) {
        sdata[warpId] = sum;
    }
    
    __syncthreads();
    
    // Final reduction across warps (only use first warp)
    if (warpId == 0) {
        sum = (tid < warpsPerBlock) ? sdata[lane] : 0.0f;
        
        // Warp-level reduction again
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // First thread writes result
        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

// Simple vector addition kernel
__global__ void vecAddKernel(const float* a, const float* b, float* c, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using device: " << prop.name << std::endl;
    
    // Problem size
    const size_t n = 1 << 24;  // 16M elements
    const size_t bytes = n * sizeof(float);
    std::cout << "Array size: " << n << " elements (" << bytes / (1024 * 1024) << " MB)" << std::endl;
    
    // Allocate host memory
    float* h_input = new float[n];
    float* h_output = new float[n];
    
    // Initialize input data
    for (size_t i = 0; i < n; i++) {
        h_input[i] = 1.0f;  // All ones for easy verification
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    // Create a benchmark object
    Benchmark benchmark;
    
    // First run: Reduction kernel
    const int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    size_t smemSize = blockSize * sizeof(float);
    
    std::cout << "\nRunning reduction kernel examples...\n";
    
    // Run basic reduction benchmark
    benchmark.run("Basic Reduction", [&]() {
        launchKernel<decltype(reduceKernel<blockSize>)>(
            reduceKernel<blockSize>,
            gridSize, blockSize, smemSize, 0,
            "reduceKernel", false,
            d_input, d_output, n
        );
    }, 5, bytes, true);
    
    // Set this as the baseline for speedup comparisons
    benchmark.setBaseline("Basic Reduction");
    
    // Run shuffle-based reduction benchmark
    benchmark.run("Shuffle Reduction", [&]() {
        launchKernel<decltype(reduceShuffleKernel)>(
            reduceShuffleKernel,
            gridSize, blockSize, blockSize * sizeof(float) / 32, 0,
            "reduceShuffleKernel", false,
            d_input, d_output, n
        );
    }, 5, bytes, true);
    
    // Vector addition (just as another example)
    benchmark.run("Vector Addition", [&]() {
        launchKernel<decltype(vecAddKernel)>(
            vecAddKernel,
            gridSize, blockSize,
            0, 0, "vecAddKernel", false,
            d_input, d_input, d_output, n
        );
    }, 5, bytes * 3, true);
    
    // Print benchmark results
    benchmark.printResults(true, true);
    
    // Free memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    delete[] h_input;
    delete[] h_output;
    
    // Reset device
    CUDA_CHECK(cudaDeviceReset());
    
    return 0;
}
