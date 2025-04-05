#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "benchmark.cuh"

// Sample kernels to demonstrate the template use

// Example kernel for reduction (sum)
__global__ void reduceSharedSumKernel(const float *a, float *b, size_t n) {
    uint tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + tid;
    
    extern __shared__ float sdata[]; // Should be equal to the total number of threads.
    
    sdata[tid] = (i < n) ? a[i] : 0.0f;
    __syncthreads();
    
    for (uint offset = blockDim.x / 2; offset > 0 && tid < offset; offset >>= 1) {
        sdata[tid] += sdata[tid + offset];
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        b[blockIdx.x] = sdata[0];
    }
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
    float* h_vec_output = new float[n];
    
    // Initialize input data
    for (size_t i = 0; i < n; i++) {
        h_input[i] = 1.0f;  // All ones for easy verification
    }
    
    // Allocate device memory
    float *d_input, *d_output, *d_vec_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMalloc(&d_vec_output, bytes));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    // Create a benchmark object
    Benchmark benchmark;
    
    // First run: Reduction kernel
    const int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    size_t smemSize = blockSize * sizeof(float);
    
    std::cout << "\nRunning reduction kernel examples...\n";
    
    // Setup throughput calculators for different kernel types
    ReductionThroughputData reductionCalc;
    reductionCalc.dataSize = bytes;
    
    ElementWiseThroughputData vecAddCalc(3); // 3 arrays: 2 input, 1 output
    vecAddCalc.dataSize = bytes;
    
    // Run and verify basic reduction
    std::cout << "\n--- CORRECTNESS VERIFICATION ---\n";
        
    // Clear output buffer
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));
    
    // Run shared memory reduction kernel
    launchKernel<decltype(reduceSharedSumKernel)>(
        reduceSharedSumKernel,
        gridSize, blockSize, smemSize, 0,
        "reduceSharedSumKernel", true,
        d_input, d_output, n
    );
    
    // Copy results back to verify
    CUDA_CHECK(cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify shared memory reduction results
    verifyReductionResults(h_output, h_input, n, gridSize, "reduceSharedSumKernel");
    
    // Clear output buffer
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));
    
    // Run shuffle-based reduction kernel
    launchKernel<decltype(reduceShuffleKernel)>(
        reduceShuffleKernel,
        gridSize, blockSize, blockSize * sizeof(float) / 32, 0,
        "reduceShuffleKernel", true,
        d_input, d_output, n
    );
    
    // Copy results back to verify
    CUDA_CHECK(cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify shuffle reduction results
    verifyReductionResults(h_output, h_input, n, gridSize, "reduceShuffleKernel");
    
    // Clear output buffer
    CUDA_CHECK(cudaMemset(d_vec_output, 0, bytes));
    
    // Run vector addition kernel
    launchKernel<decltype(vecAddKernel)>(
        vecAddKernel,
        gridSize, blockSize, 0, 0,
        "vecAddKernel", true,
        d_input, d_input, d_vec_output, n
    );
    
    // Copy results back to verify
    CUDA_CHECK(cudaMemcpy(h_vec_output, d_vec_output, bytes, cudaMemcpyDeviceToHost));
    
    // Verify vector add results
    verifyVectorAddResults(h_vec_output, h_input, h_input, n, "vecAddKernel");
    
    std::cout << "\n--- PERFORMANCE BENCHMARKING ---\n";
    
    // Run shared memory reduction benchmark
    benchmark.run("Shared Memory Reduction", [&]() {
        launchKernel<decltype(reduceSharedSumKernel)>(
            reduceSharedSumKernel,
            gridSize, blockSize, smemSize, 0,
            "reduceSharedSumKernel", false,
            d_input, d_output, n
        );
    }, reductionCalc, 5, true);
    
    // Set this as the baseline for speedup comparisons
    benchmark.setBaseline("Shared Memory Reduction");
    
    // Run shuffle-based reduction benchmark
    benchmark.run("Shuffle Reduction", [&]() {
        launchKernel<decltype(reduceShuffleKernel)>(
            reduceShuffleKernel,
            gridSize, blockSize, blockSize * sizeof(float) / 32, 0,
            "reduceShuffleKernel", false,
            d_input, d_output, n
        );
    }, reductionCalc, 5, true);
    
    // Vector addition (just as another example)
    benchmark.run("Vector Addition", [&]() {
        launchKernel<decltype(vecAddKernel)>(
            vecAddKernel,
            gridSize, blockSize, 0, 0,
            "vecAddKernel", false,
            d_input, d_input, d_vec_output, n
        );
    }, vecAddCalc, 5, true);
    
    // Print benchmark results
    benchmark.printResults(true, true);
    
    // Free memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_vec_output));
    delete[] h_input;
    delete[] h_output;
    delete[] h_vec_output;
    
    // Reset device
    CUDA_CHECK(cudaDeviceReset());
    
    return 0;
}
