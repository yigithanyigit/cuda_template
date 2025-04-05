#pragma once

#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

class CPUTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool running;

public:
    CPUTimer() : running(false) {}

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }

    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
        running = false;
    }

    // Returns elapsed time in milliseconds
    float elapsed() {
        auto duration = (running ? 
                         std::chrono::high_resolution_clock::now() - start_time : 
                         end_time - start_time);
        return std::chrono::duration<float, std::milli>(duration).count();
    }
};

// GPU Event timer
class GPUTimer {
private:
    cudaEvent_t start_event, stop_event;

public:
    GPUTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    ~GPUTimer() {
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_event, 0));
    }

    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
    }

    // Returns elapsed time in milliseconds
    float elapsed() {
        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start_event, stop_event));
        return time;
    }
};

template<typename KernelFunc, typename... Args>
void launchKernel(
    KernelFunc kernel,
    dim3 gridDim,
    dim3 blockDim,
    size_t sharedMemBytes = 0,
    cudaStream_t stream = 0,
    const char* kernelName = "unnamed_kernel",
    bool timeit = false,
    Args&&... args)
{
    if (timeit) {
        GPUTimer timer;
        timer.start();
        kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(std::forward<Args>(args)...);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        printf("Kernel '%s' execution time: %.3f ms\n", kernelName, timer.elapsed());
    } else {
        kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(std::forward<Args>(args)...);
    }
    
    CUDA_CHECK(cudaGetLastError());
}
