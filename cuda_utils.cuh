#pragma once

#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

struct ThroughputData {
    size_t dataSize;    // Size in bytes
    float time_ms;      // Time in milliseconds
    
    virtual float calculate() const {
        return (dataSize / (1024.0f * 1024.0f * 1024.0f)) / (time_ms / 1000.0f);
    }
};

struct ReductionThroughputData : public ThroughputData {
};

struct ElementWiseThroughputData : public ThroughputData {
    int numArraysAccessed;
    
    ElementWiseThroughputData(int numArrays = 2) : numArraysAccessed(numArrays) {}
    
    float calculate() const override {
        return (numArraysAccessed * dataSize / (1024.0f * 1024.0f * 1024.0f)) / (time_ms / 1000.0f);
    }
};

// Simple timer for CPU measurements
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

template<typename T>
T computeSumGroundTruth(const T* data, size_t n) {
    T sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

template<typename T>
void computeVecAddGroundTruth(const T* a, const T* b, T* result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

template<typename T>
bool verifyReductionResults(T* gpu_result, const T* input, size_t n, int numBlocks, const char* kernelName) {
    T cpuSum = computeSumGroundTruth(input, n);
    
    T gpuSum = 0;
    for (int i = 0; i < numBlocks; i++) {
        gpuSum += gpu_result[i];
    }
    
    bool correct = true;
    if constexpr (std::is_floating_point_v<T>) {
        float relError = std::abs((cpuSum - gpuSum) / cpuSum);
        correct = relError < 1e-5;
        if (!correct) {
            printf("Kernel '%s' validation failed! CPU sum: %f, GPU sum: %f, relative error: %e\n", 
                   kernelName, (float)cpuSum, (float)gpuSum, relError);
        } else {
            printf("Kernel '%s' validation passed. Sum: %f\n", kernelName, (float)gpuSum);
        }
    } else {
        correct = cpuSum == gpuSum;
        if (!correct) {
            printf("Kernel '%s' validation failed! CPU sum: %lld, GPU sum: %lld\n", 
                   kernelName, (long long)cpuSum, (long long)gpuSum);
        } else {
            printf("Kernel '%s' validation passed. Sum: %lld\n", kernelName, (long long)gpuSum);
        }
    }
    
    return correct;
}

template<typename T>
bool verifyVectorAddResults(const T* gpu_result, const T* a, const T* b, size_t n, const char* kernelName) {
    T* cpu_result = new T[n];
    
    computeVecAddGroundTruth(a, b, cpu_result, n);
    
    bool correct = true;
    size_t errorCount = 0;
    for (size_t i = 0; i < n && errorCount < 10; i++) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::abs(gpu_result[i] - cpu_result[i]) > 1e-5) {
                if (errorCount == 0) {
                    printf("Vector add validation failed at index %zu! GPU: %f, CPU: %f\n", 
                           i, (float)gpu_result[i], (float)cpu_result[i]);
                }
                correct = false;
                errorCount++;
            }
        } else {
            if (gpu_result[i] != cpu_result[i]) {
                if (errorCount == 0) {
                    printf("Vector add validation failed at index %zu! GPU: %lld, CPU: %lld\n", 
                           i, (long long)gpu_result[i], (long long)cpu_result[i]);
                }
                correct = false;
                errorCount++;
            }
        }
    }
    
    if (correct) {
        printf("Kernel '%s' validation passed.\n", kernelName);
    } else {
        printf("Kernel '%s' validation failed! Found %zu errors.\n", kernelName, errorCount);
    }
    
    delete[] cpu_result;
    return correct;
}
