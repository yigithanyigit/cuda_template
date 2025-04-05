#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <memory>
#include "cuda_utils.cuh"

struct BenchmarkResult {
    std::string name;
    float time_ms;
    float throughput_gb_s; // Optional throughput in GB/s
    float speedup;         // Optional speedup compared to baseline
};

class Benchmark {
private:
    std::vector<BenchmarkResult> results;
    std::string baselineName;

public:
    Benchmark() : baselineName("") {}

    template<typename Func, typename ThroughputCalculator>
    BenchmarkResult run(
        const std::string& name, 
        Func func, 
        ThroughputCalculator& calculator,
        int iterations = 10, 
        bool warmup = true) 
    {
        // Optional warmup run
        if (warmup) {
            func();
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        GPUTimer timer;
        timer.start();
        
        for (int i = 0; i < iterations; i++) {
            func();
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        
        float time_ms = timer.elapsed() / iterations;
        
        calculator.time_ms = time_ms;
        
        float throughput = calculator.calculate();
        
        BenchmarkResult result = {name, time_ms, throughput, 0.0f};
        results.push_back(result);
        
        return result;
    }
    
    template<typename Func>
    BenchmarkResult run(
        const std::string& name, 
        Func func, 
        size_t dataSize,
        int iterations = 10, 
        bool warmup = true) 
    {
        ThroughputData calculator;
        calculator.dataSize = dataSize;
        
        return run(name, func, calculator, iterations, warmup);
    }
    
    template<typename Func>
    BenchmarkResult run(
        const std::string& name, 
        Func func,
        int iterations = 10, 
        bool warmup = true) 
    {
        // Optional warmup run
        if (warmup) {
            func();
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        GPUTimer timer;
        timer.start();
        
        for (int i = 0; i < iterations; i++) {
            func();
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        
        float time_ms = timer.elapsed() / iterations;
        
        BenchmarkResult result = {name, time_ms, 0.0f, 0.0f};
        results.push_back(result);
        
        return result;
    }
    
    void setBaseline(const std::string& name) {
        baselineName = name;
    }
    
    void calculateSpeedups() {
        if (baselineName.empty() || results.empty()) return;
        
        float baselineTime = 0.0f;
        for (const auto& result : results) {
            if (result.name == baselineName) {
                baselineTime = result.time_ms;
                break;
            }
        }
        
        if (baselineTime == 0.0f) return;
        
        for (auto& result : results) {
            result.speedup = baselineTime / result.time_ms;
        }
    }
    
    void printResults(bool showThroughput = false, bool showSpeedup = false) {
        if (results.empty()) return;
        
        if (showSpeedup && !baselineName.empty()) {
            calculateSpeedups();
        }
        
        std::cout << "\n===== BENCHMARK RESULTS =====\n";
        std::cout << std::left << std::setw(25) << "Name" 
                  << std::right << std::setw(15) << "Time (ms)";
        
        if (showThroughput)
            std::cout << std::right << std::setw(18) << "Throughput (GB/s)";
        
        if (showSpeedup)
            std::cout << std::right << std::setw(12) << "Speedup";
        
        std::cout << "\n";
        std::cout << std::string(25 + 15 + (showThroughput ? 18 : 0) + (showSpeedup ? 12 : 0), '-') << "\n";
        
        for (const auto& result : results) {
            std::cout << std::left << std::setw(25) << result.name
                      << std::right << std::fixed << std::setprecision(3) << std::setw(15) << result.time_ms;
            
            if (showThroughput)
                std::cout << std::right << std::fixed << std::setprecision(2) << std::setw(18) << result.throughput_gb_s;
            
            if (showSpeedup) {
                if (result.name == baselineName)
                    std::cout << std::right << std::setw(12) << "1.00x";
                else
                    std::cout << std::right << std::fixed << std::setprecision(2) << std::setw(11) << result.speedup << "x";
            }
            
            std::cout << "\n";
        }
        
        std::cout << "=============================\n\n";
    }
    
    void clear() {
        results.clear();
        baselineName = "";
    }
};
