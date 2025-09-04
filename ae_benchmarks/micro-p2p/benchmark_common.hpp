#pragma once

#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>

namespace benchmark {

// Benchmark configuration
constexpr int WARMUP_ITERS = 5;
constexpr int TEST_ITERS = 20;
constexpr size_t MIN_SIZE_BYTES = 4;  // 4 Bytes
constexpr size_t MAX_SIZE_BYTES = 128L * 1024 * 1024;  // 128 MB

// Data validation
constexpr int VALIDATION_PATTERN = 0xDEADBEEF;

// Performance measurement
double calculate_average_bandwidth(std::vector<double>& times, size_t size_bytes) {
    if (times.size() < 5) {
        return 0.0;
    }

    // Sort times and remove outliers (2 highest and 2 lowest)
    std::sort(times.begin(), times.end());
    double sum = 0;
    for (std::size_t i = 2; i < times.size() - 2; i++) {
        sum += times[i];
    }
    
    double avg_time_us = sum / (times.size() - 4);  // Average time in microseconds
    double avg_time_s = avg_time_us * 1e-6;         // Convert to seconds
    return (size_bytes / (1024.0 * 1024.0)) / avg_time_s;  // MB/s
}

// Result printing
void print_header(const char* framework, const char* operation) {
    char op_title[8];
    strncpy(op_title, operation, sizeof(op_title) - 1);
    op_title[sizeof(op_title) - 1] = '\0';
    if (op_title[0] >= 'a' && op_title[0] <= 'z') {
        op_title[0] = op_title[0] - 'a' + 'A';
    }
    
    printf("\n");
    printf("=== %s %s Bandwidth Benchmark Results ===\n", framework, op_title);
    printf("+---------------+--------------+--------------+----------+--------+\n");
    printf("| Size (bytes)  | Bandwidth    | Latency      | Status   | Op     |\n");
    printf("|               |   (MB/s)     |    (us)      |          |        |\n");
    printf("+---------------+--------------+--------------+----------+--------+\n");
}

void print_result(size_t size_bytes, double bandwidth, double latency, bool is_valid, const char* operation) {
    printf("| %12zu | %12.3f | %12.3f | %8s | %6s |\n", 
           size_bytes, bandwidth, latency, is_valid ? "Valid" : "Invalid", operation);
}

void print_footer() {
    printf("+---------------+--------------+--------------+----------+--------+\n");
    printf("\n");
}

// Buffer validation
template<typename T>
void fill_validation_pattern(T* buffer, size_t count) {
    #pragma omp target teams distribute parallel for
    for (size_t i = 0; i < count; i++) {
        buffer[i] = VALIDATION_PATTERN;
    }
}

template<typename T>
bool verify_buffer(T* buffer, size_t count) {
    bool is_valid = true;
    #pragma omp target teams distribute parallel for reduction(&:is_valid)
    for (size_t i = 0; i < count; i++) {
        if (buffer[i] != VALIDATION_PATTERN) {
            is_valid = false;
        }
    }
    return is_valid;
}

} // namespace benchmark
