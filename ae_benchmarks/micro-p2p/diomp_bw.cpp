#include <ctime>
#include <cstring>
#include <vector>
#include <iostream>
#include <cerrno>
#include "benchmark_common.hpp"

// Include CUDA and OpenMP headers after C++ standard library headers
#include <omp.h>
#include <diomp.h>

using namespace benchmark;
using std::vector;

// Time measurement functions specific to DiOMP benchmark
double get_time_in_microseconds() {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec * 1e6 + time.tv_nsec * 1e-3;
}

void run_benchmark(std::size_t size_bytes, const char* operation) {
    int rank = omp_get_rank_num();
    int num_ranks = omp_get_num_ranks();
    if (num_ranks < 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: This benchmark requires at least 2 ranks\n");
        }
        exit(EXIT_FAILURE);
    }
    
    // Round up to int boundary for proper alignment
    size_t size_ints = (size_bytes + sizeof(int) - 1) / sizeof(int);
    int *data = (int *)omp_target_alloc(size_ints * sizeof(int), omp_get_default_device());
    if (!data) {
        fprintf(stderr, "Error: Failed to allocate GPU memory on rank %d\n", rank);
        exit(EXIT_FAILURE);
    }

    // Initialize data on rank 1
    if (rank == 1) {
        fill_validation_pattern(data, size_ints);
    }

    // Synchronize before starting benchmark
    ompx_barrier(nullptr);

    // Warmup phase
    for (int i = 0; i < WARMUP_ITERS; i++) {
        if (rank == 0) {
            if (strcmp(operation, "get") == 0) {
                ompx_dget(data, 1, data, size_bytes, 0, 0);
            } else if (strcmp(operation, "put") == 0) {
                ompx_dput(data, 1, data, size_bytes, 0, 0);
            }
            ompx_fence();
        }
        ompx_barrier(nullptr);
    }

    // Benchmark phase
    if (rank != 0) {
        ompx_barrier(nullptr);
    } else {
        std::vector<double> times;
        times.reserve(TEST_ITERS);
        
        for (int i = 0; i < TEST_ITERS; i++) {
            double start_time = get_time_in_microseconds();

            if (strcmp(operation, "get") == 0) {
                ompx_dget(data, 1, data, size_bytes, 0, 0);
            } else if (strcmp(operation, "put") == 0) {
                ompx_dput(data, 1, data, size_bytes, 0, 0);
            }
            
            ompx_fence();
            double end_time = get_time_in_microseconds();
            times.push_back(end_time - start_time);
        }

        ompx_barrier(nullptr);

        // Verify data integrity
        bool data_valid = verify_buffer(data, size_ints);

        // Calculate and print results
        double avg_bandwidth = calculate_average_bandwidth(times, size_bytes);
        double avg_time_us = 0;
        for (size_t i = 2; i < times.size() - 2; i++) {
            avg_time_us += times[i];
        }
        avg_time_us /= (times.size() - 4);
        
        print_result(size_bytes, avg_bandwidth, avg_time_us, data_valid, operation);
    }

    omp_target_free(data, omp_get_default_device());
}

int main(int argc, char** argv) {
    // Initialize DiOMP with 2 GPUs
    __init_diomp_target(2);

    // Parse command line arguments
    const char* operation = "get";  // default to get for DiOMP
    if (argc > 1) {
        if (strcmp(argv[1], "put") == 0 || strcmp(argv[1], "get") == 0) {
            operation = argv[1];
        } else {
            if (omp_get_rank_num() == 0) {
                fprintf(stderr, "Usage: %s [put|get]\n", argv[0]);
                fprintf(stderr, "Default operation is 'get'\n");
            }
            return 1;
        }
    }

    // Print benchmark header
    if (omp_get_rank_num() == 0) {
        print_header("DiOMP", operation);
    }

    // Run benchmark with increasing message sizes
    for (size_t size_bytes = MIN_SIZE_BYTES; size_bytes <= MAX_SIZE_BYTES; size_bytes *= 2) {
        run_benchmark(size_bytes, operation);
    }
    
    // Print benchmark footer
    if (omp_get_rank_num() == 0) {
        print_footer();
    }

    return 0;
}