#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cerrno>
#include "benchmark_common.hpp"
#include <mpi.h>
#include <omp.h>
#include <iostream>

using namespace benchmark;
using std::vector;

// Time measurement functions specific to MPI benchmark
double get_time_in_microseconds() {
    return MPI_Wtime() * 1e6;
}

void run_benchmark(std::size_t msg_size, const char* operation) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: This benchmark requires at least 2 processes\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate GPU memory using OpenMP target
    size_t size_ints = (msg_size + sizeof(int) - 1) / sizeof(int);  // Round up to int boundary
    int *gpu_buf = (int *)omp_target_alloc(size_ints * sizeof(int), 0);  // device_num = 0
    if (gpu_buf == NULL) {
        fprintf(stderr, "Error: Rank %d failed to allocate GPU memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Create MPI window
    MPI_Win win;
    MPI_Win_create(gpu_buf, msg_size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Warm-up iterations
    for (int i = 0; i < WARMUP_ITERS; i++) {
        MPI_Win_fence(0, win);
        if (rank == 1) {
            if (strcmp(operation, "put") == 0) {
                MPI_Put(gpu_buf, msg_size, MPI_BYTE, 0, 0, msg_size, MPI_BYTE, win);
            } else if (strcmp(operation, "get") == 0) {
                MPI_Get(gpu_buf, msg_size, MPI_BYTE, 0, 0, msg_size, MPI_BYTE, win);
            }
        }
        MPI_Win_fence(0, win);
    }

    // Actual benchmark
    std::vector<double> times;
    times.reserve(TEST_ITERS);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 1) {
        for (int i = 0; i < TEST_ITERS; i++) {
            double start_time = get_time_in_microseconds();
            
            MPI_Win_fence(0, win);
            if (strcmp(operation, "put") == 0) {
                MPI_Put(gpu_buf, msg_size, MPI_BYTE, 0, 0, msg_size, MPI_BYTE, win);
            } else if (strcmp(operation, "get") == 0) {
                MPI_Get(gpu_buf, msg_size, MPI_BYTE, 0, 0, msg_size, MPI_BYTE, win);
            }
            MPI_Win_fence(0, win);
            
            double end_time = get_time_in_microseconds();
            times.push_back(end_time - start_time);
        }

        // Calculate and print results
        double avg_bandwidth = calculate_average_bandwidth(times, msg_size);
        double avg_time_us = 0;
        for (size_t i = 2; i < times.size() - 2; i++) {
            avg_time_us += times[i];
        }
        avg_time_us /= (times.size() - 4);
        
        print_result(msg_size, avg_bandwidth, avg_time_us, true, operation);
    } else {
        for (int i = 0; i < TEST_ITERS; i++) {
            MPI_Win_fence(0, win);
            MPI_Win_fence(0, win);
        }
    }

    // Cleanup
    MPI_Win_free(&win);
    omp_target_free(gpu_buf, 0);
}

int main(int argc, char **argv) {
    // Initialize MPI with thread support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "Warning: MPI implementation does not support MPI_THREAD_MULTIPLE\n");
    }
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Parse command line arguments
    const char* operation = "put";  // default to put for MPI
    if (argc > 1) {
        if (strcmp(argv[1], "put") == 0 || strcmp(argv[1], "get") == 0) {
            operation = argv[1];
        } else {
            if (rank == 0) {
                fprintf(stderr, "Usage: %s [put|get]\n", argv[0]);
                fprintf(stderr, "Default operation is 'put'\n");
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Print benchmark header
    if (rank == 0) {
        print_header("MPI", operation);
    }

    // Run benchmark with different message sizes
    for (size_t size = MIN_SIZE_BYTES; size <= MAX_SIZE_BYTES; size *= 2) {
        run_benchmark(size, operation);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Print benchmark footer
    if (rank == 0) {
        print_footer();
    }

    MPI_Finalize();
    return 0;
}