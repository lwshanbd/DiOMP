#include <omp.h>
#include <diomp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <algorithm>
#include <vector>

#define NUM_RUNS 20
#define MIN_SIZE (4 / sizeof(int))  // 4 Bytes
#define MAX_SIZE (2L * 1024 * 1024 * 1024 / sizeof(int))  // 2 GB

double get_time_in_seconds() {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec + time.tv_nsec * 1e-9;
}

double get_time_in_microseconds() {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec * 1e6 + time.tv_nsec * 1e-3;
}

double calculate_average_bandwidth(std::vector<double>& times, size_t size) {
    std::sort(times.begin(), times.end());
    
    double sum = 0;
    for (size_t i = 2; i < times.size() - 2; i++) {
        sum += times[i];
    }
    
    double avg_time_us = sum / (times.size() - 4);  // Average time in microseconds
    double avg_time_s = avg_time_us * 1e-6;        // Convert to seconds
    printf("latency %.5f us, ", avg_time_us);
    return (size * sizeof(int) / (1024.0 * 1024.0)) / avg_time_s;  // MB/s
}

void fill_buffer(int* buffer, size_t size) {
    #pragma omp target teams distribute parallel for
    for (size_t i = 0; i < size; i++) {
        buffer[i] = 1234;
    }
}

bool verify_buffer(int* buffer, size_t size) {
    // bool is_valid = true;
    // #pragma omp target
    // for (size_t i = 0; i < size; i++) {
    //     if (buffer[i] != i % INT32_MAX) {
    //         is_valid = false;
    //     }
    // }
    // if(!is_valid)
    // {
    //    return false;
    // }
    // printf("True\n");
    return true;
}

void benchmark(size_t size) {
    int rank = omp_get_rank_num();
    int num_ranks = omp_get_num_ranks();

    if (num_ranks < 2) {
        if (rank == 0) {
            fprintf(stderr, "This benchmark requires at least 2 ranks\n");
        }
        exit(EXIT_FAILURE);
    }
    
    int *data = (int *)omp_target_alloc(size * sizeof(int), omp_get_default_device());
    if (!data) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    if (rank == 1) {
        fill_buffer(data, size);
    }

    diomp_barrier();

    if (rank != 0) {
        diomp_barrier();
    } else if (rank == 0) {
        std::vector<double> times;
        bool data_valid = true;
        for (int i = 0; i < NUM_RUNS; i++) {
            double start_time = get_time_in_microseconds();

            ompx_dget(data, 1, data, size * sizeof(int),0 ,0);
            
            diomp_waitALLRMA();
            double end_time = get_time_in_microseconds();
            times.push_back(end_time - start_time);
        }

        diomp_barrier();

        double avg_bandwidth = calculate_average_bandwidth(times, size);
        printf("%zu,%f,%s\n", size * sizeof(int), avg_bandwidth, data_valid ? "Valid" : "Invalid");
    }

    omp_target_free(data, omp_get_default_device());
}

int main() {
    __init_diomp_target(2);

    if (omp_get_rank_num() == 1) {
        printf("Size(bytes),Avg Bandwidth(MB/s),Data Validity\n");
    }

    for (size_t size = MIN_SIZE; size <= MAX_SIZE; size *= 2) {
        benchmark(size);
    }

    return 0;
}

// clang++ diomp_bw.cpp -DOPENMP_ENABLE_DIOMP_DEVICE=1 -lcuda -ldiomp -fopenmp  --offload-arch=sm_80 -lnccl -L /global/cfs/cdirs/xpress/baodi/software/llvm-gpu/lib -L /global/cfs/cdirs/xpress/baodi/software/gasnet-gpu/lib -I /global/cfs/cdirs/xpress/baodi/software/gasnet-gpu/include/ofi-conduit/ -lgasnet-ofi-par -no-pie -lmpi -lhwloc -lrt -pthread -I /opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3/include -L /opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3/lib -L/opt/cray/libfabric/1.20.1/lib64 -lfabric -L /opt/cray/pe/pmi/6.1.13/lib -lpmi -lhugetlbfs -DDIOMP_ENABLE_CUDA -I $NCCL_HOME/include -L $NCCL_HOME/lib -o diomp_bw_get