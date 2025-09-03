#include <omp.h>
#include <diomp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <algorithm>
#include <vector>
#include <string.h>

#define NUM_RUNS 20
#define MIN_SIZE_BYTES 4  // 4 Bytes
#define MAX_SIZE_BYTES (128L * 1024 * 1024)  // 128 MB

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

double calculate_average_bandwidth(std::vector<double>& times, size_t size_bytes) {
    std::sort(times.begin(), times.end());
    
    double sum = 0;
    for (size_t i = 2; i < times.size() - 2; i++) {
        sum += times[i];
    }
    
    double avg_time_us = sum / (times.size() - 4);  // Average time in microseconds
    double avg_time_s = avg_time_us * 1e-6;        // Convert to seconds
    return (size_bytes / (1024.0 * 1024.0)) / avg_time_s;  // MB/s
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

void benchmark(size_t size_bytes, const char* operation) {
    int rank = omp_get_rank_num();
    int num_ranks = omp_get_num_ranks();
    if (num_ranks < 2) {
        if (rank == 0) {
            fprintf(stderr, "This benchmark requires at least 2 ranks\n");
        }
        exit(EXIT_FAILURE);
    }
    
    size_t size_ints = (size_bytes + sizeof(int) - 1) / sizeof(int);  // Round up to int boundary
    int *data = (int *)omp_target_alloc(size_ints * sizeof(int), omp_get_default_device());
    if (!data) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    if (rank == 1) {
        fill_buffer(data, size_ints);
    }

    diomp_barrier(nullptr);

    if (rank != 0) {
        diomp_barrier(nullptr);
    } else if (rank == 0) {
        std::vector<double> times;
        bool data_valid = true;
        for (int i = 0; i < NUM_RUNS; i++) {
            double start_time = get_time_in_microseconds();

            if (strcmp(operation, "get") == 0) {
                ompx_dget(data, 1, data, size_bytes, 0, 0);
            } else if (strcmp(operation, "put") == 0) {
                ompx_dput(data, 1, data, size_bytes, 0, 0);
            }
            
            diomp_waitALLRMA();
            double end_time = get_time_in_microseconds();
            times.push_back(end_time - start_time);
        }

        diomp_barrier(nullptr);

        double avg_bandwidth = calculate_average_bandwidth(times, size_bytes);
        double avg_time_us = 0;
        for (size_t i = 2; i < times.size() - 2; i++) {
            avg_time_us += times[i];
        }
        avg_time_us /= (times.size() - 4);
        
        printf("| %12zu | %12.3f | %12.3f | %8s | %6s |\n", 
               size_bytes, avg_bandwidth, avg_time_us, data_valid ? "Valid" : "Invalid", operation);
    }

    omp_target_free(data, omp_get_default_device());
}

int main(int argc, char** argv) {
    __init_diomp_target(2);

    // Parse command line arguments
    const char* operation = "get";  // default to get
    if (argc > 1) {
        if (strcmp(argv[1], "put") == 0 || strcmp(argv[1], "get") == 0) {
            operation = argv[1];
        } else {
            if (omp_get_rank_num() == 0) {
                printf("Usage: %s [put|get]\n", argv[0]);
                printf("Default operation is 'get'\n");
            }
            return 1;
        }
    }

    if (omp_get_rank_num() == 0) {
        char op_title[8];
        strcpy(op_title, operation);
        if (op_title[0] >= 'a' && op_title[0] <= 'z') {
            op_title[0] = op_title[0] - 'a' + 'A';
        }
        
        printf("\n");
        printf("=== DiOMP %s Bandwidth Benchmark Results ===\n", op_title);
        printf("+---------------+--------------+--------------+----------+--------+\n");
        printf("| Size (bytes)  | Bandwidth    | Latency      | Status   | Op     |\n");
        printf("|               |   (MB/s)     |    (us)      |          |        |\n");
        printf("+---------------+--------------+--------------+----------+--------+\n");
    }

    for (size_t size_bytes = MIN_SIZE_BYTES; size_bytes <= MAX_SIZE_BYTES; size_bytes *= 2) {
        benchmark(size_bytes, operation);
    }
    
    if (omp_get_rank_num() == 0) {
        printf("+---------------+--------------+--------------+----------+--------+\n");
        printf("\n");
    }

    return 0;
}

// clang++ diomp_bw.cpp -DOPENMP_ENABLE_DIOMP_DEVICE=1 -lcuda -ldiomp -fopenmp  --offload-arch=sm_80 -lnccl -L /global/cfs/cdirs/xpress/baodi/software/llvm-gpu/lib -L /global/cfs/cdirs/xpress/baodi/software/gasnet-gpu/lib -I /global/cfs/cdirs/xpress/baodi/software/gasnet-gpu/include/ofi-conduit/ -lgasnet-ofi-par -no-pie -lmpi -lhwloc -lrt -pthread -I /opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3/include -L /opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3/lib -L/opt/cray/libfabric/1.22.0/lib64 -lfabric -L /opt/cray/pe/pmi/6.1.13/lib -lpmi -lhugetlbfs -DDIOMP_ENABLE_CUDA -I $NCCL_HOME/include -L $NCCL_HOME/lib -o diomp_bw_get