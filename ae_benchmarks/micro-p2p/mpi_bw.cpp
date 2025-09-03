#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>

#define WARMUP_ITERS 5
#define TEST_ITERS 20
#define MIN_SIZE_BYTES 4  // 4 Bytes
#define MAX_SIZE_BYTES (128L * 1024 * 1024)  // 128 MB
#define WARMUP_RUNS 5

double get_time_in_microseconds() {
    return MPI_Wtime() * 1e6;
}

double calculate_average_bandwidth(std::vector<double>& times, size_t size) {
    std::sort(times.begin(), times.end());
    
    double sum = 0;
    for (size_t i = 2; i < times.size() - 2; i++) {
        sum += times[i];
    }
    
    double avg_time_us = sum / (times.size() - 4);  // Average time in microseconds
    double avg_time_s = avg_time_us * 1e-6;         // Convert to seconds
    return (size / (1024.0 * 1024.0)) / avg_time_s;  // MB/s
}

void run_benchmark(size_t msg_size, const char* operation) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // if (size != 2) {
    //     if (rank == 0) printf("This benchmark requires exactly 2 processes\n");
    //     MPI_Abort(MPI_COMM_WORLD, 1);
    // }

    // Allocate GPU memory using OpenMP target
    void *gpu_buf = omp_target_alloc(msg_size, 0);  // device_num = 0
    if (gpu_buf == NULL) {
        printf("Rank %d: Failed to allocate GPU memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize memory on GPU
    // #pragma omp target teams distribute parallel for is_device_ptr(gpu_buf)
    // for (size_t i = 0; i < msg_size; i++) {
    //     ((char*)gpu_buf)[i] = (char)(i & 0xFF);
    // }

    // Create MPI window
    MPI_Win win;
    MPI_Win_create(gpu_buf, msg_size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Warm-up iterations
    for (int i = 0; i < WARMUP_ITERS; i++) {
        MPI_Win_fence(0, win);
        if (rank == 1) {
            MPI_Put(gpu_buf, msg_size, MPI_BYTE, 0, 0, msg_size, MPI_BYTE, win);
        }
        MPI_Win_fence(0, win);
    }

    // Actual benchmark
    std::vector<double> times;
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
        
        double avg_bandwidth = calculate_average_bandwidth(times, msg_size);
        double avg_time_us = 0;
        for (size_t i = 2; i < times.size() - 2; i++) {
            avg_time_us += times[i];
        }
        avg_time_us /= (times.size() - 4);
        
        printf("| %12zu | %12.3f | %12.3f | %8s | %6s |\n", 
               msg_size, avg_bandwidth, avg_time_us, "Valid", operation);
    } else {
        for (int i = 0; i < TEST_ITERS; i++) {
            MPI_Win_fence(0, win);
            MPI_Win_fence(0, win);
        }
    }

    // Cleanup
    MPI_Win_free(&win);
}

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Parse command line arguments
    const char* operation = "put";  // default to put for MPI
    if (argc > 1) {
        if (strcmp(argv[1], "put") == 0 || strcmp(argv[1], "get") == 0) {
            operation = argv[1];
        } else {
            if (rank == 0) {
                printf("Usage: %s [put|get]\n", argv[0]);
                printf("Default operation is 'put'\n");
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        char op_title[8];
        strcpy(op_title, operation);
        if (op_title[0] >= 'a' && op_title[0] <= 'z') {
            op_title[0] = op_title[0] - 'a' + 'A';
        }
        
        printf("\n");
        printf("=== MPI %s Bandwidth Benchmark Results ===\n", op_title);
        printf("+---------------+--------------+--------------+----------+--------+\n");
        printf("| Size (bytes)  | Bandwidth    | Latency      | Status   | Op     |\n");
        printf("|               |   (MB/s)     |    (us)      |          |        |\n");
        printf("+---------------+--------------+--------------+----------+--------+\n");
    }

    // Run benchmark with different message sizes
    for (size_t size = MIN_SIZE_BYTES; size <= MAX_SIZE_BYTES; size *= 2) {
        run_benchmark(size, operation);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        printf("+---------------+--------------+--------------+----------+--------+\n");
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}

// clang++ mpi_bw.cpp -lcuda -fopenmp  --offload-arch=sm_80 -lnccl -L /global/cfs/cdirs/xpress/baodi/software/llvm-gpu/lib -L /global/cfs/cdirs/xpress/baodi/software/gasnet-gpu/lib -I /global/cfs/cdirs/xpress/baodi/software/gasnet-gpu/include/ofi-conduit/ -lgasnet-ofi-par -no-pie -lmpi -lhwloc -lrt -pthread -I /opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3/include -L /opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3/lib -L/opt/cray/libfabric/1.22.0/lib64 -lfabric -L /opt/cray/pe/pmi/6.1.13/lib -lpmi -lhugetlbfs -DDIOMP_ENABLE_CUDA -I $NCCL_HOME/include -L $NCCL_HOME/lib -o mpi_bw_get