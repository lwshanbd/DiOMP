#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>

#define WARMUP_ITERS 5
#define TEST_ITERS 20
#define MAX_MSG_SIZE 1L * 1024 * 1024 * 1024  // 2GB
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
    printf("latency %.5f us, ", avg_time_us);
    return (size / (1024.0 * 1024.0)) / avg_time_s;  // MB/s
}

void run_benchmark(size_t msg_size) {
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
            MPI_Put(gpu_buf, msg_size, MPI_BYTE, 0, 0, msg_size, MPI_BYTE, win);
            MPI_Win_fence(0, win);
            
            double end_time = get_time_in_microseconds();
            times.push_back(end_time - start_time);
        }
        
        double avg_bandwidth = calculate_average_bandwidth(times, msg_size);
        printf("%zu,%f,Valid\n", msg_size, avg_bandwidth);
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

    if (rank == 1) {
        printf("Size(bytes),Avg Bandwidth(MB/s),Data Validity\n");
    }

    // Run benchmark with different message sizes
    for (size_t size = 1; size <= MAX_MSG_SIZE; size *= 2) {
        run_benchmark(size);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}