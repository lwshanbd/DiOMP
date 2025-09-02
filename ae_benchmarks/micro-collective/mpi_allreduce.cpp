#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>

#define ITERATIONS 100 // Number of test iterations

int main(int argc, char **argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Data size from 4 bytes to 128MB, doubling each time
    for (size_t data_size = 1; data_size <= (256 * 1024 * 1024) / sizeof(double); data_size *= 2) {
        double *gpu_src = new double[data_size];
        double *gpu_dst = new double[data_size];

        // Allocate GPU memory using OpenMP target offloading
        #pragma omp target enter data map(alloc: gpu_src[0:data_size], gpu_dst[0:data_size])

        if (rank == 0) {
            // Initialize data using OpenMP
            #pragma omp target teams distribute parallel for
            for (size_t i = 0; i < data_size; ++i) {
                gpu_src[i] = 1.0;
                gpu_dst[i] = 0.0;
            }
        }

        // Synchronize to ensure all processes are ready
        MPI_Barrier(MPI_COMM_WORLD);

        // Measure Allreduce performance
        double total_time = 0.0;
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();

            // Use CUDA-aware MPI Allreduce operation
            MPI_Allreduce(gpu_src, gpu_dst, data_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            total_time += elapsed.count();
        }

        // Calculate average Allreduce rate
        if (rank == 0) {
            double avg_time = total_time / ITERATIONS;
            double bandwidth = (data_size * sizeof(double) / (1024.0 * 1024.0)) / avg_time; // MB/s
            if (data_size == 1) {
                std::cout << "Size(bytes),Avg Time(ms),Bandwidth(MB/s)\n";
            }
            std::cout << data_size * sizeof(double) << "," 
                      << avg_time * 1e3 << "," 
                      << bandwidth << "\n";
        }

        // Clean up GPU memory
        #pragma omp target exit data map(delete: gpu_src[0:data_size], gpu_dst[0:data_size])

        delete[] gpu_src;
        delete[] gpu_dst;
    }

    // Finalize MPI environment
    MPI_Finalize();

    return 0;
}
// CC -fopenmp -target-accel=nvidia80 mpi_allreduce.cpp -o mpi_allreduce