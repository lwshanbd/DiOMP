#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>

static constexpr int ITERATIONS = 100;
static constexpr size_t MAX_DATA_SIZE = 256 * 1024 * 1024;

int main(int argc, char **argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Test data sizes from 8 bytes to 256MB, doubling each time
    for (size_t data_size = 1; data_size <= MAX_DATA_SIZE / sizeof(double); data_size *= 2) {
        double *gpu_data = new double[data_size];
        
        // Allocate GPU memory using OpenMP target offloading
        #pragma omp target enter data map(alloc: gpu_data[0:data_size])

        // Initialize data on root process
        if (rank == 0) {
            #pragma omp target teams distribute parallel for
            for (size_t i = 0; i < data_size; ++i) {
                gpu_data[i] = 1.0;
            }
        }
        
        // Synchronize all processes before timing
        MPI_Barrier(MPI_COMM_WORLD);

        // Benchmark broadcast performance
        double total_time = 0.0;
        for (int iter = 0; iter < ITERATIONS; ++iter) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // Perform GPU-aware MPI broadcast
            MPI_Bcast(gpu_data, data_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            total_time += elapsed.count();
        }

        // Output results from root process
        if (rank == 0) {
            double avg_time = total_time / ITERATIONS;
            double data_size_bytes = data_size * sizeof(double);
            double bandwidth = (data_size_bytes / (1024.0 * 1024.0)) / avg_time; // MB/s
            
            // Print header for first iteration
            if (data_size == 1) {
                std::cout << "Size(bytes),Avg Time(ms),Bandwidth(MB/s)\n";
            }
            
            std::cout << static_cast<size_t>(data_size_bytes) << ","
                      << avg_time * 1000.0 << ","
                      << bandwidth << "\n";
        }

        // Clean up GPU memory
        #pragma omp target exit data map(delete: gpu_data[0:data_size])
        delete[] gpu_data;
    }
    
    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}