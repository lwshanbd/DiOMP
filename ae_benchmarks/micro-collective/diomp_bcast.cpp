#include <cstdio>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <diomp.h>

static constexpr int ITERATIONS = 100;
static constexpr size_t MAX_DATA_SIZE = 256 * 1024 * 1024;

int main() {
    // Initialize DiOMP runtime
    __init_diomp_target(1);
    
    int rank = omp_get_rank_num();
    int size = omp_get_num_ranks();
    int devices_num = omp_get_num_devices();
    // Test data sizes from 8 bytes to 256MB, doubling each time
    for (size_t data_size = 1; data_size <= MAX_DATA_SIZE / sizeof(double); data_size *= 2) {
        double *gpu_data = new double[data_size];
        
        // Allocate data on all available devices
        for (int device_id = 0; device_id < devices_num; device_id++) {
            #pragma omp target enter data map(to:gpu_data[0:data_size]) device(device_id)
            {}
        }

        // Initialize data on root process
        if (rank == 0) {
            #pragma omp target teams distribute parallel for
            for (size_t i = 0; i < data_size; ++i) {
                gpu_data[i] = 1.0;
            }
        }
        
        // Synchronize all processes before timing
        diomp_barrier();

        // Benchmark broadcast performance
        double total_time = 0.0;
        for (int iter = 0; iter < ITERATIONS; ++iter) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // Perform DiOMP device broadcast
            #pragma omp target data use_device_ptr(gpu_data)
            ompx_dbcast(gpu_data, data_size, omp_device_dt_t::ompx_d_double, 0, 0);
            
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
        
        // Clean up memory
        delete[] gpu_data;
    }

    return 0;
}
