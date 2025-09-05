#include <cstdio>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <diomp.h>

#define ITERATIONS 100 // Number of test iterations

int main() {
    // Initialize DiOMP runtime (if required by the environment)
    __init_diomp_target(1);

    int rank = omp_get_rank_num();
    int size = omp_get_num_ranks();
    int DevicesNum = omp_get_num_devices();

    // Data size from 4 bytes to 128MB, doubling each time
    for (size_t data_size = 1; data_size <= (256 * 1024 * 1024) / sizeof(double); data_size *= 2) {
        // Allocate GPU memory
        double *gpu_src = new double[data_size];
        double *gpu_dst = new double[data_size];

        for (int i = 0; i < DevicesNum; i++) {
            #pragma omp target enter data map(to:gpu_src[0:data_size], gpu_dst[0:data_size]) device(i)
            {}
        }

        if (rank == 0) {
            #pragma omp target teams distribute parallel for
            for (int i = 0; i < data_size; ++i) {
                gpu_src[i] = 1.0;
                gpu_dst[i] = 0.0;
            }
        }

        diomp_barrier();

        // Measure Allreduce performance
        double total_time = 0.0;
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();

            #pragma omp target data use_device_ptr(gpu_src, gpu_dst)
            ompx_dallreduce(gpu_src, gpu_dst, data_size, omp_device_dt_t::ompx_d_double, omp_red_op_t::ompx_d_sum, 0);

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

        for (int i = 0; i < DevicesNum; i++) {
            #pragma omp target exit data map(delete:gpu_src[0:data_size], gpu_dst[0:data_size]) device(i)
            {}
        }

        delete[] gpu_src;
        delete[] gpu_dst;
    }

    return 0;
}
