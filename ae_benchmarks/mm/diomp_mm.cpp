#include <iostream>
#include <utility>
#include <ctime>
#include <omp.h>
#include <diomp.h>



float timediff_us(const timespec& t_start, const timespec& t_end)
{
    return (t_end.tv_sec - t_start.tv_sec) * 1.0e6 + (t_end.tv_nsec - t_start.tv_nsec) / 1.0e3;
}


int main()
{
    __init_diomp_target(2);
    int mype, npes;
    mype = omp_get_rank_num();
    npes = omp_get_num_ranks();

    const int Ns = 30240/npes;        // Width of the stripes
    const int N = 30240;    // Size of the matrices
    const size_t stripe_size = N * Ns * sizeof(float);

    if (mype == 0)
        std::cout << "Matrix stripe: " << N << 'x' << Ns << ", " << stripe_size << " bytes\n";

    auto As = new float[N * Ns];    // Horizontal stripes of A
    auto Bs = new float[N * Ns];    // Vertical stripes of B
    auto Cs = new float[N * Ns];    // Horizontal stripes of C
    auto Bn = new float[N * Ns];    // Next stripe of B

    // Initialize the matrices
    for(int i = 0; i < N * Ns; i++) {
        As[i] = (i + mype) % 11 + 7;
        Bs[i] = (i + mype) % 13 + 5;
        Cs[i] = 0;
        Bn[i] = 0;
    }

    #pragma omp target data map(to:As[0:N*Ns],Bs[0:N*Ns]) map(tofrom:Cs[0:N*Ns]) map(alloc:Bn[0:N*Ns])
    {
        timespec t0, t1;

        // Make sure all the stripes are initialized
        ompx_barrier(nullptr);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        for (int s = 0; s < npes; s++) {
            const int block_num = (mype + s) % npes;

            // #pragma omp target data use_device_ptr(Bs,Bn)
            ompx_dget(Bn, (mype + 1) % npes, Bs, N * Ns * sizeof(float), 0, 0);

            float* const Cb = Cs + block_num * Ns;

            #pragma omp target teams distribute parallel for collapse(2)
            for (int k = 0; k < N; k++) {
                for (int j = 0; j < Ns; j++) {
                    const float b_kj = Bs[k * Ns + j];
                    for (int i = 0; i < Ns; i++) {
                        Cb[i * N + j] += As[i * N + k] * b_kj;
                    }
                }
            }
            ompx_fence();
            std::swap(Bs, Bn);

            ompx_barrier(nullptr);
        }

        ompx_barrier(nullptr);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

        if (mype == 0) {
            std::cout << "DiOMP: " << timediff_us(t0, t1) << " us\n";
        }
    }



    ompx_barrier(nullptr);

    delete [] Bn;
    delete [] Cs;
    delete [] Bs;
    delete [] As;
}
