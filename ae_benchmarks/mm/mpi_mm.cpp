#include <iostream>
#include <utility>
#include <ctime>

#include <mpi.h>


void print_matrix(const float* mat, const int Is, const int Js)
{
    for (int i = 0; i < Is; i++) {
        for (int j = 0; j < Js; j++)
            std::cout << mat[i * Js + j] << ' ';
        std::cout << '\n';
    }
}


float timediff_us(const timespec& t_start, const timespec& t_end)
{
    return (t_end.tv_sec - t_start.tv_sec) * 1.0e6 + (t_end.tv_nsec - t_start.tv_nsec) / 1.0e3;
}


int main()
{
    MPI_Init(nullptr, nullptr);

    int mype, npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

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
        MPI_Barrier(MPI_COMM_WORLD);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        for (int s = 0; s < npes; s++) {
            const int block_num = (mype + s) % npes;

            MPI_Request sreq;
            #pragma omp target data use_device_ptr(Bs)
            MPI_Isend(Bs, N * Ns, MPI_FLOAT, (npes + mype - 1) % npes, 0, MPI_COMM_WORLD, &sreq);

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

            #pragma omp target data use_device_ptr(Bn)
            MPI_Recv(Bn, N * Ns, MPI_FLOAT, (mype + 1) % npes, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Wait(&sreq, MPI_STATUS_IGNORE);
            std::swap(Bs, Bn);

            MPI_Barrier(MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

        if (mype == 0) {
            std::cout << "MPI + OpenMP: " << timediff_us(t0, t1) << " us\n";
        }
    }

    // Collect and print the matrix product
    if (N < 32) {
        if (mype == 0) {
            auto C = new float[N * N];

            for (int i = 0; i < Ns * N; i++)
                C[i] = Cs[i];

            for (int i = 1; i < npes; i++)
                MPI_Recv(C + i * Ns * N, N * Ns, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            print_matrix(C, N, N);

            delete[] C;
        } else {
            MPI_Send(Cs, N * Ns, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    delete [] Bn;
    delete [] Cs;
    delete [] Bs;
    delete [] As;

    MPI_Finalize();
}
