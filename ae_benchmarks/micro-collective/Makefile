# Compiler settings
CC = clang
CXX = clang++

# Base compiler flags
CFLAGS = -O3 -march=native -std=c++17
CXXFLAGS = $(CFLAGS) -fopenmp --offload-arch=sm_80 -D__STRICT_ANSI__ -D_GLIBCXX_USE_CXX11_ABI=1

# Library paths and flags
LLVM_GPU_PATH = /global/cfs/cdirs/xpress/baodi/software/llvm-gpu
GASNET_GPU_PATH = /global/cfs/cdirs/xpress/baodi/software/gasnet-gpu
DIOMP_PATH = /global/cfs/cdirs/xpress/baodi/DiOMP-GPU

# For MPICH on Perlmutter
MPICH_PATH = /opt/cray/pe/mpich/8.1.30
LIBFABRIC_PATH = /opt/cray/libfabric/1.22.0
PMI_PATH = /opt/cray/pe/pmi/6.1.13


# NCCL configuration
NCCL_FLAGS = -I$(NCCL_HOME)/include -L$(NCCL_HOME)/lib -lcuda -lnccl

# MPI configuration
MPI_FLAGS = -I$(MPICH_PATH)/ofi/gnu/12.3/include \
            -L$(MPICH_PATH)/ofi/gnu/12.3/lib \
            -L$(MPICH_PATH)/gtl/lib \
            -L$(LIBFABRIC_PATH)/lib64 \
            -lfabric -lhugetlbfs -lmpi -lmpi_gtl_cuda

# GASNet configuration
GASNET_FLAGS = -I$(GASNET_GPU_PATH)/include/ofi-conduit \
               -L$(GASNET_GPU_PATH)/lib \
               -lgasnet-ofi-par -no-pie -lhwloc -lrt -pthread

# PMI configuration
PMI_FLAGS = -L$(PMI_PATH)/lib -lpmi

# DiOMP configuration
DIOMP_FLAGS = -DDIOMP_ENABLE_CUDA -ldiomp \
              -L$(LLVM_GPU_PATH)/lib \
              -I$(DIOMP_PATH)/openmp/diomp/include

# Common flags for all targets
COMMON_FLAGS = $(CXXFLAGS) $(NCCL_FLAGS) $(MPI_FLAGS) $(PMI_FLAGS)

# Benchmark targets
TARGETS = diomp_bcast mpi_bcast diomp_allreduce mpi_allreduce

.PHONY: all clean help

all: $(TARGETS)

# Build rules
diomp_bcast: diomp_bcast.cpp
	$(CXX) $(COMMON_FLAGS) $(GASNET_FLAGS) $(DIOMP_FLAGS) -o $@ $<

mpi_bcast: mpi_bcast.cpp
	$(CXX) $(COMMON_FLAGS) $(MPI_FLAGS) -o $@ $<

diomp_allreduce: diomp_allreduce.cpp
	$(CXX) $(COMMON_FLAGS) $(GASNET_FLAGS) $(DIOMP_FLAGS) -o $@ $<

mpi_allreduce: mpi_allreduce.cpp
	$(CXX) $(COMMON_FLAGS) $(MPI_FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

help:
	@echo "Available targets:"
	@echo "  all     - Build all benchmarks (default)"
	@echo "  diomp_bcast - Build DiOMP broadcast benchmark"
	@echo "  mpi_bcast   - Build MPI broadcast benchmark"
	@echo "  diomp_allreduce - Build DiOMP allreduce benchmark"
	@echo "  mpi_allreduce - Build MPI allreduce benchmark"
	@echo "  clean    - Remove all built targets"
	@echo "  help     - Show this help message"