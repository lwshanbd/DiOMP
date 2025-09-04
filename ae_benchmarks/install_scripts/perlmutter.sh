#!/bin/bash
set -euo pipefail

# Load required modules
module load nccl

# Create installation directory
mkdir -p install

# Set installation paths
readonly INSTALL_DIR="$PWD/install"
readonly GASNET_ROOT="$INSTALL_DIR/gasnet"
readonly LLVM_DIR="$INSTALL_DIR/llvm-gpu"

# Common LLVM/CUDA configuration
readonly CUDA_ARCH="sm_80"
readonly BUILD_JOBS="32"

# 1. Install GASNet
echo "Installing GASNet..."
git clone https://bitbucket.org/berkeleylab/gasnet.git
cd gasnet
./Bootstrap

./configure \
    --prefix="$GASNET_ROOT" \
    --enable-ofi \
    --with-ofi-provider=cxi \
    --enable-pshm \
    --enable-par \
    --with-ibv-spawner=mpi \
    --enable-segment-fast \
    --disable-mpi \
    --disable-smp \
    --disable-portals \
    --disable-mxm \
    --enable-pthreads \
    --with-max-segsize=16GB \
    --disable-seq \
    --disable-parsync \
    --disable-ibv-rcv-thread \
    --disable-aligned-segments \
    --disable-fca \
    --enable-memory-kinds \
    --with-mpi-cflags=-fPIC \
    --with-cflags=-fPIC \
    --enable-kind-cuda-uva

make -j"$BUILD_JOBS"
make install

# Setup GasNet environment
setup_gasnet_env() {
    # Initialize environment variables if they don't exist
    : "${LD_LIBRARY_PATH:=}"
    : "${LIBRARY_PATH:=}"
    : "${MANPATH:=}"
    : "${CPATH:=}"
    : "${C_INCLUDE_PATH:=}"
    : "${CPLUS_INCLUDE_PATH:=}"
    
    export PATH="$GASNET_ROOT/bin:$PATH"
    export LD_LIBRARY_PATH="$GASNET_ROOT/lib:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="$GASNET_ROOT/lib:$LIBRARY_PATH"
    export MANPATH="$GASNET_ROOT/share/man:$MANPATH"
    export CPATH="$GASNET_ROOT/include:$CPATH"
    export C_INCLUDE_PATH="$GASNET_ROOT/include:$C_INCLUDE_PATH"
    export CPLUS_INCLUDE_PATH="$GASNET_ROOT/include:$CPLUS_INCLUDE_PATH"
}
setup_gasnet_env

cd ..

# 2. Install DiOMP
echo "Installing DiOMP..."
git clone https://github.com/lwshanbd/DiOMP.git -b ae --depth 1
cd DiOMP
mkdir -p build
cd build

# Common LLVM CMake options
common_llvm_options=(
    -G Ninja
    -B llvm_build
    -S ../llvm
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DCMAKE_C_COMPILER_LAUNCHER=gcc
    -DCMAKE_CXX_COMPILER_LAUNCHER=g++
    -DCMAKE_INSTALL_PREFIX="$LLVM_DIR"
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX"
    -DLLVM_ENABLE_PROJECTS="clang"
    -DLLVM_ENABLE_RUNTIMES="openmp;offload"
    -DLLVM_INSTALL_UTILS=ON
    -DLLVM_INCLUDE_BENCHMARKS=OFF
    -DLLVM_INCLUDE_EXAMPLES=OFF
    -DLLVM_ENABLE_ASSERTIONS=ON
    -DLLVM_APPEND_VC_REV=OFF
    -DBUILD_SHARED_LIBS=ON
    -DENABLE_DIOMP_DEVICE=ON
    -DOPENMP_DIOMP_ENABLE_CUDA=ON
    -DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=ON
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DLIBOMPTARGET_BUILD_DEVICERTL_BCLIB=ON
    -DOPENMP_ENABLE_DIOMP_DEVICE=1
    -DCLANG_OPENMP_NVPTX_DEFAULT_ARCH="$CUDA_ARCH"
    -DLIBOMPTARGET_NVPTX_ARCH="$CUDA_ARCH"
    -DDLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES="$CUDA_ARCH"
    -DLIBOMPTARGET_PLUGINS_TO_BUILD="cuda"
    -DLIBOMPTARGET_DEVICE_ARCHITECTURES="$CUDA_ARCH"
    -DLIBOMPTARGET_AMDGPU_ARCH=""
    -DLIBOMPTARGET_AMDGCN_GFXLIST=""
)

# 2.1 Install LLVM+Clang+Offload+OpenMP without DiOMP
echo "Building LLVM without DiOMP..."
cmake "${common_llvm_options[@]}" \
    -DOPENMP_ENABLE_DIOMP=OFF

ninja -j "$BUILD_JOBS" -C llvm_build install

# Setup LLVM environment
setup_llvm_env() {
    export PATH="$LLVM_DIR/bin:$PATH"
    export LD_LIBRARY_PATH="$LLVM_DIR/lib:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="$LLVM_DIR/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="$LLVM_DIR/lib:$LIBRARY_PATH"
    export LIBRARY_PATH="$LLVM_DIR/lib/x86_64-unknown-linux-gnu:$LIBRARY_PATH"
    export MANPATH="$LLVM_DIR/share/man:$MANPATH"
    export CPATH="$LLVM_DIR/include:$CPATH"
}
setup_llvm_env

# 2.2 Install LLVM+Clang+Offload+OpenMP with DiOMP
echo "Building LLVM with DiOMP..."
cmake "${common_llvm_options[@]}" \
    -DOPENMP_ENABLE_DIOMP=ON \
    -DOPENMP_DIOMP_GASNET_ROOT="$GASNET_ROOT" \
    -DOPENMP_GASNET_API="OFI" \
    -DOPENMP_LIBFABRIC_LIB="/opt/cray/libfabric/1.22.0/lib64" \
    -DOPENMP_PMI_ROOT="/opt/cray/pe/pmi/6.1.15" \
    -DOPENMP_MPI_ROOT="/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3"

ninja -j "$BUILD_JOBS" -C llvm_build install

cd ../ae_benchmarks/

echo "Installation completed successfully!"


