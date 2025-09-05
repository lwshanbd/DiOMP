#!/bin/bash
set -euo pipefail

# Function to capture current environment variables
capture_env_vars() {
    local env_file="$1"
    echo "# Environment variables set by tioga.sh installation script" > "$env_file"
    echo "# Source this file to load the environment: source $env_file" >> "$env_file"
    echo "" >> "$env_file"
    
    # Export all relevant environment variables
    echo "export PATH=\"$PATH\"" >> "$env_file"
    echo "export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\"" >> "$env_file"
    echo "export LIBRARY_PATH=\"$LIBRARY_PATH\"" >> "$env_file"
    echo "export MANPATH=\"$MANPATH\"" >> "$env_file"
    echo "export CPATH=\"$CPATH\"" >> "$env_file"
    echo "export C_INCLUDE_PATH=\"$C_INCLUDE_PATH\"" >> "$env_file"
    echo "export CPLUS_INCLUDE_PATH=\"$CPLUS_INCLUDE_PATH\"" >> "$env_file"
    
    # Add specific installation paths as comments for reference
    echo "" >> "$env_file"
    echo "# Installation directories:" >> "$env_file"
    echo "# GASNET_ROOT=\"$GASNET_ROOT\"" >> "$env_file"
    echo "# LLVM_DIR=\"$LLVM_DIR\"" >> "$env_file"
    echo "# INSTALL_DIR=\"$INSTALL_DIR\"" >> "$env_file"
    
    echo "" >> "$env_file"
    echo "echo \"Environment loaded from tioga.sh installation\"" >> "$env_file"
}

# Load required modules
module load rocm

# Create installation directory
mkdir -p install

# Set installation paths
readonly INSTALL_DIR="$PWD/install"
readonly GASNET_ROOT="$INSTALL_DIR/gasnet"
readonly LLVM_DIR="$INSTALL_DIR/llvm-gpu"

# Common LLVM/CUDA configuration
readonly AMD_ARCH="gfx90a"
readonly BUILD_JOBS="64"

# 1. Install GASNet
echo "Installing GASNet..."
git clone https://bitbucket.org/berkeleylab/gasnet.git
cd gasnet
./Bootstrap

./configure --prefix="$GASNET_ROOT" --disable-ibv --enable-ofi --enable-par --enable-pthreads --with-ibv-spawner=pmi --enable-segment-fast --disable-mpi --disable-smp --disable-portals --disable-mxm --with-max-segsize=32GB --enable-par --disable-seq --disable-parsync --disable-ibv-rcv-thread --disable-aligned-segments --disable-fca --enable-memory-kinds --with-mpi-cflags=-fPIC --with-cflags=-fPIC --enable-hwloc
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
    -B llvm_build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_TARGETS_TO_BUILD="host;AMDGPU" \
    -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt" \
    -DLLVM_ENABLE_RUNTIMES="openmp;offload"\
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON\
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_APPEND_VC_REV=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DENABLE_DIOMP_DEVICE=ON \
    -DOPENMP_MPI_ROOT="/opt/cray/pe/mpich/8.1.31" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON\
    -DLIBOMPTARGET_BUILD_DEVICERTL_BCLIB=ON \
    -DOPENMP_DIOMP_GASNET_ROOT="$GASNET_ROOT" \
    -DOPENMP_GASNET_API="OFI" \
    -DOPENMP_ENABLE_DIOMP_DEVICE=ON \
    -DOPENMP_DIOMP_ENABLE_HIP=ON \
    -DCMAKE_INSTALL_PREFIX="$LLVM_DIR" \
    -DRUNTIMES_CMAKE_ARGS="-DLIBOMPTARGET_PLUGINS_TO_BUILD='amdgpu';-DLIBOMPTARGET_DEVICE_ARCHITECTURES='$AMD_ARCH';-DLIBOMPTARGET_DLOPEN_PLUGINS='';-DLIBOMPTARGET_AMDGPU_ARCH='$AMD_ARCH';-DLIBOMPTARGET_AMDGCN_GFXLIST='$AMD_ARCH';-DLIBOMPTARGET_FORCE_AMDGPU_TESTS=ON" \
    -S ../llvm
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
    -DOPENMP_ENABLE_DIOMP=ON

ninja -j "$BUILD_JOBS" -C llvm_build install

cd ../ae_benchmarks/

echo "Installation completed successfully!"

# Generate environment variables file
ENV_FILE="$INSTALL_DIR/tioga_env.sh"
capture_env_vars "$ENV_FILE"

echo ""
echo "=================================================="
echo "Environment variables have been saved to:"
echo "  $ENV_FILE"
echo ""
echo "To load the environment in future sessions, run:"
echo "  source $ENV_FILE"
echo "=================================================="


