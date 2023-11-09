# CMakeLists.txt for diomp

set(GASNET "$HOME/gasnet-install")
set(GASNET_IBV_INCLUDE "/home/l1065028/gasnet-install/include/ibv-conduit/")
set(GASNET_LIB "/home/l1065028/gasnet-install/gasnet-install/lib")

include_directories(${GASNET_IBV_INCLUDE})
link_directories(${GASNET_LIB} )


add_library(diomp SHARED
  src/diomp.cpp
)
target_link_libraries(diomp hwloc ibverbs rt)

target_include_directories(diomp
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

install(TARGETS diomp LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")
install(FILES include/diomp.h DESTINATION "${LIBOMP_HEADERS_INSTALL_PATH}")


# clang++ test.cpp -ldiomp -isystem  /home/l1065028/llvm-install/include -I $HOME/gasnet-install/include/ibv-conduit/ -I $MPI_HOME/include -L $MPI_HOME/lib -lmpi  -lgasnet-ibv-par -lhwloc -libverbs -lrt -pthread -fopenmp