#ifndef DIOMP_HPP
#define DIOMP_HPP

#include "diomp.h"

namespace omp {
namespace diomp {

template <typename T>
void* diomp_alloc(size_t Size);

} // namespace diomp
} // namespace omp

#endif //DIOMP_HPP