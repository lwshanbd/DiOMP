/*
 * diompmem.h
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIOMP_MEM_H
#define DIOMP_MEM_H

namespace diomp{
  class MemoryManager {
  public:
    MemoryManager(size_t global_size, size_t local_size);
    ~MemoryManager();

    void* allocate(size_t size);
    void free(void* ptr);

  private:
    void* global_memory_;
    void* local_memory_;
    size_t global_size_;
    size_t local_size_;
};

  
}