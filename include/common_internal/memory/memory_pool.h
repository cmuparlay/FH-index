#pragma once

#include <cstddef>
#include <cstdlib>

namespace memory {

inline void* Allocate(size_t size) { return std::malloc(size); }

inline void Free(void* ptr, size_t /*size*/) { std::free(ptr); }

// Allocate with alignment
inline void* AllocateAligned(size_t size, size_t alignment) {
  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size) != 0) {
    return nullptr;
  }
  return ptr;
}

// Free aligned memory
inline void FreeAligned(void* ptr, size_t /*size*/) { std::free(ptr); }

}  // namespace memory
