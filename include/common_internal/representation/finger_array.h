#pragma once

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <new>
#include <type_traits>

#include "common_internal/memory/memory_pool.h"

namespace fh_index {
namespace internal {

// FingerArray structure with contiguous memory layout:
// [size, data[]]
// Each FingerGroupDescriptor points to an independent FingerArray.
struct FingerArray {
    size_t size;          // Number of fingers in this array
    uint8_t data[];       // Flexible array member for fingers

    // Access fingers array
    template <typename FingerType>
    FingerType* fingers() {
      return reinterpret_cast<FingerType*>(data);
    }

    template <typename FingerType>
    const FingerType* fingers() const {
      return reinterpret_cast<const FingerType*>(data);
    }

    // Get finger at index
    template <typename FingerType>
    FingerType& GetFinger(size_t idx) {
      assert(idx < size);
      return fingers<FingerType>()[idx];
    }

    template <typename FingerType>
    const FingerType& GetFinger(size_t idx) const {
      assert(idx < size);
      return fingers<FingerType>()[idx];
    }

    // Get pointer to finger at index
    template <typename FingerType>
    FingerType* GetFingerAddr(size_t idx) {
      assert(idx < size);
      return &fingers<FingerType>()[idx];
    }

    template <typename FingerType>
    const FingerType* GetFingerAddr(size_t idx) const {
      assert(idx < size);
      return &fingers<FingerType>()[idx];
    }

    template <typename FingerType>
    static FingerArray* Create(size_t count) {
      static_assert(std::is_trivially_destructible_v<FingerType>);
      const size_t total_bytes = AllocationBytes<FingerType>(count);
      void* mem = memory::Allocate(total_bytes);
      assert(mem != nullptr);
      FingerArray* arr = new (mem) FingerArray();
      arr->size = count;
      return arr;
    }

    template <typename FingerType>
    static void Destroy(FingerArray* arr) {
      static_assert(std::is_trivially_destructible_v<FingerType>);
      assert(arr != nullptr);
      const size_t total_bytes = AllocationBytes<FingerType>(arr->size);
      arr->~FingerArray();
      memory::Free(arr, total_bytes);
    }

    // Delete copy/move to prevent misuse
    FingerArray(const FingerArray&) = delete;
    FingerArray& operator=(const FingerArray&) = delete;
    FingerArray(FingerArray&&) = delete;
    FingerArray& operator=(FingerArray&&) = delete;

private:
    template <typename FingerType>
    static size_t AllocationBytes(size_t count) {
      return sizeof(FingerArray) + count * sizeof(FingerType);
    }

    // Private constructor - use Create() instead
    FingerArray() : size(0) {}
    ~FingerArray() = default;
};

}  // namespace internal
}  // namespace fh_index
