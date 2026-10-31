#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

namespace fh_index_rw {

// One 64-bit finger publishes every state needed to read a bucket. Blocks are
// at least 16-byte aligned, so storing pointer >> 4 leaves room for logical
// size, physical capacity, a short mutation version, and the writer/rebuild
// state. A version never wraps on one physical block: the writer replaces the
// block when kVersionMask is reached.
//
// Layout: [block_pointer>>4:44][rebuilding:1][writer_lock:1][version:4]
//         [capacity:7][size:7]
class FingerRW {
 public:
  static constexpr unsigned kPointerAlignmentBits = 4;
  static constexpr size_t kPointerAlignment = 1ULL << kPointerAlignmentBits;

  static constexpr unsigned kSizeBits = 7;
  static constexpr int kSizeShift = 0;
  static constexpr uint64_t kSizeMask = (1ULL << kSizeBits) - 1;

  static constexpr unsigned kCapacityBits = 7;
  static constexpr int kCapacityShift = kSizeShift + kSizeBits;
  static constexpr uint64_t kCapacityMask = (1ULL << kCapacityBits) - 1;

  static constexpr unsigned kVersionBits = 4;
  static constexpr int kVersionShift = kCapacityShift + kCapacityBits;
  static constexpr uint64_t kVersionMask = (1ULL << kVersionBits) - 1;

  static constexpr int kWriteLockShift = kVersionShift + kVersionBits;
  static constexpr uint64_t kWriteLockBit = 1ULL << kWriteLockShift;
  static constexpr int kRebuildingShift = kWriteLockShift + 1;
  static constexpr uint64_t kRebuildingBit = 1ULL << kRebuildingShift;

  static constexpr int kPtrShift = kRebuildingShift + 1;
  static constexpr unsigned kEncodedPtrBits = 44;
  static constexpr uint64_t kEncodedPtrMask =
      (1ULL << kEncodedPtrBits) - 1;
  static constexpr uint64_t kPtrMask = kEncodedPtrMask << kPtrShift;
  static_assert(kPtrShift + kEncodedPtrBits == 64);

  FingerRW(void* block = nullptr, uint16_t size = 0)
      : FingerRW(block, size, size, 0) {}

  FingerRW(void* block, uint16_t size, uint16_t capacity, uint8_t version) {
    data_.store(Pack(block, size, capacity, version),
                std::memory_order_relaxed);
  }

  FingerRW(const FingerRW& other) {
    data_.store(other.data_.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
  }

  FingerRW& operator=(const FingerRW& other) {
    if (this != &other) {
      data_.store(other.data_.load(std::memory_order_relaxed),
                  std::memory_order_relaxed);
    }
    return *this;
  }

  uint64_t GetData() const {
    return data_.load(std::memory_order_acquire);
  }

  void SetData(uint64_t val) {
    data_.store(val, std::memory_order_relaxed);
  }

  void Publish(uint64_t val) {
    assert(!IsWriteLocked(val));
    data_.store(val, std::memory_order_release);
  }

  static uint64_t Pack(void* block, uint16_t size) {
    return Pack(block, size, size, 0);
  }

  static uint64_t Pack(void* block, uint16_t size, uint16_t capacity,
                       uint8_t version) {
    assert(size <= kSizeMask && "Size exceeds packed finger limit");
    assert(capacity <= kCapacityMask &&
           "Capacity exceeds packed finger limit");
    assert(size <= capacity && "Logical size exceeds physical capacity");
    assert(version <= kVersionMask && "Version exceeds packed finger limit");

    const uint64_t raw_ptr = reinterpret_cast<uint64_t>(block);
    assert((raw_ptr & (kPointerAlignment - 1)) == 0 &&
           "Block pointer is not 16-byte aligned");
    assert((raw_ptr >> 48) == 0 && "Block pointer exceeds 48-bit address space");
    const uint64_t encoded_ptr = raw_ptr >> kPointerAlignmentBits;
    assert((encoded_ptr & ~kEncodedPtrMask) == 0);

    return (encoded_ptr << kPtrShift) |
           static_cast<uint64_t>(size) |
           (static_cast<uint64_t>(capacity) << kCapacityShift) |
           (static_cast<uint64_t>(version) << kVersionShift);
  }

  static uint16_t UnpackSize(uint64_t data) {
    return static_cast<uint16_t>((data >> kSizeShift) & kSizeMask);
  }

  static uint16_t UnpackCapacity(uint64_t data) {
    return static_cast<uint16_t>((data >> kCapacityShift) & kCapacityMask);
  }

  static uint8_t UnpackVersion(uint64_t data) {
    return static_cast<uint8_t>((data >> kVersionShift) & kVersionMask);
  }

  template <typename BlockType>
  static BlockType* UnpackBlock(uint64_t data) {
    const uint64_t raw_ptr =
        (data >> kPtrShift) << kPointerAlignmentBits;
    return reinterpret_cast<BlockType*>(raw_ptr);
  }

  static bool IsRebuilding(uint64_t data) {
    return (data & kRebuildingBit) != 0;
  }

  static bool IsWriteLocked(uint64_t data) {
    return (data & kWriteLockBit) != 0;
  }

  static bool IsVersionExhausted(uint64_t data) {
    return UnpackVersion(data) == kVersionMask;
  }

  static uint64_t SetWriteLockBit(uint64_t data) {
    return data | kWriteLockBit;
  }

  static uint64_t SetRebuildingBit(uint64_t data) {
    return data | kRebuildingBit;
  }

  static bool IsShadow(uint64_t) { return false; }

  bool CompareExchange(uint64_t& expected, uint64_t desired) {
    return data_.compare_exchange_strong(expected, desired,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire);
  }

  std::string Print() const {
    const uint64_t val = data_.load(std::memory_order_relaxed);
    return "FingerRW: {block: " +
           std::to_string(reinterpret_cast<uint64_t>(
               UnpackBlock<void>(val))) +
           ", size: " + std::to_string(UnpackSize(val)) +
           ", capacity: " + std::to_string(UnpackCapacity(val)) +
           ", version: " + std::to_string(UnpackVersion(val)) +
           ", rebuilding: " + std::to_string(IsRebuilding(val)) + "}";
  }

  template <typename BlockType>
  void Predict(const uint64_t key, BlockType*& out_block,
               uint16_t& out_size) const {
    (void)key;
    const uint64_t val = data_.load(std::memory_order_acquire);
    out_block = UnpackBlock<BlockType>(val);
    out_size = UnpackSize(val);
  }

  void SetRebuilding() {
    uint64_t old_val = data_.load(std::memory_order_acquire);
    while (true) {
      if (IsRebuilding(old_val)) return;
      if (IsWriteLocked(old_val)) {
        old_val = data_.load(std::memory_order_acquire);
        continue;
      }
      const uint64_t new_val = SetRebuildingBit(old_val);
      if (data_.compare_exchange_weak(old_val, new_val,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
        return;
      }
    }
  }

 private:
  std::atomic<uint64_t> data_{0};
};

static_assert(sizeof(FingerRW) == 8, "FingerRW should be 8 bytes");

}  // namespace fh_index_rw
