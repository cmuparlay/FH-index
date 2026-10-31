#ifndef FH_INDEX_RO_FINGER_H
#define FH_INDEX_RO_FINGER_H

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace fh_index_ro {

class FingerRO {
 public:
  FingerRO(size_t index = 0, uint16_t size = 0) {
    // assert((size < (1ull << kBitSize)));
    assert((index < (1ull << kBitIndex)));
    index_size_ = (index << kBitSize) | size;
  }

  void SetSize(uint16_t size) {
    // assert((size < (1ull << kBitSize)));
    index_size_ = (index_size_ & kMaskIndex) | size;
  }
  void SetIndex(size_t index) {
    assert((index < (1ull << kBitIndex)));
    index_size_ = (index << kBitSize) | (index_size_ & kMaskSize);
  }
  uint16_t GetSize() const { return index_size_ & kMaskSize; }
  size_t GetIndex() const { return index_size_ >> kBitSize; }

  std::string Print() {
    std::string result;
    result += "FingerRO: {";
    result += "index: " + std::to_string(GetIndex()) + ", ";
    result += "size: " + std::to_string(GetSize()) + "}";
    return result;
  }

  void Predict(const uint64_t key, uint64_t& left_pos,
               uint64_t& right_pos) const {
    (void)key;
    left_pos = GetIndex();
    right_pos = left_pos + GetSize();
  }

  static constexpr size_t kBitSize = 16;
  static constexpr size_t kBitIndex = 48;
  static constexpr uint64_t kMaskSize = (1ull << kBitSize) - 1;
  static constexpr uint64_t kMaskIndex = ~kMaskSize;

  // val_[0:15] = size(16)
  // val_[16:63] = index(48)
  uint64_t index_size_;
};

}  // namespace fh_index_ro

#endif  // FH_INDEX_RO_FINGER_H