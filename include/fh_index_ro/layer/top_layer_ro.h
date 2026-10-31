#pragma once

#include <algorithm>
#include <utility>

#include "parlay/parallel.h"

namespace fh_index_ro {

template <typename KeyType, typename ValueType>
class TopLayerRO {
 public:
  using EntryType = std::pair<KeyType, ValueType>;

  size_t LowerBound(KeyType key) {
    size_t pos = std::lower_bound(keys_, keys_ + size_, key) - keys_;
    assert(pos <= size_);
    return pos;
  }

  //   size_t PointPredecessorPosition(KeyType key) {
  //     return std::upper_bound(keys_, keys_ + size_, key) - keys_ - 1;
  //   }

  //   template <size_t kNumRequests>
  //   void BatchPredecessorPosition(KeyType* keys, size_t* results) {
  //     for (size_t i = 0; i < kNumRequests; i++) {
  //       results[i] = std::upper_bound(keys_, keys_ + size_, keys[i]) - keys_
  //       - 1;
  //     }
  //   }

  //   // smallest i with keys[i] >= query
  //   template <size_t kNumRequests>
  //   void BatchLowerBoundPosition(KeyType* keys, size_t* results) {
  //     for (size_t i = 0; i < kNumRequests; i++) {
  //       results[i] = std::lower_bound(keys_, keys_ + size_, keys[i]) - keys_;
  //     }
  //   }

  //   // smallest i with keys[i] > query
  //   template <size_t kNumRequests>
  //   void BatchUpperBoundPosition(KeyType* keys, size_t* results) {
  //     for (size_t i = 0; i < kNumRequests; i++) {
  //       results[i] = std::upper_bound(keys_, keys_ + size_, keys[i]) - keys_;
  //     }
  //   }

  size_t GetSize() {
    assert(initialized_);
    return size_;
  }

  template <size_t kNumRequests>
  void BatchGetKey(size_t* idx, KeyType* keys) {
    for (size_t i = 0; i < kNumRequests; i++) {
      assert(idx[i] < size_);
      keys[i] = keys_[idx[i]];
    }
  }

  // void BatchGetKeyOptional(size_t* idx, std::optional<KeyType>* keys) {
  //     #pragma unroll
  //     for (size_t i = 0; i < kNumRequests; i++) {
  //         if (idx[i] < size_) {
  //             keys[i] = std::make_optional(keys_[idx[i]]);
  //         } else {
  //             keys[i] = std::nullopt;
  //         }
  //     }
  // }

  template <size_t kNumRequests>
  void BatchGetValue(size_t* idx, ValueType* values) {
    for (size_t i = 0; i < kNumRequests; i++) {
      assert(idx[i] < size_);
      values[i] = values_[idx[i]];
    }
  }

  // void BatchGetValueOptional(size_t* idx, std::optional<ValueType>* values) {
  //     #pragma unroll
  //     for (size_t i = 0; i < kNumRequests; i++) {
  //         if (idx[i] < size_) {
  //             values[i] = std::make_optional(values_[idx[i]]);
  //         } else {
  //             values[i] = std::nullopt;
  //         }
  //     }
  // }

  EntryType GetKV(size_t idx) {
    assert(idx < size_);
    return std::make_pair(keys_[idx], values_[idx]);
  }

  template <size_t kNumRequests>
  void BatchGetKV(size_t* idx, EntryType* kvs) {
    for (size_t i = 0; i < kNumRequests; i++) {
      assert(idx[i] < size_);
      kvs[i] = std::make_pair(keys_[idx[i]], values_[idx[i]]);
    }
  }

  // void BatchGetKVOptional(size_t* idx, std::optional<EntryType>* kvs) {
  //     #pragma unroll
  //     for (size_t i = 0; i < kNumRequests; i++) {
  //         if (idx[i] < size_) {
  //             kvs[i] = std::make_optional(
  //                 std::make_pair(keys_[idx[i]], values_[idx[i]]));
  //         } else {
  //             kvs[i] = std::nullopt;
  //         }
  //     }
  // }

  TopLayerRO() {
    layer_id_ = 0;

    initialized_ = false;
    size_ = 0;
    array_size_ = 0;
    keys_ = nullptr;
    values_ = nullptr;
  }

  TopLayerRO(const TopLayerRO&) = delete;
  TopLayerRO& operator=(const TopLayerRO&) = delete;

  TopLayerRO(TopLayerRO&& other) noexcept : TopLayerRO() {
    *this = std::move(other);
  }

  TopLayerRO& operator=(TopLayerRO&& other) noexcept {
    if (this != &other) {
      delete[] keys_;
      delete[] values_;
      layer_id_ = std::exchange(other.layer_id_, 0);
      initialized_ = std::exchange(other.initialized_, false);
      size_ = std::exchange(other.size_, 0);
      array_size_ = std::exchange(other.array_size_, 0);
      keys_ = std::exchange(other.keys_, nullptr);
      values_ = std::exchange(other.values_, nullptr);
    }
    return *this;
  }

  void Print(bool print_keys = false) {
    std::cout << "TopLayerRO: " << std::endl;
    std::cout << "size_: " << size_ << std::endl;
    if (print_keys || size_ < 10) {
      for (size_t i = 0; i < size_; i++) {
        std::cout << keys_[i] << " ";
      }
    }
    std::cout << std::endl;
  }

  void BulkLoad(size_t layer_id, size_t size, size_t array_size,
                KeyType*&& keys, ValueType*&& values) {
    if (initialized_) {
      delete[] keys_;
      delete[] values_;
    }

    layer_id_ = layer_id;

    initialized_ = true;
    size_ = size;
    array_size_ = array_size;
    keys_ = keys;
    values_ = values;
  }

  ~TopLayerRO() {
    if (initialized_) {
      delete[] keys_;
      delete[] values_;
    }
  }

  size_t layer_id_;

  bool initialized_ = false;
  size_t size_, array_size_;
  KeyType* keys_;
  ValueType* values_;
};

}  // namespace fh_index_ro
