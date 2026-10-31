#pragma once

#include <immintrin.h>

#include <vector>

#include "common_internal/construction/fraction.h"
#include "common_internal/utility/math.h"
#include "parlay/parallel.h"

namespace fh_index_ro {

template <typename KeyType, typename ValueType>
class KVLayerRO {

 public:
  using EntryType = std::pair<KeyType, ValueType>;

  KVLayerRO() {
    initialized_ = false;

    layer_id_ = 0;
    size_ = 0;
    element_array_size_ = 0;
    B_ = 0;
    finger_density_ = fh_index::internal::Fraction();
    keys_ = nullptr;
    values_ = nullptr;
  }

  // Delete copy constructor to prevent copying
  KVLayerRO(const KVLayerRO& other) = delete;

  // Delete copy assignment operator to prevent copying
  KVLayerRO& operator=(const KVLayerRO& other) = delete;

  KVLayerRO(KVLayerRO&& other) noexcept {
    layer_id_ = other.layer_id_;
    size_ = other.size_;
    element_array_size_ = other.element_array_size_;
    B_ = other.B_;
    finger_density_ = other.finger_density_;
    initialized_ = other.initialized_;
    keys_ = other.keys_;
    values_ = other.values_;

    // Reset the other object's resources
    other.keys_ = nullptr;
    other.values_ = nullptr;
    other.initialized_ = false;
  }

  KVLayerRO& operator=(KVLayerRO&& other) noexcept {
    if (this != &other) {
      Clear();

      layer_id_ = other.layer_id_;
      size_ = other.size_;
      element_array_size_ = other.element_array_size_;
      B_ = other.B_;
      finger_density_ = other.finger_density_;
      initialized_ = other.initialized_;
      keys_ = other.keys_;
      values_ = other.values_;

      // Reset the other object's resources
      other.keys_ = nullptr;
      other.values_ = nullptr;
      other.initialized_ = false;
    }
    return *this;
  }

  ~KVLayerRO() { Clear(); }

  void Clear() {
    if (initialized_) {
      delete[] keys_;
      delete[] values_;
      keys_ = nullptr;
      values_ = nullptr;
    }
    initialized_ = false;

    layer_id_ = 0;
    size_ = 0;
    element_array_size_ = 0;
    B_ = 0;
    finger_density_ = fh_index::internal::Fraction();
  }

  void BulkLoad(size_t layer_id, size_t size, size_t array_size,
                KeyType*&& keys, ValueType*&& values, size_t B,
                fh_index::internal::Fraction finger_density) {
    Clear();
    initialized_ = true;

    layer_id_ = layer_id;
    size_ = size;
    element_array_size_ = array_size;
    B_ = B;
    finger_density_ = finger_density;
    keys_ = keys;
    values_ = values;
  }

  template<bool PrefetchValue = true>
  size_t LowerBoundOverKVArray(
      KeyType key, size_t left_pos, size_t right_pos) const {
    assert((right_pos >= left_pos && right_pos - left_pos <= B_) && "Violate Search Span Constraint");
    if (right_pos == left_pos) return left_pos;
#ifdef __AVX512F__
    return LowerBoundOverKVArrayAvx512<PrefetchValue>(key, left_pos, right_pos);
#elif defined(__AVX2__)
    return LowerBoundOverKVArrayAvx2<PrefetchValue>(key, left_pos, right_pos);
#else
    return LowerBoundOverKVArrayNaive<PrefetchValue>(key, left_pos, right_pos);
#endif
  }

#ifdef __AVX512F__
  template<bool PrefetchValue = true>
  size_t LowerBoundOverKVArrayAvx512(KeyType key, size_t left_pos, size_t right_pos) const {
    assert(left_pos == 0 || key > keys_[left_pos - 1]);
    assert(right_pos >= element_array_size_ || key <= keys_[right_pos]);

    __m512i keyv = _mm512_set1_epi64(key);

    _mm_prefetch(reinterpret_cast<const char*>(&keys_[left_pos]), _MM_HINT_T0);
    if constexpr(PrefetchValue) {
        _mm_prefetch(reinterpret_cast<const char*>(&values_[left_pos]), _MM_HINT_T0);
    }

    for (size_t pos = left_pos + 8; pos < right_pos; pos += 8) {
      if constexpr(PrefetchValue) {
        _mm_prefetch(reinterpret_cast<const char*>(&values_[pos]), _MM_HINT_T0);
      }
      if (keys_[pos] >= key) {
        right_pos = pos;
        break;
      } else {
        left_pos = pos;
      }
    }

    __m512i ent2 = _mm512_loadu_si512((__m512i*)&keys_[left_pos]);
    __mmask8 mask2 = _mm512_cmpge_epu64_mask(ent2, keyv);
    if (mask2) {
      left_pos += __builtin_ctz(mask2);
    } else {
      left_pos += 8;
    }

    left_pos = std::min(left_pos, size_);

    assert(keys_[left_pos] >= key);
    if (left_pos > 0) {
      assert(keys_[left_pos - 1] < key);
    }
    return left_pos;
  }
#endif

#ifdef __AVX2__
  template<bool PrefetchValue = true>
  size_t LowerBoundOverKVArrayAvx2(KeyType key, size_t left_pos, size_t right_pos) const {
    assert(left_pos == 0 || key > keys_[left_pos - 1]);
    assert(right_pos >= element_array_size_ || key <= keys_[right_pos]);

    __m256i keyv = _mm256_set1_epi64x(key);

    _mm_prefetch(reinterpret_cast<const char*>(&keys_[left_pos]), _MM_HINT_T0);
    if constexpr(PrefetchValue) {
        _mm_prefetch(reinterpret_cast<const char*>(&values_[left_pos]), _MM_HINT_T0);
    }

    for (size_t pos = left_pos + 4; pos < right_pos; pos += 4) {
      if constexpr(PrefetchValue) {
        _mm_prefetch(reinterpret_cast<const char*>(&values_[pos]), _MM_HINT_T0);
      }
      if (keys_[pos] >= key) {
        right_pos = pos;
        break;
      } else {
        left_pos = pos;
      }
    }

    // Load 4 keys and compare using AVX2
    __m256i ent2 = _mm256_loadu_si256((__m256i*)&keys_[left_pos]);

    // For unsigned 64-bit comparison >= key, we need to check if each element >= key
    // AVX2 doesn't have direct unsigned comparison, so we use a workaround:
    // x >= y is equivalent to NOT(x < y)
    // For unsigned comparison, we can use: (x XOR 0x8000...) >= (y XOR 0x8000...)
    __m256i sign_bit = _mm256_set1_epi64x(0x8000000000000000ULL);
    __m256i ent2_signed = _mm256_xor_si256(ent2, sign_bit);
    __m256i keyv_signed = _mm256_xor_si256(keyv, sign_bit);

    // Now do signed comparison: ent2_signed >= keyv_signed
    __m256i cmp_gt = _mm256_cmpgt_epi64(ent2_signed, keyv_signed);
    __m256i cmp_eq = _mm256_cmpeq_epi64(ent2, keyv);
    __m256i cmp_ge = _mm256_or_si256(cmp_gt, cmp_eq);

    // Extract mask from comparison result
    int mask2 = _mm256_movemask_pd(_mm256_castsi256_pd(cmp_ge));

    if (mask2) {
      left_pos += __builtin_ctz(mask2);
    } else {
      left_pos += 4;
    }

    left_pos = std::min(left_pos, size_);

    assert(keys_[left_pos] >= key);
    if (left_pos > 0) {
      assert(keys_[left_pos - 1] < key);
    }
    return left_pos;
  }
#endif

  template<bool PrefetchValue = true>
  size_t LowerBoundOverKVArrayNaive(KeyType key, size_t left_pos, size_t right_pos) const {
    size_t answer = left_pos;
    if constexpr (PrefetchValue) {
      _mm_prefetch(reinterpret_cast<const char*>(&values_[answer]),
                   _MM_HINT_T0);
    }

    while (keys_[answer] < key) {
      answer++;
      if constexpr (PrefetchValue) {
        _mm_prefetch(reinterpret_cast<const char*>(&values_[answer]),
                      _MM_HINT_T0);
      }
    }

    assert(keys_[answer] >= key);
    if (answer > 0) {
      assert(keys_[answer - 1] < key);
    }
    assert(left_pos < right_pos);

    return answer;
  }

  // LowerBoundFromPosition: search directly in keys_ array (key-space positions)
  template<bool PrefetchValue = true>
  size_t LowerBoundFromPosition(KeyType key, size_t left_pos, size_t right_pos) const {
    return LowerBoundOverKVArray<PrefetchValue>(key, left_pos, right_pos);
  }

  // LowerBoundFuzzyFromPosition: fuzzy search directly in keys_ array (key-space positions)
  template<bool PrefetchValue = true>
  size_t LowerBoundFuzzyFromPosition(KeyType key, size_t left_pos, size_t right_pos) const {
    assert(right_pos >= left_pos && right_pos - left_pos <= B_);
    return left_pos;
  }

  // getters
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

  template <size_t kNumRequests>
  void BatchGetValue(size_t* idx, ValueType* values) {
    for (size_t i = 0; i < kNumRequests; i++) {
      assert(idx[i] < size_);
      values[i] = values_[idx[i]];
    }
  }

  EntryType GetKV(size_t idx) {
    // Find inspects the first padded sentinel when its lower bound is past
    // the last real key, then reports that the query is absent.
    assert(idx < size_ || (idx == size_ && idx < element_array_size_));
    return std::make_pair(keys_[idx], values_[idx]);
  }

  template <size_t kNumRequests>
  void BatchGetKV(size_t* idx, EntryType* kvs) {
    for (size_t i = 0; i < kNumRequests; i++) {
      assert(idx[i] < size_);
      kvs[i] = std::make_pair(keys_[idx[i]], values_[idx[i]]);
    }
  }

  void Print() {
    std::cout << "Layer ID: " << layer_id_ << std::endl;
    std::cout << "Size: " << size_ << std::endl;
    std::cout << "Element Array Size: " << element_array_size_ << std::endl;
    std::cout << "B: " << B_ << std::endl;
    std::cout << "finger density: " << finger_density_.numerator() << "/"
              << finger_density_.denominator() << std::endl;
    std::cout << std::endl;
  }

  size_t layer_id_, size_, element_array_size_;
  size_t B_;
  fh_index::internal::Fraction finger_density_;
  bool initialized_ = false;
  KeyType* keys_;
  ValueType* values_;
};

}  // namespace fh_index_ro
