#ifndef FH_INDEX_RO_H
#define FH_INDEX_RO_H

#include <cstdlib>
#include <stdexcept>

#include "fh_index_ro/layer/fh_layer_ro.h"
#include "fh_index_ro/layer/top_layer_ro.h"

namespace fh_index_ro {

template <typename KeyType, typename ValueType>
class FHIndexRO {
  using EntryType = std::pair<KeyType, ValueType>;

  static_assert(std::is_same_v<KeyType, uint64_t>);

 public:
  FHIndexRO(size_t top_level_threshold, size_t B, size_t F)
      : FHIndexRO(top_level_threshold, B,
                  fh_index::internal::Fraction(1, F)) {}

  FHIndexRO(size_t top_level_threshold, size_t B,
            fh_index::internal::Fraction finger_density)
      : size_(0), layer_count_(0),
        top_level_threshold_(top_level_threshold), B_(B),
        finger_density_(finger_density) {
    assert(finger_density_.IsPositive());
    if (!finger_density_.IsPositive()) std::abort();
  }

  void BulkLoad(const size_t size, const KeyType* keys,
                const ValueType* values) {
    if (size == 0) {
      throw std::invalid_argument("FHIndexRO::BulkLoad requires non-empty input");
    }

    size_ = size;
    layer_count_ = 0;

    constexpr KeyType key_max = std::numeric_limits<KeyType>::max();

    size_t array_size = fh_index::internal::RoundUp(size + ((keys[size - 1] != key_max) ? 1 : 0), fh_index::internal::kArrayPaddingSize);

    KeyType* keys_rounded = new KeyType[array_size];
    ValueType* values_rounded = new ValueType[array_size];
    parlay::parallel_for(0, size, [&](size_t i) {
      keys_rounded[i] = keys[i];
      values_rounded[i] = values[i];
    });
    for (size_t i = size; i < array_size; i++) {
      keys_rounded[i] = key_max;
      values_rounded[i] = std::numeric_limits<ValueType>::max();
    }

#ifndef NDEBUG
    // Check if keys are sorted and unique
    parlay::parallel_for(1, size,
                         [&](size_t i) { assert(keys_rounded[i] > keys_rounded[i - 1]); });
#endif
    assert(keys_rounded[array_size - 1] == key_max);

    auto [fgd_size, fgd_array_size, fgd_keys, fgd_values] =
        leaf_level_.BulkLoad(layer_count_ ++, size, array_size, std::move(keys_rounded),
                             std::move(values_rounded), B_, finger_density_);

    internal_levels_.clear();

    while (fgd_size > top_level_threshold_) {
      internal_levels_.emplace_back();
      // Internal levels use fixed B=16 and finger density 1.
      auto [nxt_fgd_size, nxt_fgd_array_size, nxt_fgd_keys, nxt_fgd_values] =
          internal_levels_.back().BulkLoad(layer_count_ ++, fgd_size, fgd_array_size,
                                           std::move(fgd_keys),
                                           std::move(fgd_values), 16,
                                           fh_index::internal::Fraction(1, 1));
      fgd_size = nxt_fgd_size;
      fgd_array_size = nxt_fgd_array_size;
      fgd_keys = nxt_fgd_keys;
      fgd_values = nxt_fgd_values;
    }
    top_level_.BulkLoad(layer_count_ ++, fgd_size, fgd_array_size, std::move(fgd_keys),
                        std::move(fgd_values));
    Print();
  }

  size_t LowerBoundFuzzy(const KeyType key) {
    std::pair<KeyType, FingerGroupDescriptorValue> descriptor;
    size_t idx = top_level_.LowerBound(key);
    descriptor = top_level_.GetKV(idx);
    for (int level = internal_levels_.size() - 1; level >= 0; level--) {
      idx = internal_levels_[level].LowerBound(key, descriptor);
      descriptor = internal_levels_[level].GetKV(idx);
    }
    return leaf_level_.LowerBoundFuzzy(key, descriptor);
  }

  size_t LowerBound(const KeyType key) {
    std::pair<KeyType, FingerGroupDescriptorValue> descriptor;
    size_t idx = top_level_.LowerBound(key);
    descriptor = top_level_.GetKV(idx);
    for (int level = internal_levels_.size() - 1; level >= 0; level--) {
      idx = internal_levels_[level].LowerBound(key, descriptor);
      descriptor = internal_levels_[level].GetKV(idx);
    }
    return leaf_level_.LowerBound(key, descriptor);
  }

  void Print() {
    std::cout << "Top level: " << std::endl;
    top_level_.Print();
    for (size_t i = internal_levels_.size(); i -- > 0;) {
      std::cout << "Internal level " << i << ": " << std::endl;
      internal_levels_[i].Print();
    }
    std::cout << "Leaf level: " << std::endl;
    leaf_level_.Print();
    std::cout << "------------------------" << std::endl;
  }

  // Scan: find up to `len` key-value pairs with key >= lower.
  // Results are written to `results` in ascending key order.
  // Returns the number of results found.
  size_t Scan(const KeyType lower, size_t len, EntryType* results) {
    size_t start_idx = LowerBound(lower);
    size_t count = std::min(len, size_ - start_idx);

    const KeyType* keys = leaf_level_.kv_layer_.keys_;
    const ValueType* values = leaf_level_.kv_layer_.values_;

    for (size_t i = 0; i < count; i++) {
      results[i] = {keys[start_idx + i], values[start_idx + i]};
    }
    return count;
  }

  void Find(const KeyType key, ValueType& value, bool& found) {
    size_t idx = LowerBound(key);
    EntryType result_kv = leaf_level_.GetKV(idx);
    found = (result_kv.first == key) && (idx < size_);
    value = result_kv.second;
  }

  size_t layer_count_;
  size_t size_;
  size_t top_level_threshold_;
  size_t B_;
  fh_index::internal::Fraction finger_density_;

  TopLayerRO<KeyType, FingerGroupDescriptorValue> top_level_;
  std::vector<FHLayerRO<KeyType, FingerGroupDescriptorValue>>
      internal_levels_;
  FHLayerRO<KeyType, ValueType>
      leaf_level_;
};
} // namespace fh_index_ro

#endif
