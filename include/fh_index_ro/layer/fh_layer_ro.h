#pragma once

#include <immintrin.h>

#include <vector>

#include "common_internal/construction/single_level_segmentor.h"
#include "common_internal/representation/finger_group_descriptor.h"
#include "common_internal/utility/hasher.h"
#include "common_internal/utility/math.h"
#include "../finger/finger.h"
#include "../storage/kv_layer_ro.h"
#include "parlay/parallel.h"

inline size_t predict_count = 0;
inline size_t predict_sum = 0;

namespace fh_index_ro {

// Import types from fh_index::internal
using FingerGroupDescriptorValue = fh_index::internal::FingerGroupDescriptorValue;
using FingerArray = fh_index::internal::FingerArray;

template <typename KeyType, typename ValueType>
class FHLayerRO {

 public:
  using EntryType = std::pair<KeyType, ValueType>;
  using KVLayerType = KVLayerRO<KeyType, ValueType>;
  using FGDType = fh_index::internal::FingerGroupDescriptor<KeyType, FingerRO>;
  using FingerType = FingerRO;

  FHLayerRO() {
    initialized_ = false;
    layer_id_ = 0;
    B_ = 0;
    finger_density_ = fh_index::internal::Fraction();
    finger_count_ = 0;
  }

  // Delete copy constructor to prevent copying
  FHLayerRO(const FHLayerRO& other) = delete;

  // Delete copy assignment operator to prevent copying
  FHLayerRO& operator=(const FHLayerRO& other) = delete;

  FHLayerRO(FHLayerRO&& other) noexcept
      : kv_layer_(std::move(other.kv_layer_)),
        finger_arrays_(std::move(other.finger_arrays_)) {
    layer_id_ = other.layer_id_;
    initialized_ = other.initialized_;
    B_ = other.B_;
    finger_density_ = other.finger_density_;
    finger_count_ = other.finger_count_;

    // Reset the other object's resources
    other.initialized_ = false;
  }

  FHLayerRO& operator=(FHLayerRO&& other) noexcept {
    if (this != &other) {
      Clear();

      kv_layer_ = std::move(other.kv_layer_);
      finger_arrays_ = std::move(other.finger_arrays_);
      layer_id_ = other.layer_id_;
      initialized_ = other.initialized_;
      B_ = other.B_;
      finger_density_ = other.finger_density_;
      finger_count_ = other.finger_count_;

      // Reset the other object's resources
      other.initialized_ = false;
    }
    return *this;
  }

  ~FHLayerRO() { Clear(); }

  void Clear() {
    if (initialized_) {
      // Destroy all FingerArrays
      parlay::parallel_for(0, finger_arrays_.size(), [&](size_t i) {
        FingerArray::Destroy<FingerType>(finger_arrays_[i]);
      });
      finger_arrays_.clear();
    }
    kv_layer_.Clear();
    initialized_ = false;
    layer_id_ = 0;
    B_ = 0;
    finger_density_ = fh_index::internal::Fraction();
    finger_count_ = 0;
  }

  std::tuple<size_t, size_t, KeyType*, FingerGroupDescriptorValue*> BulkLoad(
      size_t layer_id, size_t size, size_t array_size, KeyType*&& keys,
      ValueType*&& values, size_t B,
      fh_index::internal::Fraction finger_density) {
    Clear();
    initialized_ = true;
    layer_id_ = layer_id;
    B_ = B;
    finger_density_ = finger_density;

    // Delegate key-value storage to kv_layer_
    kv_layer_.BulkLoad(layer_id, size, array_size, std::move(keys),
                       std::move(values), B, finger_density);

    // Use kv_layer_'s keys directly for segmenting
    size_t key_size = kv_layer_.size_;
    KeyType* seg_keys = kv_layer_.keys_;

    // The overlapping-cover phase uses half the final density.
    auto covering_finger_density = finger_density.Halve();
    fh_index::internal::segmenting::Segmentor<KeyType> segmentor(
        key_size, seg_keys, B, covering_finger_density);
    segmentor.Run();

    size_t fgd_size = segmentor.fgd_size;
    size_t fgd_array_size = segmentor.fgd_array_size;
    KeyType* fgd_keys = segmentor.fgd_keys;
    FingerGroupDescriptorValue* fgd_values = segmentor.fgd_values;

    finger_count_ = segmentor.total_finger_count;
    size_t n = key_size;
    size_t block_size = B;

    // Create FingerArrays - one per FGD
    finger_arrays_ = parlay::sequence<FingerArray*>(fgd_size);

    // Parallel allocation of FingerArrays
    parlay::parallel_for(0, fgd_size, [&](size_t i) {
      // Calculate number of fingers for this FGD
      size_t finger_start = segmentor.finger_offsets[i];
      size_t finger_end = (i + 1 < fgd_size)
                          ? segmentor.finger_offsets[i + 1]
                          : finger_count_;
      size_t num_fingers = finger_end - finger_start;

      // Create FingerArray for this FGD
      finger_arrays_[i] = FingerArray::Create<FingerType>(num_fingers);

      // Populate fingers in this array
      FingerType* fingers =
          finger_arrays_[i]->template fingers<FingerType>();
      for (size_t j = 0; j < num_fingers; j++) {
        size_t global_finger_idx = finger_start + j;
        size_t l_pos = segmentor.finger_indexes[global_finger_idx];
        // notice: n - 1, not n. otherwise predict gives overflowing right_pos
        size_t r_pos = (global_finger_idx == finger_count_ - 1)
                           ? n
                           : segmentor.finger_indexes[global_finger_idx + 1];
        size_t finger_size = r_pos - l_pos;
        fingers[j] = FingerRO(l_pos, finger_size);
        assert(r_pos >= l_pos && finger_size <= block_size);
      }

      // Set FGD's finger_array_ pointer
      fgd_values[i].finger_array_ = finger_arrays_[i];
    });

    // A non-empty layer needs one finger even when rho * size is below one.
    assert(finger_count_ <= std::max<size_t>(1, finger_density.FloorMultiply(size)) &&
           "Violate finger-density constraint");

    auto [owned_keys, owned_values] = segmentor.ReleaseFGDArrays();
    return std::make_tuple(fgd_size, fgd_array_size, owned_keys, owned_values);
  }

  template<bool PrefetchValue = true>
  size_t LowerBoundFuzzy(
      KeyType key,
      std::pair<KeyType, FingerGroupDescriptorValue> left_descriptor) const {
    FGDType fgd(left_descriptor);
    size_t finger_idx = fgd.Decode(key);

    assert(finger_idx < fgd.GetSize());
    const FingerRO& finger = *fgd.GetFingerAddr(finger_idx);

    uint64_t left_pos = 0, right_pos = 0;
    finger.Predict(key, left_pos, right_pos);

    // Delegate to kv_layer_ (positions are already in key-space)
    return kv_layer_.template LowerBoundFuzzyFromPosition<PrefetchValue>(key, left_pos, right_pos);
  }

  template<bool PrefetchValue = true>
  size_t LowerBound(
      KeyType key,
      std::pair<KeyType, FingerGroupDescriptorValue> left_descriptor) const {
    FGDType fgd(left_descriptor);
    size_t finger_idx = fgd.Decode(key);

    assert(finger_idx < fgd.GetSize());
    const FingerRO& finger = *fgd.GetFingerAddr(finger_idx);

    uint64_t left_pos = 0, right_pos = 0;
    finger.Predict(key, left_pos, right_pos);

    // Delegate to kv_layer_ (positions are already in key-space)
    size_t result = kv_layer_.template LowerBoundFromPosition<PrefetchValue>(key, left_pos, right_pos);

    // Validation assertions
    assert(kv_layer_.keys_[result] >= key);
    if (result > 0) {
      assert(kv_layer_.keys_[result - 1] < key);
    }

    return result;
  }

  // getters
  size_t GetSize() {
    assert(initialized_);
    return kv_layer_.GetSize();
  }

  template <size_t kNumRequests>
  void BatchGetKey(size_t* idx, KeyType* keys) {
    kv_layer_.template BatchGetKey<kNumRequests>(idx, keys);
  }

  template <size_t kNumRequests>
  void BatchGetValue(size_t* idx, ValueType* values) {
    kv_layer_.template BatchGetValue<kNumRequests>(idx, values);
  }

  EntryType GetKV(size_t idx) {
    return kv_layer_.GetKV(idx);
  }

  template <size_t kNumRequests>
  void BatchGetKV(size_t* idx, EntryType* kvs) {
    kv_layer_.template BatchGetKV<kNumRequests>(idx, kvs);
  }

  void Print() {
    std::cout << "Layer ID: " << layer_id_ << std::endl;
    std::cout << "Finger Count: " << finger_count_ << std::endl;
    std::cout << "Size: " << kv_layer_.size_ << std::endl;
    std::cout << "Element Array Size: " << kv_layer_.element_array_size_ << std::endl;
    std::cout << std::endl;
  }

 public:
  size_t layer_id_;
  bool initialized_ = false;
  size_t B_;
  fh_index::internal::Fraction finger_density_;
  size_t finger_count_;
  parlay::sequence<FingerArray*> finger_arrays_;

  // Delegate key-value storage to KVLayerRO
  KVLayerType kv_layer_;
};

}  // namespace fh_index_ro
