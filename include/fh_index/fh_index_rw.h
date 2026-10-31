#pragma once

#include <chrono>
#include <atomic>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <cassert>
#include <limits>
#include <new>
#include <thread>
#include <type_traits>
#include <vector>

#include "fh_index_rw/finger/finger_rw.h"
#include "fh_index_rw/retrain_stats.h"
#include "fh_index_rw/storage/kv_layer_rw.h"
#include "common_internal/construction/multi_level_segmentor.h"
#include "common_internal/memory/epoch.h"
#include "common_internal/memory/epoch_reclamation.h"
#include "common_internal/representation/finger_group_descriptor.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"

namespace fh_index_rw {

template <typename KeyType, typename ValueType, bool TrackRetrain = false>
class FHIndexRW {
  using EntryType = std::pair<KeyType, ValueType>;
  using FGDValueType = fh_index::internal::FingerGroupDescriptorValue;
  using FingerArray = fh_index::internal::FingerArray;
  using FGDType =
      fh_index::internal::FingerGroupDescriptor<KeyType, FingerRW>;
  using LeafKVLayerType = KVLayerRW<KeyType, ValueType>;
  using InternalKVLayerType = KVLayerRW<KeyType, FGDValueType>;
  using LeafBlockType = typename LeafKVLayerType::BlockType;
  using InternalBlockType = Block<KeyType, FGDValueType>;

  static_assert(std::is_same_v<KeyType, uint64_t>);

  enum class UpdateResult {
    kSuccess,
    kKeyExists,
    kKeyNotFound,
    kRebuilding,
    kOverflow
  };

  enum class LookupResult {
    kSuccess,
    kRebuilding,
    kOverflow
  };

  static bool FindInLeafFHB(
      KeyType key, std::pair<KeyType, FGDValueType> descriptor,
      KeyType& out_key, ValueType& out_value) {
    FGDType fgd(descriptor);
    const size_t finger_idx = fgd.Decode(key);
    assert(finger_idx < fgd.GetSize());
    const FingerRW& finger = *fgd.GetFingerAddr(finger_idx);

    while (true) {
      const uint64_t before = finger.GetData();
      if (FingerRW::IsWriteLocked(before)) {
        _mm_pause();
        continue;
      }

      LeafBlockType* block = FingerRW::UnpackBlock<LeafBlockType>(before);
      const uint16_t block_size = FingerRW::UnpackSize(before);
      const uint16_t capacity = FingerRW::UnpackCapacity(before);
      if (block == nullptr) {
        if (finger.GetData() == before) return false;
        continue;
      }

      const bool found = LeafKVLayerType::Get(
          key, block, block_size, capacity, out_key, out_value);
      std::atomic_signal_fence(std::memory_order_acq_rel);
      if (finger.GetData() == before) return found;
    }
  }

  static bool FuzzyFindInLeafFHB(
      KeyType key, std::pair<KeyType, FGDValueType> descriptor) {
    FGDType fgd(descriptor);
    const size_t finger_idx = fgd.Decode(key);
    assert(finger_idx < fgd.GetSize());
    const FingerRW& finger = *fgd.GetFingerAddr(finger_idx);

    LeafBlockType* block = nullptr;
    uint16_t block_size = 0;
    finger.Predict<LeafBlockType>(key, block, block_size);
    return block != nullptr;
  }

  static void FindNextDescriptor(
      KeyType key, std::pair<KeyType, FGDValueType> descriptor,
      KeyType& out_key, FGDValueType& out_value) {
    FGDType fgd(descriptor);
    const size_t finger_idx = fgd.Decode(key);
    assert(finger_idx < fgd.GetSize());
    const FingerRW& finger = *fgd.GetFingerAddr(finger_idx);

    const uint64_t finger_data = finger.GetData();
    InternalBlockType* block =
        FingerRW::UnpackBlock<InternalBlockType>(finger_data);
    const uint16_t block_size = FingerRW::UnpackSize(finger_data);
    const uint16_t capacity = FingerRW::UnpackCapacity(finger_data);
    assert(block != nullptr &&
           "Internal FHB should not contain nullptr blocks");

    const bool found = InternalKVLayerType::GetLowerBoundKV(
        key, block, block_size, capacity, out_key, out_value);
    assert(found &&
           "AugmentLevel guarantees lower bound is in the current block");
    (void)found;
  }

  static LookupResult FindNextDescriptorForMutation(
      KeyType key, std::pair<KeyType, FGDValueType> descriptor,
      KeyType& out_key, FGDValueType& out_value, size_t& out_finger_idx,
      size_t overflow_threshold) {
    FGDType fgd(descriptor);
    const size_t finger_idx = fgd.Decode(key);
    assert(finger_idx < fgd.GetSize());
    const FingerRW& finger = *fgd.GetFingerAddr(finger_idx);

    const uint64_t finger_data = finger.GetData();
    if (FingerRW::IsRebuilding(finger_data)) {
      return LookupResult::kRebuilding;
    }

    InternalBlockType* block =
        FingerRW::UnpackBlock<InternalBlockType>(finger_data);
    const uint16_t block_size = FingerRW::UnpackSize(finger_data);
    if (block_size >= overflow_threshold) {
      return LookupResult::kOverflow;
    }
    assert(block != nullptr &&
           "Internal FHB should not contain nullptr blocks");

    const uint16_t capacity = FingerRW::UnpackCapacity(finger_data);
    const bool found = InternalKVLayerType::GetLowerBoundKV(
        key, block, block_size, capacity, out_key, out_value);
    assert(found &&
           "AugmentLevel guarantees lower bound is in the current block");
    (void)found;

    out_finger_idx = finger_idx;
    return LookupResult::kSuccess;
  }

  static UpdateResult InsertIntoLeafFHB(
      KeyType key, ValueType value,
      std::pair<KeyType, FGDValueType> descriptor,
      size_t overflow_threshold) {
    FGDType fgd(descriptor);
    const size_t finger_idx = fgd.Decode(key);
    assert(finger_idx < fgd.GetSize());
    FingerRW* finger = const_cast<FingerRW*>(fgd.GetFingerAddr(finger_idx));

    while (true) {
      const uint64_t old_data = finger->GetData();
      if (FingerRW::IsRebuilding(old_data)) {
        return UpdateResult::kRebuilding;
      }
      if (FingerRW::IsWriteLocked(old_data)) {
        _mm_pause();
        continue;
      }

      const uint64_t locked_data = FingerRW::SetWriteLockBit(old_data);
      uint64_t expected = old_data;
      if (!finger->CompareExchange(expected, locked_data)) continue;

      LeafBlockType* old_block =
          FingerRW::UnpackBlock<LeafBlockType>(old_data);
      const uint16_t old_size = FingerRW::UnpackSize(old_data);
      const uint16_t old_capacity = FingerRW::UnpackCapacity(old_data);
      const uint8_t old_version = FingerRW::UnpackVersion(old_data);

      if (old_size >= overflow_threshold) {
        finger->Publish(old_data);
        return UpdateResult::kOverflow;
      }

      auto [insert_pos, key_exists] =
          LeafKVLayerType::FindKeyOrInsertionPosition(
              old_block, old_size, key);
      if (key_exists) {
        finger->Publish(old_data);
        return UpdateResult::kKeyExists;
      }

      if (old_block != nullptr && old_size < old_capacity &&
          !FingerRW::IsVersionExhausted(old_data)) {
        LeafKVLayerType::InsertInPlaceAt(
            insert_pos, key, value, old_block, old_size, old_capacity);
        finger->Publish(FingerRW::Pack(
            old_block, old_size + 1, old_capacity, old_version + 1));
        return UpdateResult::kSuccess;
      }

      const size_t replacement_capacity =
          old_size < old_capacity ? old_capacity : 0;
      auto [new_block, new_size, new_capacity] =
          LeafKVLayerType::CreateBlockWithInsertAt(
              old_block, old_size, old_capacity, insert_pos, key, value,
              overflow_threshold, replacement_capacity);
      finger->Publish(FingerRW::Pack(
          new_block, static_cast<uint16_t>(new_size),
          static_cast<uint16_t>(new_capacity), 0));
      LeafBlockType::Retire(old_block, old_capacity);
      return UpdateResult::kSuccess;
    }
  }

  static UpdateResult UpdateLeafFHB(
      KeyType key, ValueType value,
      std::pair<KeyType, FGDValueType> descriptor) {
    FGDType fgd(descriptor);
    const size_t finger_idx = fgd.Decode(key);
    assert(finger_idx < fgd.GetSize());
    FingerRW* finger = const_cast<FingerRW*>(fgd.GetFingerAddr(finger_idx));

    while (true) {
      const uint64_t old_data = finger->GetData();
      if (FingerRW::IsRebuilding(old_data)) {
        return UpdateResult::kRebuilding;
      }
      if (FingerRW::IsWriteLocked(old_data)) {
        _mm_pause();
        continue;
      }
      assert(!FingerRW::IsShadow(old_data) &&
             "Update requires an owning (non-shadow) leaf finger");

      const uint64_t locked_data = FingerRW::SetWriteLockBit(old_data);
      uint64_t expected = old_data;
      if (!finger->CompareExchange(expected, locked_data)) continue;

      LeafBlockType* block =
          FingerRW::UnpackBlock<LeafBlockType>(old_data);
      const uint16_t block_size = FingerRW::UnpackSize(old_data);
      const uint16_t capacity = FingerRW::UnpackCapacity(old_data);
      const uint8_t version = FingerRW::UnpackVersion(old_data);

      auto [update_pos, key_found] =
          LeafKVLayerType::FindKeyOrInsertionPosition(
              block, block_size, key);
      if (!key_found) {
        finger->Publish(old_data);
        return UpdateResult::kKeyNotFound;
      }

      if (!FingerRW::IsVersionExhausted(old_data)) {
        LeafKVLayerType::UpdateInPlaceAt(
            update_pos, value, block, block_size, capacity);
        finger->Publish(FingerRW::Pack(
            block, block_size, capacity, version + 1));
        return UpdateResult::kSuccess;
      }

      auto [new_block, new_size, new_capacity] =
          LeafKVLayerType::CreateBlockWithUpdateAt(
              block, block_size, capacity, update_pos, value);
      assert(new_block != nullptr);
      finger->Publish(FingerRW::Pack(
          new_block, static_cast<uint16_t>(new_size),
          static_cast<uint16_t>(new_capacity), 0));
      LeafBlockType::Retire(block, capacity);
      return UpdateResult::kSuccess;
    }
  }

  static UpdateResult DeleteFromLeafFHB(
      KeyType key, std::pair<KeyType, FGDValueType> descriptor) {
    FGDType fgd(descriptor);
    const size_t finger_idx = fgd.Decode(key);
    assert(finger_idx < fgd.GetSize());
    FingerRW* finger = const_cast<FingerRW*>(fgd.GetFingerAddr(finger_idx));

    while (true) {
      const uint64_t old_data = finger->GetData();
      if (FingerRW::IsRebuilding(old_data)) {
        return UpdateResult::kRebuilding;
      }
      if (FingerRW::IsWriteLocked(old_data)) {
        _mm_pause();
        continue;
      }

      const uint64_t locked_data = FingerRW::SetWriteLockBit(old_data);
      uint64_t expected = old_data;
      if (!finger->CompareExchange(expected, locked_data)) continue;

      LeafBlockType* block =
          FingerRW::UnpackBlock<LeafBlockType>(old_data);
      const uint16_t old_size = FingerRW::UnpackSize(old_data);
      const uint16_t capacity = FingerRW::UnpackCapacity(old_data);
      const uint8_t version = FingerRW::UnpackVersion(old_data);

      auto [delete_pos, key_found] =
          LeafKVLayerType::FindKeyOrInsertionPosition(block, old_size, key);
      if (!key_found) {
        finger->Publish(old_data);
        return UpdateResult::kKeyNotFound;
      }

      if (!FingerRW::IsVersionExhausted(old_data)) {
        LeafKVLayerType::DeleteInPlaceAt(
            delete_pos, block, old_size, capacity);
        finger->Publish(FingerRW::Pack(
            block, old_size - 1, capacity, version + 1));
        return UpdateResult::kSuccess;
      }

      auto [new_block, new_size, new_capacity] =
          LeafKVLayerType::CreateBlockWithDeleteAt(
              block, old_size, capacity, delete_pos);
      finger->Publish(FingerRW::Pack(
          new_block, static_cast<uint16_t>(new_size),
          static_cast<uint16_t>(new_capacity), 0));
      LeafBlockType::Retire(block, capacity);
      return UpdateResult::kSuccess;
    }
  }

  static void PrintFingerLayer(
      size_t finger_count, size_t B,
      const fh_index::internal::Fraction& finger_density) {
    std::cout << "Finger Count: " << finger_count << std::endl;
    std::cout << "B: " << B << ", finger density: "
              << finger_density.numerator() << "/"
              << finger_density.denominator() << std::endl;
  }

  // One 64-bit publication unit identifies a complete root generation.
  // Root blocks are 32-byte aligned, so their five zero low address bits store
  // height - 1. The high size field stores size - 1. Consequently, the common
  // one-entry, one-level root is represented by its raw pointer and can bypass
  // all unpacking. Pointer, size, height, and rebuild ownership still change
  // atomically.
  struct RootHandle {
    static constexpr size_t kAlignment = 32;
    static constexpr uint64_t kHeightMask = kAlignment - 1;
    static constexpr uint64_t kRawPointerMask = (1ULL << 48) - 1;
    static constexpr uint64_t kPointerMask = kRawPointerMask & ~kHeightMask;
    static constexpr uint64_t kSizeMask = (1ULL << 14) - 1;
    static constexpr int kSizeShift = 48;
    static constexpr uint64_t kEncodedSizeMask =
        kSizeMask << kSizeShift;
    static constexpr uint64_t kRebuildingBit = 1ULL << 63;
    static constexpr uint64_t kDirectMetadataMask =
        kHeightMask | kEncodedSizeMask | kRebuildingBit;

    static uint64_t Pack(InternalBlockType* block, size_t size,
                         size_t height) {
      assert(block != nullptr);
      assert(size > 0 && size <= kSizeMask);
      assert(height > 0 && height <= kHeightMask + 1);
      const uint64_t ptr = reinterpret_cast<uint64_t>(block);
      assert((ptr & ~kRawPointerMask) == 0);
      assert((ptr & kHeightMask) == 0);
      return (ptr & kPointerMask) |
             static_cast<uint64_t>(height - 1) |
             (static_cast<uint64_t>(size - 1) << kSizeShift);
    }

    static InternalBlockType* Block(uint64_t data) {
      return reinterpret_cast<InternalBlockType*>(data & kPointerMask);
    }

    static size_t Size(uint64_t data) {
      return ((data >> kSizeShift) & kSizeMask) + 1;
    }

    static size_t Height(uint64_t data) {
      assert(Block(data) != nullptr);
      return LevelsBelowRoot(data) + 1;
    }

    static size_t LevelsBelowRoot(uint64_t data) {
      return data & kHeightMask;
    }

    static bool IsRebuilding(uint64_t data) {
      return (data & kRebuildingBit) != 0;
    }

    static uint64_t SetRebuilding(uint64_t data) {
      return data | kRebuildingBit;
    }

    static bool IsDirect(uint64_t data) {
      return (data & kDirectMetadataMask) == 0;
    }

    static InternalBlockType* DirectBlock(uint64_t data) {
      assert(IsDirect(data));
      return reinterpret_cast<InternalBlockType*>(data);
    }

  };
  static_assert(std::atomic<uint64_t>::is_always_lock_free);

 public:
  // Result of Rebuild operation
  enum class RebuildResult {
    kSuccess,      // Rebuild succeeded
    kRebuilding,   // Another thread is already rebuilding this subtree
    kCASFailed     // Final CAS failed, caller should retry
  };

  struct RootMetadata {
    size_t root_size;
    size_t height;
  };

  FHIndexRW(size_t top_level_threshold, size_t B, size_t B_overflow, size_t F,
            size_t internal_B, size_t internal_B_overflow, size_t internal_F)
      : FHIndexRW(top_level_threshold, B, B_overflow,
                  fh_index::internal::Fraction(1, F), internal_B,
                  internal_B_overflow,
                  fh_index::internal::Fraction(1, internal_F)) {}

  FHIndexRW(size_t top_level_threshold, size_t B, size_t B_overflow,
            fh_index::internal::Fraction finger_density, size_t internal_B,
            size_t internal_B_overflow,
            fh_index::internal::Fraction internal_finger_density)
      : top_level_threshold_(top_level_threshold),
        B_(B), B_overflow_(B_overflow), finger_density_(finger_density),
        internal_B_(internal_B), internal_B_overflow_(internal_B_overflow),
        internal_finger_density_(internal_finger_density),
        root_data_(0) {
    assert(B_overflow_ <= FingerRW::kSizeMask);
    assert(internal_B_overflow_ <= FingerRW::kSizeMask);
    assert(finger_density_.IsPositive() &&
           internal_finger_density_.IsPositive());
    if (B_overflow_ > FingerRW::kSizeMask ||
        internal_B_overflow_ > FingerRW::kSizeMask ||
        !finger_density_.IsPositive() ||
        !internal_finger_density_.IsPositive()) {
      std::abort();
    }
  }

  ~FHIndexRW() {
    uint64_t root_data = root_data_.load(std::memory_order_relaxed);
    if (root_data != 0) {
      DestroyAllFingerArrays(root_data);
      DestroyRootBlock(root_data);
    }
  }

  void ResetRetrainStats() {
    retrain_stats_.Reset();
  }

  RetrainStatsSnapshot GetRetrainStats() const {
    return retrain_stats_.GetSnapshot();
  }

  // Destroy live blocks and their FingerArrays bottom-up. Retired generations
  // remain owned by the epoch manager and must not be freed here.
  void DestroyAllFingerArrays(uint64_t root_data) {
    // Destroy FingerArrays for each layer from leaf to highest internal
    for (size_t layer_id = 0; layer_id < RootHandle::Height(root_data);
         layer_id++) {
      auto [finger_count, finger_arrays] =
          GetFingerLayerMetadataFromRoot(layer_id, root_data);
      parlay::parallel_for(0, finger_arrays.size(), [&](size_t i) {
        FingerArray* fa = finger_arrays[i];
        const FingerRW* fingers = fa->fingers<FingerRW>();
        for (size_t j = 0; j < fa->size; ++j) {
          const uint64_t data = fingers[j].GetData();
          const size_t capacity = FingerRW::UnpackCapacity(data);
          if (layer_id == 0) {
            auto* block = FingerRW::UnpackBlock<LeafBlockType>(data);
            if (block != nullptr) LeafBlockType::Destroy(block, capacity);
          } else {
            auto* block = FingerRW::UnpackBlock<InternalBlockType>(data);
            if (block != nullptr) InternalBlockType::Destroy(block, capacity);
          }
        }
        FingerArray::Destroy<FingerRW>(fa);
      });
    }
  }

  // Construct layers using MultiLevelSegmentor, returning FGD data for the topmost layer.
  // height=0: build until fgd_size <= top_level_threshold (used by BulkLoad)
  // height>=1: build exactly `height` levels (1=leaf only, 2=leaf+1 internal, etc.)
  // max_key: max key for the last FGD (UINT64_MAX for full dataset, or subtree max for rebuild)
  // Returns {layer_count, fgd_size, fgd_keys, fgd_values} for the topmost constructed layer.
  static std::tuple<size_t, size_t, KeyType*, FGDValueType*> Construct(
      const size_t size, const KeyType* keys, const ValueType* values,
      size_t height, size_t B, fh_index::internal::Fraction finger_density,
      size_t internal_B,
      fh_index::internal::Fraction internal_finger_density,
      size_t top_level_threshold,
      KeyType max_key = std::numeric_limits<KeyType>::max()) {
    using namespace fh_index::internal::segmenting;

#ifndef NDEBUG
    // Check if keys are sorted and unique
    parlay::parallel_for(1, size,
                         [&](size_t i) { assert(keys[i] > keys[i - 1]); });
#endif

    // Run MultiLevelSegmentor
    // When height > 0, use height_limit; when height == 0, use top_level_threshold
    // MultiLevelSegmentor requires exactly one to be > 0
    LevelConfig leaf_config(B, finger_density);
    LevelConfig internal_config(internal_B, internal_finger_density);
    size_t height_limit = (height > 0) ? height : 0;
    size_t threshold = (height > 0) ? 0 : top_level_threshold;
    MultiLevelSegmentor<KeyType> segmentor(
        size, keys, leaf_config, internal_config, height_limit, threshold, max_key);
    segmentor.Run();

    size_t num_levels = segmentor.GetLevelCount();

    // Process each level: create blocks and FingerArrays
    // We need to process levels bottom-up because internal blocks store FGDValueType
    // which includes finger_array_ pointers from the level below

    // First, build leaf level (level 0)
    {
      const auto& level = segmentor.GetLevel(0);
      size_t fgd_size = level.size();
      size_t finger_count = level.finger_count();
      const size_t* breakpoints = level.finger_indexes.data();

      // Create leaf blocks
      auto blocks = LeafKVLayerType::BulkLoad(
          keys, values, size, breakpoints, finger_count);

      // Create FingerArrays for leaf level
      parlay::parallel_for(0, fgd_size, [&](size_t i) {
        size_t finger_start = level.finger_offsets[i];
        size_t finger_end = level.finger_offsets[i + 1];
        size_t num_fingers = finger_end - finger_start;

        FingerArray* fa = FingerArray::Create<FingerRW>(num_fingers);
        FingerRW* fingers = fa->fingers<FingerRW>();

        for (size_t j = 0; j < num_fingers; j++) {
          size_t global_idx = finger_start + j;
          size_t start_pos = breakpoints[global_idx];
          size_t end_pos = (global_idx + 1 < finger_count) ? breakpoints[global_idx + 1] : size;
          size_t block_size = end_pos - start_pos;
          assert(block_size <= B &&
                 "Materialized leaf block exceeds the configured bound");
          fingers[j] = FingerRW(
              blocks[global_idx], block_size,
              ComputeBlockCapacity(block_size), 0);
        }

        // Set finger_array_ in fgd_values (need non-const access)
        const_cast<LevelData<KeyType>&>(level).fgd_values[i].finger_array_ = fa;
      });
    }

    // Build internal levels (level 1 to num_levels-1)
    for (size_t lvl = 1; lvl < num_levels; ++lvl) {
      const auto& level = segmentor.GetLevel(lvl);
      const auto& lower_level = segmentor.GetLevel(lvl - 1);

      size_t fgd_size = level.size();
      size_t finger_count = level.finger_count();
      size_t lower_size = lower_level.size();
      const size_t* breakpoints = level.finger_indexes.data();

      // Create internal blocks from lower level's fgd data
      auto blocks = InternalKVLayerType::BulkLoad(
          lower_level.fgd_keys.data(), lower_level.fgd_values.data(),
          lower_size, breakpoints, finger_count);

      // Create FingerArrays for this internal level
      parlay::sequence<FingerArray*> finger_arrays(fgd_size);

      parlay::parallel_for(0, fgd_size, [&](size_t i) {
        size_t finger_start = level.finger_offsets[i];
        size_t finger_end = level.finger_offsets[i + 1];
        size_t num_fingers = finger_end - finger_start;

        FingerArray* fa = FingerArray::Create<FingerRW>(num_fingers);
        FingerRW* fingers = fa->fingers<FingerRW>();

        for (size_t j = 0; j < num_fingers; j++) {
          size_t global_idx = finger_start + j;
          size_t start_pos = breakpoints[global_idx];
          size_t end_pos = (global_idx + 1 < finger_count) ? breakpoints[global_idx + 1] : lower_size;
          size_t block_size = end_pos - start_pos;
          assert(block_size > 0 && "After AugmentLevel, internal fingers should not be empty");
          assert(block_size <= internal_B &&
                 "Materialized internal block exceeds the configured bound");
          fingers[j] = FingerRW(
              blocks[global_idx], block_size,
              ComputeBlockCapacity(block_size), 0);
        }

        finger_arrays[i] = fa;
        const_cast<LevelData<KeyType>&>(level).fgd_values[i].finger_array_ = fa;
      });
    }

    // Return topmost level's data
    const auto& top_level = segmentor.GetLevel(num_levels - 1);
    size_t fgd_size = top_level.size();

    // Allocate and copy fgd_keys and fgd_values (caller takes ownership)
    KeyType* fgd_keys = new KeyType[fgd_size];
    FGDValueType* fgd_values = new FGDValueType[fgd_size];

    parlay::parallel_for(0, fgd_size, [&](size_t i) {
      fgd_keys[i] = top_level.fgd_keys[i];
      fgd_values[i] = top_level.fgd_values[i];
    });

    return {num_levels, fgd_size, fgd_keys, fgd_values};
  }

  void BulkLoad(const size_t size, const KeyType* keys,
                const ValueType* values) {
    // BulkLoad is an exclusive operation and synchronously releases the
    // previous tree.
    uint64_t old_root_data = root_data_.exchange(0, std::memory_order_acq_rel);
    if (old_root_data != 0) {
      DestroyAllFingerArrays(old_root_data);
      DestroyRootBlock(old_root_data);
    }

    auto [layer_count, fgd_size, fgd_keys, fgd_values] = Construct(
        size, keys, values, 0, B_, finger_density_, internal_B_,
        internal_finger_density_, top_level_threshold_);

    // Build root block
    auto [block, capacity] = InternalBlockType::CreateWithCapacity(fgd_size);

    KeyType* block_keys = block->keys();
    FGDValueType* block_values = block->values(capacity);
    for (size_t i = 0; i < fgd_size; i++) {
      block_keys[i] = fgd_keys[i];
      block_values[i] = fgd_values[i];
    }

    // Free final fgd arrays
    delete[] fgd_keys;
    delete[] fgd_values;

    uint64_t new_root_data = CreateRootData(block, fgd_size, layer_count);
    root_data_.store(new_root_data, std::memory_order_release);

    Print();
  }

  void Find(const KeyType key, ValueType& value, bool& found) const {
    memory::OperationEpochGuard guard;
    uint64_t root_data = LoadRootData();
    if (root_data == 0) {
      found = false;
      return;
    }
    if (RootHandle::IsDirect(root_data)) {
      const InternalBlockType* root_block =
          RootHandle::DirectBlock(root_data);
      std::pair<KeyType, FGDValueType> descriptor = {
          root_block->keys()[0],
          root_block->values(ComputeBlockCapacity(1))[0]};
      KeyType out_key;
      found = FindInLeafFHB(key, descriptor, out_key, value);
      if (found) {
        found = (out_key == key);
      }
      return;
    }
    const InternalBlockType* root_block = RootHandle::Block(root_data);
    size_t root_size = RootHandle::Size(root_data);

    // Lower bound on the immutable root snapshot directly.
    size_t pos =
        lower_bound::SearchInternal(key, root_block->keys(), root_size);
    assert(pos < root_size &&
           "Root block should always contain lowerbound");

    size_t root_capacity = ComputeBlockCapacity(root_size);
    KeyType out_key = root_block->keys()[pos];
    FGDValueType out_value = root_block->values(root_capacity)[pos];

    // Traverse internal levels: FGD decode -> finger predict -> lower bound.
    for (int level = static_cast<int>(RootHandle::LevelsBelowRoot(root_data));
         level > 0; --level) {
      FGDType fgd({out_key, out_value});
      size_t finger_idx = fgd.Decode(key);
      const FingerRW& finger = *fgd.GetFingerAddr(finger_idx);

      const uint64_t finger_data = finger.GetData();
      InternalBlockType* block =
          FingerRW::UnpackBlock<InternalBlockType>(finger_data);
      const uint16_t block_size = FingerRW::UnpackSize(finger_data);
      const uint16_t capacity = FingerRW::UnpackCapacity(finger_data);
      pos = lower_bound::SearchInternal(key, block->keys(), block_size);
      assert(pos < block_size &&
             "AugmentLevel guarantees lowerbound is in current block");

      out_key = block->keys()[pos];
      out_value = block->values(capacity)[pos];
    }

    // Search in leaf level (level 0)
    std::pair<KeyType, FGDValueType> descriptor = {out_key, out_value};
    found = FindInLeafFHB(key, descriptor, out_key, value);
    if (found) {
      found = (out_key == key);
    }
  }

  // FuzzyFind: same as Find but skips final block search at leaf level.
  // Used for measuring model inference time.
  void FuzzyFind(const KeyType key, ValueType& value, bool& found) const {
    memory::OperationEpochGuard guard;
    uint64_t root_data = LoadRootData();
    if (root_data == 0) {
      found = false;
      return;
    }
    if (RootHandle::IsDirect(root_data)) {
      const InternalBlockType* root_block =
          RootHandle::DirectBlock(root_data);
      std::pair<KeyType, FGDValueType> descriptor = {
          root_block->keys()[0],
          root_block->values(ComputeBlockCapacity(1))[0]};
      found = FuzzyFindInLeafFHB(key, descriptor);
      return;
    }
    const InternalBlockType* root_block = RootHandle::Block(root_data);
    size_t root_size = RootHandle::Size(root_data);

    size_t pos =
        lower_bound::SearchInternal(key, root_block->keys(), root_size);
    assert(pos < root_size &&
           "Root block should always contain lowerbound");

    size_t root_capacity = ComputeBlockCapacity(root_size);
    KeyType out_key = root_block->keys()[pos];
    FGDValueType out_value = root_block->values(root_capacity)[pos];

    for (int level = static_cast<int>(RootHandle::LevelsBelowRoot(root_data));
         level > 0; --level) {
      FGDType fgd({out_key, out_value});
      size_t finger_idx = fgd.Decode(key);
      const FingerRW& finger = *fgd.GetFingerAddr(finger_idx);

      const uint64_t finger_data = finger.GetData();
      InternalBlockType* block =
          FingerRW::UnpackBlock<InternalBlockType>(finger_data);
      const uint16_t block_size = FingerRW::UnpackSize(finger_data);
      const uint16_t capacity = FingerRW::UnpackCapacity(finger_data);
      pos = lower_bound::SearchInternal(key, block->keys(), block_size);
      assert(pos < block_size &&
             "AugmentLevel guarantees lowerbound is in current block");

      out_key = block->keys()[pos];
      out_value = block->values(capacity)[pos];
    }

    std::pair<KeyType, FGDValueType> descriptor = {out_key, out_value};
    found = FuzzyFindInLeafFHB(key, descriptor);
  }

  // Scan: find up to `len` key-value pairs with key >= lower.
  // Results are written to `results` in ascending key order.
  // Returns the number of results found.
  //
  // Uses path-based traversal: navigates from root to leaf to find the starting
  // block, then advances through successive blocks using the FGD/Finger hierarchy.
  size_t Scan(const KeyType lower, size_t len, EntryType* results) const {
    if (len == 0) {
      return 0;
    }

    memory::OperationEpochGuard guard;
    uint64_t root_data = LoadRootData();
    if (root_data == 0) {
      return 0;
    }
    const InternalBlockType* root_block = RootHandle::Block(root_data);
    size_t root_size = RootHandle::Size(root_data);
    const size_t height = RootHandle::Height(root_data);

    // The common single-level case needs no hierarchy cursor.  Walk the leaf
    // finger arrays directly and stage a small window of finger snapshots so
    // that the corresponding blocks can be prefetched before they are copied.
    // Each bucket is still validated independently, preserving the same OLC
    // retry granularity as the general path below.
    if (height == 1) {
      static constexpr size_t kScanWindow = 8;
      const size_t root_capacity = ComputeBlockCapacity(root_size);
      size_t root_pos = InternalKVLayerType::LowerBoundInInternalBlock(
          lower, root_block, root_size);
      if (root_pos >= root_size) return 0;

      size_t collected = 0;
      bool first_leaf = true;
      while (root_pos < root_size && collected < len) {
        const FGDValueType val =
            root_block->values(root_capacity)[root_pos];
        FingerArray* const fa = val.finger_array_;
        if (fa == nullptr || fa->size == 0) {
          ++root_pos;
          continue;
        }

        const KeyType fgd_key = root_block->keys()[root_pos];
        const FGDType fgd({fgd_key, val});
        FingerRW* const fingers = fa->template fingers<FingerRW>();
        const size_t finger_count = fgd.GetSize();
        size_t finger_idx = first_leaf ? fgd.Decode(lower) : 0;

        while (finger_idx < finger_count && collected < len) {
          const size_t window_size =
              std::min(kScanWindow, finger_count - finger_idx);
          uint64_t snapshots[kScanWindow];

          for (size_t j = 0; j < window_size; ++j) {
            const uint64_t snapshot = fingers[finger_idx + j].GetData();
            snapshots[j] = snapshot;
            if (!FingerRW::IsWriteLocked(snapshot)) {
              LeafBlockType* const block =
                  FingerRW::UnpackBlock<LeafBlockType>(snapshot);
              if (block != nullptr) {
                const size_t capacity =
                    FingerRW::UnpackCapacity(snapshot);
                __builtin_prefetch(block->keys(), 0, 1);
                __builtin_prefetch(block->values(capacity), 0, 1);
              }
            }
          }

          for (size_t j = 0;
               j < window_size && collected < len; ++j) {
            FingerRW* const leaf_finger = &fingers[finger_idx + j];
            uint64_t before = snapshots[j];
            while (true) {
              if (FingerRW::IsWriteLocked(before)) {
                _mm_pause();
                before = leaf_finger->GetData();
                continue;
              }

              LeafBlockType* const leaf_block =
                  FingerRW::UnpackBlock<LeafBlockType>(before);
              const size_t leaf_block_size =
                  FingerRW::UnpackSize(before);
              const size_t leaf_cap =
                  FingerRW::UnpackCapacity(before);
              size_t leaf_pos = (first_leaf && leaf_block != nullptr)
                                    ? LeafKVLayerType::LowerBoundInLeafBlock(
                                          lower, leaf_block,
                                          leaf_block_size)
                                    : 0;
              const size_t checkpoint = collected;
              if (leaf_block != nullptr) {
                while (leaf_pos < leaf_block_size && collected < len) {
                  results[collected++] = {
                      leaf_block->keys()[leaf_pos],
                      leaf_block->values(leaf_cap)[leaf_pos]};
                  ++leaf_pos;
                }
              }

              std::atomic_signal_fence(std::memory_order_acq_rel);
              const uint64_t after = leaf_finger->GetData();
              if (after == before) break;
              collected = checkpoint;
              before = after;
              _mm_pause();
            }
            first_leaf = false;
          }
          finger_idx += window_size;
        }
        ++root_pos;
      }
      return collected;
    }

    static constexpr size_t kMaxDepth = 16;
    assert(height <= kMaxDepth);

    // Path state: one entry per FGD navigation layer.
    // path[0] = root block, path[1..height-1] = internal blocks.
    // Each entry's finger points to the block at the next layer (or leaf block
    // for the last entry).
    struct LayerState {
      const InternalBlockType* block;
      size_t block_size;
      size_t capacity;
      size_t block_pos;       // position in block → selects FGDValue
      FingerArray* fa;        // FingerArray from selected FGDValue
      size_t finger_idx;      // current finger in fa
      size_t finger_count;    // total fingers in fa
    };
    LayerState path[kMaxDepth];

    // Leaf block state.  leaf_finger remains valid for the duration of the
    // epoch even if a concurrent rebuild replaces the surrounding hierarchy.
    FingerRW* leaf_finger = nullptr;

    // Helper: load FGD state at path[lvl] from its current block_pos.
    // Sets fa, finger_idx=0, finger_count.
    auto load_fgd_first = [&](size_t lvl) {
      auto& s = path[lvl];
      FGDValueType val = s.block->values(s.capacity)[s.block_pos];
      s.fa = val.finger_array_;
      s.finger_idx = 0;
      s.finger_count = (s.fa != nullptr) ? s.fa->size : 0;
    };

    // Helper: descend from path[from_lvl] to leaf, taking first entries at each
    // level below. Loads internal blocks at path[from_lvl+1..height-1]
    // and the leaf block.
    auto descend_first = [&](size_t from_lvl) {
      for (size_t d = from_lvl + 1; d < height; ++d) {
        if (path[d - 1].fa == nullptr || path[d - 1].finger_count == 0) {
          leaf_finger = nullptr;
          return;
        }
        const FingerRW& f =
            path[d - 1].fa->template fingers<FingerRW>()[path[d - 1].finger_idx];
        uint64_t fd = f.GetData();
        auto& ns = path[d];
        ns.block = FingerRW::UnpackBlock<InternalBlockType>(fd);
        ns.block_size = FingerRW::UnpackSize(fd);
        if (ns.block == nullptr || ns.block_size == 0) {
          leaf_finger = nullptr;
          return;
        }
        ns.capacity = FingerRW::UnpackCapacity(fd);
        ns.block_pos = 0;
        load_fgd_first(d);
      }
      // Load leaf block from the last layer's finger
      size_t last = height - 1;
      if (path[last].fa == nullptr || path[last].finger_count == 0) {
        leaf_finger = nullptr;
        return;
      }
      FingerRW& f =
          path[last].fa->template fingers<FingerRW>()[path[last].finger_idx];
      leaf_finger = &f;
    };

    // === Phase 1: Navigate from root to leaf using LowerBound ===

    // Root layer (path[0])
    {
      auto& s = path[0];
      s.block = root_block;
      s.block_size = root_size;
      s.capacity = ComputeBlockCapacity(root_size);
      s.block_pos = InternalKVLayerType::LowerBoundInInternalBlock(
          lower, root_block, root_size);
      if (s.block_pos >= s.block_size) return 0;

      FGDValueType val = s.block->values(s.capacity)[s.block_pos];
      s.fa = val.finger_array_;
      if (s.fa == nullptr) return 0;
      KeyType fgd_key = s.block->keys()[s.block_pos];
      FGDType fgd({fgd_key, val});
      s.finger_idx = fgd.Decode(lower);
      s.finger_count = fgd.GetSize();
    }

    // Internal layers (path[1] to path[height-1])
    for (size_t i = 1; i < height; ++i) {
      const FingerRW& finger =
          path[i - 1].fa->template fingers<FingerRW>()[path[i - 1].finger_idx];
      uint64_t fdata = finger.GetData();

      auto& s = path[i];
      s.block = FingerRW::UnpackBlock<InternalBlockType>(fdata);
      s.block_size = FingerRW::UnpackSize(fdata);
      if (s.block == nullptr || s.block_size == 0) return 0;

      s.capacity = FingerRW::UnpackCapacity(fdata);
      s.block_pos = InternalKVLayerType::LowerBoundInInternalBlock(
          lower, s.block, s.block_size);
      if (s.block_pos >= s.block_size) s.block_pos = s.block_size - 1;

      FGDValueType val = s.block->values(s.capacity)[s.block_pos];
      s.fa = val.finger_array_;
      if (s.fa == nullptr) return 0;
      KeyType fgd_key = s.block->keys()[s.block_pos];
      FGDType fgd({fgd_key, val});
      s.finger_idx = fgd.Decode(lower);
      s.finger_count = fgd.GetSize();
    }

    // Load initial leaf block
    {
      size_t last = height - 1;
      FingerRW& finger =
          path[last].fa->template fingers<FingerRW>()[path[last].finger_idx];
      leaf_finger = &finger;
    }

    // === Phase 2: Collect results ===
    size_t collected = 0;
    bool first_leaf = true;

    while (collected < len) {
      // Optimistically copy one leaf bucket and accept the result only if the
      // finger's pointer, size, capacity, version, and state remain unchanged.
      // A conflicting writer therefore causes only this bucket to be retried;
      // scans neither acquire the writer bit nor block other readers.
      if (leaf_finger != nullptr) {
        while (true) {
          const uint64_t before = leaf_finger->GetData();
          if (FingerRW::IsWriteLocked(before)) {
            _mm_pause();
            continue;
          }

          LeafBlockType* leaf_block =
              FingerRW::UnpackBlock<LeafBlockType>(before);
          const size_t leaf_block_size = FingerRW::UnpackSize(before);
          const size_t leaf_cap = FingerRW::UnpackCapacity(before);
          size_t leaf_pos = (first_leaf && leaf_block != nullptr)
                                ? LeafKVLayerType::LowerBoundInLeafBlock(
                                      lower, leaf_block, leaf_block_size)
                                : 0;
          const size_t checkpoint = collected;
          if (leaf_block != nullptr) {
            while (leaf_pos < leaf_block_size && collected < len) {
              results[collected++] = {
                  leaf_block->keys()[leaf_pos],
                  leaf_block->values(leaf_cap)[leaf_pos]};
              leaf_pos++;
            }
          }

          std::atomic_signal_fence(std::memory_order_acq_rel);
          const uint64_t after = leaf_finger->GetData();
          if (after == before) break;
          collected = checkpoint;
          _mm_pause();
        }
      }
      first_leaf = false;
      if (collected >= len) break;

      // Current leaf block exhausted. Advance to next leaf block by walking
      // up the path until we find a level that can advance.
      bool advanced = false;
      for (int lvl = static_cast<int>(height) - 1; lvl >= 0; --lvl) {
        // Try next finger in current FGD
        path[lvl].finger_idx++;
        if (path[lvl].finger_idx < path[lvl].finger_count) {
          descend_first(static_cast<size_t>(lvl));
          advanced = true;
          break;
        }
        // Try next FGD entry in current block
        path[lvl].block_pos++;
        if (path[lvl].block_pos < path[lvl].block_size) {
          load_fgd_first(static_cast<size_t>(lvl));
          descend_first(static_cast<size_t>(lvl));
          advanced = true;
          break;
        }
        // Level exhausted, try level above
      }

      if (!advanced) break;
    }

    return collected;
  }

  // Insert a key-value pair into the index.
  // Returns true if inserted, false if key already exists.
  // Serializes writers through the selected finger and retries from the root
  // if the target subtree is rebuilding.
  bool Insert(const KeyType key, const ValueType value) {
    memory::OperationEpochGuard guard;

    while (true) {
      uint64_t root_data = LoadRootData();
      if (root_data == 0) {
        return false;
      }
      const size_t height = RootHandle::Height(root_data);
      if (RootHandle::IsRebuilding(root_data)) {
        continue;
      }
      const InternalBlockType* root_block = RootHandle::Block(root_data);
      size_t root_size = RootHandle::Size(root_data);
      if (root_size >= RootOverflowThreshold()) {
        RebuildRoot(root_data);
        continue;
      }

      // Tik-tok pattern: track current and parent descriptors
      std::pair<KeyType, FGDValueType> descriptors[2];
      int idx = 0;
      size_t root_pos = lower_bound::SearchInternal(
          key, root_block->keys(), root_size);
      assert(root_pos < root_size);
      size_t root_capacity = ComputeBlockCapacity(root_size);
      descriptors[idx] = {
          root_block->keys()[root_pos],
          root_block->values(root_capacity)[root_pos]};

      auto current = [&]() -> auto& { return descriptors[idx]; };
      auto parent = [&]() -> auto& { return descriptors[1 - idx]; };

      // The root first hop is explicit; ordinary descriptor traversal starts
      // immediately below it.
      bool retry = false;
      for (int level = static_cast<int>(height) - 1;
           level > 0; --level) {
        KeyType out_key;
        FGDValueType out_value;
        size_t out_finger_idx;
        LookupResult lookup_result = FindNextDescriptorForMutation(
            key, current(), out_key, out_value, out_finger_idx,
            internal_B_overflow_);
        if (lookup_result == LookupResult::kRebuilding) {
          retry = true;
          break;
        }
        if (lookup_result == LookupResult::kOverflow) {
          if (level + 1 == static_cast<int>(height)) {
            RebuildTopLevel(root_data, current(), level);
          } else {
            Rebuild(parent(), current(), level);
          }
          retry = true;
          break;
        }
        // Toggle index and store new descriptor
        idx = 1 - idx;
        descriptors[idx] = {out_key, out_value};

        if (level > 1) {
        }
      }
      if (retry) {
        continue;  // Retry from root
      }

      // After loop: current() = leaf descriptor, parent() = leaf's parent descriptor
      // Insert into leaf level (level 0)
      UpdateResult result =
          InsertIntoLeafFHB(key, value, current(), B_overflow_);

      if (result == UpdateResult::kRebuilding) {
        // Finger is rebuilding, retry from root
        continue;
      }
      if (result == UpdateResult::kOverflow) {
        // Trigger rebuild for this subtree
        // At height 1, the leaf descriptor lives directly in the root block.
        if (height == 1) {
          RebuildTopLevel(root_data, current(), 0);
        } else {
          Rebuild(parent(), current(), 0);
        }
        continue;  // Retry from root
      }
      return result == UpdateResult::kSuccess;
    }
  }

  // Atomically replace the value associated with an existing key.
  // Returns true when the key exists and the replacement is published, or
  // false when the key is absent. The leaf-finger CAS is the linearization
  // point; the old immutable block is reclaimed through the epoch manager.
  bool Update(const KeyType key, const ValueType value) {
    memory::OperationEpochGuard guard;

    while (true) {
      uint64_t root_data = LoadRootData();
      if (root_data == 0) {
        return false;
      }
      const InternalBlockType* root_block = RootHandle::Block(root_data);
      size_t root_size = RootHandle::Size(root_data);
      size_t root_pos = lower_bound::SearchInternal(
          key, root_block->keys(), root_size);
      assert(root_pos < root_size);
      size_t root_capacity = ComputeBlockCapacity(root_size);
      std::pair<KeyType, FGDValueType> descriptor = {
          root_block->keys()[root_pos],
          root_block->values(root_capacity)[root_pos]};

      // UPDATE does not change keys or block sizes, so it never needs to
      // inspect overflow thresholds or trigger an internal rebuild. Descend
      // exactly like a reader and perform the only mutation at the leaf.
      for (int level = static_cast<int>(RootHandle::LevelsBelowRoot(root_data));
           level > 0; --level) {
        KeyType out_key;
        FGDValueType out_value;
        FindNextDescriptor(key, descriptor, out_key, out_value);
        descriptor = {out_key, out_value};
      }

      UpdateResult result = UpdateLeafFHB(key, value, descriptor);
      if (result == UpdateResult::kRebuilding) {
        continue;
      }
      return result == UpdateResult::kSuccess;
    }
  }

  // Delete a key from the index.
  // Returns true if deleted, false if key not found.
  // Serializes writers through the selected finger and retries from the root
  // if the target subtree is rebuilding.
  bool Delete(const KeyType key) {
    memory::OperationEpochGuard guard;

    while (true) {
      uint64_t root_data = LoadRootData();
      if (root_data == 0) {
        return false;
      }
      const size_t height = RootHandle::Height(root_data);
      if (RootHandle::IsRebuilding(root_data)) {
        continue;
      }
      const InternalBlockType* root_block = RootHandle::Block(root_data);
      size_t root_size = RootHandle::Size(root_data);
      if (root_size >= RootOverflowThreshold()) {
        RebuildRoot(root_data);
        continue;
      }

      // Tik-tok pattern: track current and parent descriptors
      std::pair<KeyType, FGDValueType> descriptors[2];
      int idx = 0;
      size_t root_pos = lower_bound::SearchInternal(
          key, root_block->keys(), root_size);
      assert(root_pos < root_size);
      size_t root_capacity = ComputeBlockCapacity(root_size);
      descriptors[idx] = {
          root_block->keys()[root_pos],
          root_block->values(root_capacity)[root_pos]};

      auto current = [&]() -> auto& { return descriptors[idx]; };
      auto parent = [&]() -> auto& { return descriptors[1 - idx]; };

      // Traverse ordinary descriptors below the explicit root first hop.
      bool retry = false;
      for (int level = static_cast<int>(height) - 1;
           level > 0; --level) {
        KeyType out_key;
        FGDValueType out_value;
        size_t out_finger_idx;
        LookupResult lookup_result = FindNextDescriptorForMutation(
            key, current(), out_key, out_value, out_finger_idx,
            internal_B_overflow_);
        if (lookup_result == LookupResult::kRebuilding) {
          retry = true;
          break;
        }
        if (lookup_result == LookupResult::kOverflow) {
          if (level + 1 == static_cast<int>(height)) {
            RebuildTopLevel(root_data, current(), level);
          } else {
            Rebuild(parent(), current(), level);
          }
          retry = true;
          break;
        }
        // Toggle index and store new descriptor
        idx = 1 - idx;
        descriptors[idx] = {out_key, out_value};
      }
      if (retry) {
        continue;  // Retry from root
      }

      // After loop: current() = leaf descriptor, parent() = leaf's parent descriptor
      // Delete from leaf level (level 0)
      UpdateResult result = DeleteFromLeafFHB(key, current());

      if (result == UpdateResult::kRebuilding) {
        // Finger is rebuilding, retry from root
        continue;
      }
      return result == UpdateResult::kSuccess;
    }
  }

  void Print() const {
    memory::OperationEpochGuard guard;
    uint64_t root_data = LoadRootData();

    std::cout << "=== FHIndexRW ===" << std::endl;
    if (root_data == 0) {
      std::cout << "Empty" << std::endl;
      std::cout << "========================" << std::endl;
      return;
    }

    const size_t height = RootHandle::Height(root_data);
    std::cout << "Level count: " << height << std::endl;
    std::cout << "B: " << B_ << ", B_overflow: " << B_overflow_
              << ", finger density: " << finger_density_.numerator() << "/"
              << finger_density_.denominator() << std::endl;
    std::cout << "internal_B: " << internal_B_ << ", internal_B_overflow: " << internal_B_overflow_
              << ", internal finger density: "
              << internal_finger_density_.numerator() << "/"
              << internal_finger_density_.denominator() << std::endl;
    std::cout << std::endl;

    // Print root level info
    std::cout << "Root level: " << std::endl;
    const InternalBlockType* root_block = RootHandle::Block(root_data);
    size_t root_size = RootHandle::Size(root_data);
    std::cout << "size: " << root_size << std::endl;
    if (root_size < 10) {
      const KeyType* keys = root_block->keys();
      for (size_t i = 0; i < root_size; i++) {
        std::cout << keys[i] << " ";
      }
      std::cout << std::endl;
    }

    // Print finger layers (level height - 1 down to level 1)
    for (int level = static_cast<int>(height) - 1;
         level >= 1; --level) {
      std::cout << "Finger level " << level << ": " << std::endl;
      auto [finger_count, finger_arrays] =
          GetFingerLayerMetadataFromRoot(level, root_data);
      PrintFingerLayer(
          finger_count, internal_B_, internal_finger_density_);
    }

    std::cout << "Leaf level: " << std::endl;
    auto [leaf_finger_count, leaf_finger_arrays] =
        GetFingerLayerMetadataFromRoot(0, root_data);
    PrintFingerLayer(leaf_finger_count, B_, finger_density_);

    std::cout << "========================" << std::endl;
  }

  // Getters for testing
  size_t GetLevelCount() const {
    return GetRootMetadata().height;
  }

  RootMetadata GetRootMetadata() const {
    memory::OperationEpochGuard guard;
    uint64_t root_data = LoadRootData();
    if (root_data == 0) {
      return {0, 0};
    }
    return {RootHandle::Size(root_data), RootHandle::Height(root_data)};
  }

  std::vector<FingerArray*> GetLeafFingerArrays() const {
    auto [finger_count, finger_arrays] = GetFingerLayerMetadata(0);
    return finger_arrays;
  }

  std::vector<FingerArray*> GetInternalFingerArrays(size_t level) const {
    auto [finger_count, finger_arrays] = GetFingerLayerMetadata(level + 1);
    return finger_arrays;
  }

  template <bool ClaimForRebuild>
  static std::vector<FingerRW> CollectNextLayerFromBlock(
      const InternalBlockType* block, size_t block_size, size_t capacity) {
    std::vector<FingerRW> result;
    if (block == nullptr) return result;

    const FGDValueType* values = block->values(capacity);
    for (size_t i = 0; i < block_size; ++i) {
      FingerArray* next_array = values[i].finger_array_;
      if (next_array == nullptr) continue;

      FingerRW* next_fingers = next_array->fingers<FingerRW>();
      for (size_t j = 0; j < next_array->size; ++j) {
        FingerRW snapshot = ClaimForRebuild
                                ? ClaimFingerForRebuild(&next_fingers[j])
                                : next_fingers[j];
        if ((snapshot.GetData() & FingerRW::kPtrMask) != 0) {
          result.push_back(snapshot);
        }
      }
    }
    return result;
  }

  // Given an array of fingers, for each finger:
  // - Get the block it points to
  // - For each FGDValueType in the block, get its finger_array_
  // - Mark all fingers in that finger_array_ as rebuilding
  // Returns all newly marked FingerRW (copies).
  // After AugmentLevel, all internal fingers have valid blocks (no shadows).
  std::vector<FingerRW> MarkNextLayer(const std::vector<FingerRW>& fingers) {

    std::vector<FingerRW> result;

    using InternalBlockType = Block<KeyType, FGDValueType>;

    for (const FingerRW& finger : fingers) {
      uint64_t data = finger.GetData();
      uint64_t block_ptr = data & FingerRW::kPtrMask;

      // Skip null pointer (should not happen after AugmentLevel)
      if (block_ptr == 0) {
        continue;
      }

      InternalBlockType* block = FingerRW::UnpackBlock<InternalBlockType>(data);
      uint16_t block_size = FingerRW::UnpackSize(data);

      const size_t capacity = FingerRW::UnpackCapacity(data);
      const FGDValueType* values = block->values(capacity);

      // For each FGD in the block, mark all fingers in its finger_array_
      for (size_t i = 0; i < block_size; i++) {
        FingerArray* next_array = values[i].finger_array_;
        if (next_array == nullptr) {
          continue;
        }

        size_t finger_count = next_array->size;
        FingerRW* next_fingers = next_array->fingers<FingerRW>();

        for (size_t j = 0; j < finger_count; j++) {
          FingerRW claimed = ClaimFingerForRebuild(&next_fingers[j]);
          uint64_t next_data = claimed.GetData();
          uint64_t next_block_ptr = next_data & FingerRW::kPtrMask;

          // Add to result if not null
          if (next_block_ptr != 0) {
            result.push_back(claimed);
          }
        }
      }
    }

    return result;
  }

  // Collect all key-value pairs from leaf level fingers.
  // Returns {size, keys, values} using allocate_array.
  // Caller is responsible for freeing the arrays with free_array.
  std::tuple<size_t, KeyType*, ValueType*>
  CollectKeyValuePairs(const std::vector<FingerRW>& fingers) {

    // First pass: count total size
    size_t total_size = 0;
    for (const FingerRW& finger : fingers) {
      uint64_t data = finger.GetData();
      uint16_t block_size = FingerRW::UnpackSize(data);
      LeafBlockType* block = FingerRW::UnpackBlock<LeafBlockType>(data);

      if (block != nullptr && block_size > 0) {
        total_size += block_size;
      }
    }

    if (total_size == 0) {
      return {0, nullptr, nullptr};
    }

    // Allocate arrays
    KeyType* keys = new KeyType[total_size];
    ValueType* values = new ValueType[total_size];

    // Second pass: fill arrays
    size_t idx = 0;
    for (const FingerRW& finger : fingers) {
      uint64_t data = finger.GetData();
      LeafBlockType* block = FingerRW::UnpackBlock<LeafBlockType>(data);
      uint16_t block_size = FingerRW::UnpackSize(data);

      if (block == nullptr || block_size == 0) {
        continue;
      }

      size_t capacity = FingerRW::UnpackCapacity(data);
      const KeyType* block_keys = block->keys();
      const ValueType* block_values = block->values(capacity);

      for (size_t j = 0; j < block_size; j++) {
        keys[idx] = block_keys[j];
        values[idx] = block_values[j];
        idx++;
      }
    }

    return {total_size, keys, values};
  }

  // Descend one layer without marking rebuilding bit.
  // Given an array of fingers, for each finger:
  // - Get the block it points to
  // - For each FGDValueType in the block, get its finger_array_
  // - Collect all fingers from that finger_array_
  // Returns all collected FingerRW (copies).
  // After AugmentLevel, all internal fingers have valid blocks (no shadows).
  std::vector<FingerRW> DescendOneLayer(const std::vector<FingerRW>& fingers) const {

    std::vector<FingerRW> result;

    using InternalBlockType = Block<KeyType, FGDValueType>;

    for (const FingerRW& finger : fingers) {
      uint64_t data = finger.GetData();
      uint64_t block_ptr = data & FingerRW::kPtrMask;

      // Skip null pointer (should not happen after AugmentLevel)
      if (block_ptr == 0) {
        continue;
      }

      InternalBlockType* block = FingerRW::UnpackBlock<InternalBlockType>(data);
      uint16_t block_size = FingerRW::UnpackSize(data);

      const size_t capacity = FingerRW::UnpackCapacity(data);
      const FGDValueType* values = block->values(capacity);

      // For each FGD in the block, collect all fingers from its finger_array_
      for (size_t i = 0; i < block_size; i++) {
        FingerArray* next_array = values[i].finger_array_;
        if (next_array == nullptr) {
          continue;
        }

        size_t finger_count = next_array->size;
        FingerRW* next_fingers = next_array->fingers<FingerRW>();

        for (size_t j = 0; j < finger_count; j++) {
          uint64_t next_data = next_fingers[j].GetData();
          uint64_t next_block_ptr = next_data & FingerRW::kPtrMask;

          // Add to result if not null
          if (next_block_ptr != 0) {
            result.push_back(next_fingers[j]);
          }
        }
      }
    }

    return result;
  }

  // Get KV layer metadata by traversing from virtual level down to target layer.
  // layer_id: 0 = leaf level, 1+ = internal levels
  // Returns: {total_size, num_blocks, blocks}
  // - total_size: sum of all block sizes
  // - num_blocks: number of unique blocks
  // - blocks: vector of block pointers (as void* since block type varies by layer)
  std::tuple<size_t, size_t, std::vector<void*>> GetKVLayerMetadata(size_t layer_id) const {
    memory::OperationEpochGuard guard;
    uint64_t root_data = LoadRootData();
    return GetKVLayerMetadataFromRoot(layer_id, root_data);
  }

  std::tuple<size_t, size_t, std::vector<void*>>
  GetKVLayerMetadataFromRoot(size_t layer_id, uint64_t root_data) const {
    if (root_data == 0 || layer_id >= RootHandle::Height(root_data)) {
      return {0, 0, {}};
    }

    std::vector<FingerRW> current_fingers =
        CollectNextLayerFromBlock<false>(
            RootHandle::Block(root_data), RootHandle::Size(root_data),
            ComputeBlockCapacity(RootHandle::Size(root_data)));

    // The first step above descends from the root block to the highest finger
    // layer. Continue until the target layer is reached.
    for (size_t level = RootHandle::Height(root_data) - 1;
         level > layer_id;
         --level) {
      current_fingers = DescendOneLayer(current_fingers);
    }

    // Collect metadata from current_fingers at target layer
    size_t total_size = 0;
    size_t num_blocks = 0;
    std::vector<void*> blocks;

    for (const FingerRW& finger : current_fingers) {
      uint64_t data = finger.GetData();
      uint64_t block_ptr = data & FingerRW::kPtrMask;

      // Skip null pointer (should not happen after AugmentLevel)
      if (block_ptr == 0) {
        continue;
      }

      uint16_t block_size = FingerRW::UnpackSize(data);
      total_size += block_size;
      num_blocks++;
      blocks.push_back(FingerRW::UnpackBlock<void>(data));
    }

    return {total_size, num_blocks, blocks};
  }

  // Get finger layer metadata by traversing from virtual level.
  // layer_id: 0 = leaf level, 1+ = internal levels
  // Returns: {finger_count, finger_arrays}
  // - finger_count: total number of fingers across all finger arrays
  // - finger_arrays: vector of FingerArray pointers
  std::tuple<size_t, std::vector<FingerArray*>> GetFingerLayerMetadata(size_t layer_id) const {
    memory::OperationEpochGuard guard;
    uint64_t root_data = LoadRootData();
    return GetFingerLayerMetadataFromRoot(layer_id, root_data);
  }

  std::tuple<size_t, std::vector<FingerArray*>>
  GetFingerLayerMetadataFromRoot(size_t layer_id, uint64_t root_data) const {
    if (root_data == 0 || layer_id >= RootHandle::Height(root_data)) {
      return {0, {}};
    }

    // Collect finger_arrays from FGDValueType in blocks at layer_id + 1
    std::vector<FingerArray*> finger_arrays;
    size_t total_finger_count = 0;

    const size_t height = RootHandle::Height(root_data);
    if (height == layer_id + 1) {
      const InternalBlockType* root_block = RootHandle::Block(root_data);
      const size_t root_size = RootHandle::Size(root_data);
      const FGDValueType* values =
          root_block->values(ComputeBlockCapacity(root_size));
      for (size_t i = 0; i < root_size; ++i) {
        FingerArray* fa = values[i].finger_array_;
        if (fa != nullptr) {
          finger_arrays.push_back(fa);
          total_finger_count += fa->size;
        }
      }
      return {total_finger_count, finger_arrays};
    }

    std::vector<FingerRW> current_fingers =
        CollectNextLayerFromBlock<false>(
            RootHandle::Block(root_data), RootHandle::Size(root_data),
            ComputeBlockCapacity(RootHandle::Size(root_data)));

    // Descend until current fingers point to blocks at layer_id + 1.
    for (size_t level = height - 1; level > layer_id + 1; --level) {
      current_fingers = DescendOneLayer(current_fingers);
    }

    for (const FingerRW& finger : current_fingers) {
      uint64_t data = finger.GetData();
      uint64_t block_ptr = data & FingerRW::kPtrMask;

      // Skip null pointer (should not happen after AugmentLevel)
      if (block_ptr == 0) {
        continue;
      }

      InternalBlockType* block = FingerRW::UnpackBlock<InternalBlockType>(data);
      uint16_t block_size = FingerRW::UnpackSize(data);

      const size_t capacity = FingerRW::UnpackCapacity(data);
      const FGDValueType* values = block->values(capacity);

      for (size_t i = 0; i < block_size; i++) {
        FingerArray* fa = values[i].finger_array_;
        if (fa != nullptr) {
          finger_arrays.push_back(fa);
          total_finger_count += fa->size;
        }
      }
    }

    return {total_finger_count, finger_arrays};
  }

  // Mark all fingers in the subtree as rebuilding.
  // layer_id: 0 = leaf level, 1+ = internal levels
  // Returns vector of leaf level fingers (copies).
  // After AugmentLevel, all internal fingers have valid blocks (no shadows).
  std::vector<FingerRW> MarkSubtreeRebuilding(FingerArray* finger_array, size_t layer_id) {
    if (finger_array == nullptr) {
      return {};
    }

    // First, mark all fingers in the current finger_array
    std::vector<FingerRW> current_fingers;
    size_t finger_count = finger_array->size;
    FingerRW* fingers = finger_array->fingers<FingerRW>();

    for (size_t i = 0; i < finger_count; i++) {
      FingerRW claimed = ClaimFingerForRebuild(&fingers[i]);
      uint64_t data = claimed.GetData();
      uint64_t block_ptr = data & FingerRW::kPtrMask;

      // Add if not null (should always be true after AugmentLevel)
      if (block_ptr != 0) {
        current_fingers.push_back(claimed);
      }
    }

    // Loop to descend layers until reaching leaf level
    for (size_t level = layer_id; level > 0; level--) {
      current_fingers = MarkNextLayer(current_fingers);
    }

    return current_fingers;
  }

  // Rebuild a subtree rooted at the given FGD (internal helper).
  // Marks all fingers in the subtree as rebuilding and collects all key-value pairs.
  // layer_id: the layer where the FGD resides (0 = leaf, 1+ = internal)
  // max_key: the FGD's key, used as max_key for the last FGD in rebuilt subtree
  // Returns {layer_count, fgd_size, fgd_keys, fgd_values} from Construct.
  std::tuple<size_t, size_t, KeyType*, FGDValueType*>
  RebuildFGDSubtree(const FGDValueType& fgd_value, size_t layer_id, KeyType max_key) {
    // Mark all fingers in subtree as rebuilding and get leaf fingers
    std::vector<FingerRW> leaf_fingers = MarkSubtreeRebuilding(fgd_value.finger_array_, layer_id);

    // Collect all key-value pairs from leaf level
    auto [size, keys, values] = CollectKeyValuePairs(leaf_fingers);

    // Construct with height = layer_id + 1, passing max_key for correct FGD key
    auto result = Construct(size, keys, values, layer_id + 1,
                            B_, finger_density_, internal_B_,
                            internal_finger_density_, top_level_threshold_,
                            max_key);

    assert(std::get<0>(result) == layer_id + 1 &&
           "Rebuild should produce correct number of layers");

    // Free collected arrays
    delete[] keys;
    delete[] values;

    return result;
  }

  // Rebuild a subtree and atomically update the parent's block.
  // After AugmentLevel, rebuilt keys stay within current finger's range,
  // so only current finger needs to be updated.
  // Steps:
  // 1. Decode parent to find current finger
  // 2. CAS finger to set rebuilding bit
  // 3. RebuildFGDSubtree to reconstruct the subtree
  // 4. Create new block with rebuilt keys replacing old entry
  // 5. CAS finger to update block pointer
  RebuildResult Rebuild(
      const std::pair<KeyType, FGDValueType>& parent_descriptor,
      const std::pair<KeyType, FGDValueType>& current_descriptor,
      size_t current_layer_id) {
    using KVLayerType = KVLayerRW<KeyType, FGDValueType>;

    // Step 1: Decode parent to find current finger
    FGDType parent_fgd(parent_descriptor);
    size_t finger_idx = parent_fgd.Decode(current_descriptor.first);

    FingerRW* curr_finger = const_cast<FingerRW*>(parent_fgd.GetFingerAddr(finger_idx));
    uint64_t curr_finger_data = curr_finger->GetData();

    // Step 2: CAS to set rebuilding bit
    if (FingerRW::IsRebuilding(curr_finger_data)) {
      return RebuildResult::kRebuilding;
    }
    while (FingerRW::IsWriteLocked(curr_finger_data)) {
      _mm_pause();
      curr_finger_data = curr_finger->GetData();
      if (FingerRW::IsRebuilding(curr_finger_data)) {
        return RebuildResult::kRebuilding;
      }
    }
    uint64_t curr_expected = curr_finger_data;
    uint64_t curr_desired = FingerRW::SetRebuildingBit(curr_finger_data);
    if (!curr_finger->CompareExchange(curr_expected, curr_desired)) {
      return RebuildResult::kRebuilding;
    }

    std::chrono::steady_clock::time_point retrain_start;
    if constexpr (TrackRetrain) {
      retrain_start = std::chrono::steady_clock::now();
    }

    // Step 3: RebuildFGDSubtree
    auto [layer_count, fgd_size, fgd_keys, fgd_values] =
        RebuildFGDSubtree(current_descriptor.second, current_layer_id, current_descriptor.first);

    // Step 4: Create new block replacing current_descriptor.first with rebuilt keys
    InternalBlockType* old_curr_block = FingerRW::UnpackBlock<InternalBlockType>(curr_finger_data);
    uint16_t old_curr_size = FingerRW::UnpackSize(curr_finger_data);
    const size_t old_curr_capacity =
        FingerRW::UnpackCapacity(curr_finger_data);

    auto [new_curr_block, new_curr_size, new_curr_capacity] =
        KVLayerType::CreateBlockWithReplace(
            old_curr_block, old_curr_size, old_curr_capacity,
            current_descriptor.first,
            fgd_keys, fgd_values, fgd_size);

    // Free the arrays from RebuildFGDSubtree
    delete[] fgd_keys;
    delete[] fgd_values;

    // Step 5: CAS finger to update block pointer
    uint64_t new_curr_data = FingerRW::Pack(new_curr_block, static_cast<uint16_t>(new_curr_size));
    [[maybe_unused]] bool curr_cas = curr_finger->CompareExchange(curr_desired, new_curr_data);
    assert(curr_cas && "CAS must succeed since we hold the rebuilding bit");

    // Retire the replaced entry's subtree (FingerArray + all sub-blocks recursively)
    RetireSubtree(current_descriptor.second, current_layer_id);
    // Retire old block (contains the entry that was replaced + entries copied to new block)
    InternalBlockType::Retire(old_curr_block, old_curr_capacity);

    if constexpr (TrackRetrain) {
      auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - retrain_start)
                             .count();
      retrain_stats_.Record(
          RebuildKind::kSubtree, static_cast<uint64_t>(duration_ns));
    }
    return RebuildResult::kSuccess;
  }

  // Rebuild a descriptor stored directly in the root block. The rebuilt
  // subtree is local, while the containing root block is copied so that one
  // root-word CAS publishes the complete replacement generation.
  RebuildResult RebuildTopLevel(
      uint64_t old_root_data,
      const std::pair<KeyType, FGDValueType>& current_descriptor,
      size_t current_layer_id) {
    using KVLayerType = KVLayerRW<KeyType, FGDValueType>;

    if (old_root_data == 0 || LoadRootData() != old_root_data) {
      return RebuildResult::kRebuilding;
    }
    assert(current_layer_id + 1 == RootHandle::Height(old_root_data));

    if (RootHandle::IsRebuilding(old_root_data)) {
      return RebuildResult::kRebuilding;
    }

    uint64_t claimed_root_data = RootHandle::SetRebuilding(old_root_data);
    uint64_t expected_root_data = old_root_data;
    if (!root_data_.compare_exchange_strong(
            expected_root_data, claimed_root_data,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      return RebuildResult::kRebuilding;
    }

    std::chrono::steady_clock::time_point retrain_start;
    if constexpr (TrackRetrain) {
      retrain_start = std::chrono::steady_clock::now();
    }

    InternalBlockType* old_root_block = RootHandle::Block(old_root_data);
    size_t old_root_size = RootHandle::Size(old_root_data);

    auto [layer_count, fgd_size, fgd_keys, fgd_values] =
        RebuildFGDSubtree(
            current_descriptor.second, current_layer_id,
            current_descriptor.first);
    assert(layer_count == current_layer_id + 1);
    (void)layer_count;

    size_t old_root_capacity =
        ComputeBlockCapacity(old_root_size);
    auto [new_root_block, new_root_size, new_root_capacity] =
        KVLayerType::CreateBlockWithReplace(
            old_root_block,
            old_root_size,
            old_root_capacity,
            current_descriptor.first,
            fgd_keys,
            fgd_values,
            fgd_size);
    (void)new_root_capacity;
    assert(new_root_size <= RootHandle::kSizeMask);

    uint64_t new_root_data = CreateRootData(
        new_root_block, new_root_size, RootHandle::Height(old_root_data));

    expected_root_data = claimed_root_data;
    if (!root_data_.compare_exchange_strong(
            expected_root_data, new_root_data,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      // The replacement subtree belongs only to the unpublished root. Other
      // descriptors in the new root are shared with the old root and must
      // not be retired here.
      for (size_t i = 0; i < fgd_size; i++) {
        RetireSubtree(fgd_values[i], current_layer_id);
      }
      delete[] fgd_keys;
      delete[] fgd_values;
      RetireRootBlock(new_root_data);
      return RebuildResult::kCASFailed;
    }

    delete[] fgd_keys;
    delete[] fgd_values;

    // Unchanged root descriptors share their existing subtrees with the new
    // snapshot. Retire only the replaced subtree and the old root shell.
    RetireSubtree(current_descriptor.second, current_layer_id);
    RetireRootBlock(old_root_data);

    if constexpr (TrackRetrain) {
      auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - retrain_start)
                             .count();
      retrain_stats_.Record(
          RebuildKind::kTopLevelSubtree,
          static_cast<uint64_t>(duration_ns));
    }
    return RebuildResult::kSuccess;
  }

  // Rebuild the entire tree when the root block overflows. The public entry
  // point pins one root snapshot; update paths that already hold an epoch guard
  // call the overload below directly to avoid nesting epoch guards.
  RebuildResult RebuildRoot() {
    memory::OperationEpochGuard guard;
    return RebuildRoot(LoadRootData());
  }

  RebuildResult RebuildRoot(uint64_t old_root_data) {
    if (old_root_data == 0) {
      return RebuildResult::kSuccess;
    }

    // Do not claim an already superseded tree.
    if (LoadRootData() != old_root_data) {
      return RebuildResult::kRebuilding;
    }

    if (RootHandle::IsRebuilding(old_root_data)) {
      return RebuildResult::kRebuilding;  // Someone else is already rebuilding
    }

    uint64_t claimed_root_data = RootHandle::SetRebuilding(old_root_data);
    uint64_t expected_root_data = old_root_data;
    if (!root_data_.compare_exchange_strong(
            expected_root_data, claimed_root_data,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      return RebuildResult::kRebuilding;  // CAS failed, someone else got there first
    }

    std::chrono::steady_clock::time_point retrain_start;
    if constexpr (TrackRetrain) {
      retrain_start = std::chrono::steady_clock::now();
    }

    // Mark the old subtree and collect its leaf contents. Readers pinned to the
    // old immutable snapshot may continue; writers will retry from root.
    std::vector<FingerRW> current_fingers =
        CollectNextLayerFromBlock<true>(
            RootHandle::Block(old_root_data),
            RootHandle::Size(old_root_data),
            ComputeBlockCapacity(RootHandle::Size(old_root_data)));

    for (size_t level = RootHandle::Height(old_root_data) - 1;
         level > 0; --level) {
      current_fingers = MarkNextLayer(current_fingers);
    }

    // current_fingers now contains leaf level fingers
    auto [size, keys, values] = CollectKeyValuePairs(current_fingers);

    // Build a complete replacement without modifying the published snapshot.
    auto [new_layer_count, fgd_size, fgd_keys, fgd_values] = Construct(
        size, keys, values, 0, B_, finger_density_, internal_B_,
        internal_finger_density_, top_level_threshold_);

    // Free collected arrays
    delete[] keys;
    delete[] values;

    // Build new root block
    auto [new_block, new_capacity] = InternalBlockType::CreateWithCapacity(fgd_size);
    KeyType* block_keys = new_block->keys();
    FGDValueType* block_values = new_block->values(new_capacity);
    for (size_t i = 0; i < fgd_size; i++) {
      block_keys[i] = fgd_keys[i];
      block_values[i] = fgd_values[i];
    }

    // Free final fgd arrays
    delete[] fgd_keys;
    delete[] fgd_values;

    uint64_t new_root_data =
        CreateRootData(new_block, fgd_size, new_layer_count);

    // The one root publication and the root rebuild's linearization point.
    expected_root_data = claimed_root_data;
    if (!root_data_.compare_exchange_strong(
            expected_root_data, new_root_data,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      RetireWholeRoot(new_root_data);
      return RebuildResult::kCASFailed;
    }

    RetireWholeRoot(old_root_data);

    if constexpr (TrackRetrain) {
      auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - retrain_start)
                             .count();
      retrain_stats_.Record(
          RebuildKind::kRoot, static_cast<uint64_t>(duration_ns));
    }
    return RebuildResult::kSuccess;
  }

 private:
  // Establish exclusive ownership of a descendant finger. An already-set bit
  // belongs to an update or a narrower rebuild that was in flight before its
  // ancestor was frozen; wait for that owner to publish, then claim the newly
  // published value. This prevents overlapping rebuilds from retiring the same
  // subtree twice.
  static FingerRW ClaimFingerForRebuild(FingerRW* finger) {
    uint64_t observed = finger->GetData();
    while (true) {
      if (FingerRW::IsRebuilding(observed)) {
        std::this_thread::yield();
        observed = finger->GetData();
        continue;
      }
      if (FingerRW::IsWriteLocked(observed)) {
        _mm_pause();
        observed = finger->GetData();
        continue;
      }

      uint64_t desired = FingerRW::SetRebuildingBit(observed);
      if (finger->CompareExchange(observed, desired)) {
        FingerRW claimed;
        claimed.SetData(desired);
        return claimed;
      }
    }
  }

  size_t RootOverflowThreshold() const {
    if (top_level_threshold_ == std::numeric_limits<size_t>::max()) {
      return top_level_threshold_;
    }
    return top_level_threshold_ + 1;
  }

  uint64_t LoadRootData() const {
    return root_data_.load(std::memory_order_acquire);
  }

  static size_t RootBlockBytes(size_t root_size) {
    return InternalBlockType::AllocationBytes(root_size);
  }

  // Consume an ordinary temporary root block and copy it into explicitly
  // aligned storage whose low address bits can carry the root height.
  static uint64_t CreateRootData(InternalBlockType* root_block,
                                 size_t root_size,
                                 size_t height) {
    assert(root_block != nullptr);
    assert(root_size > 0 && root_size <= RootHandle::kSizeMask);
    void* storage = memory::AllocateAligned(
        RootBlockBytes(root_size), RootHandle::kAlignment);
    assert(storage != nullptr);
    InternalBlockType* aligned_block = InternalBlockType::ConstructAt(storage);

    const size_t capacity = ComputeBlockCapacity(root_size);
    std::copy_n(root_block->keys(), root_size, aligned_block->keys());
    std::copy_n(root_block->values(capacity), root_size,
                aligned_block->values(capacity));
    InternalBlockType::Destroy(root_block, capacity);
    return RootHandle::Pack(aligned_block, root_size, height);
  }

  // Used only by exclusive BulkLoad and destruction. Child FingerArrays have
  // already been destroyed by DestroyAllFingerArrays.
  static void DestroyRootBlock(uint64_t root_data) {
    if (root_data == 0) return;

    InternalBlockType* root_block = RootHandle::Block(root_data);
    const size_t allocation_size = RootBlockBytes(RootHandle::Size(root_data));
    InternalBlockType::DestroyAt(root_block);
    memory::FreeAligned(root_block, allocation_size);
  }

  // Retire a FingerArray via epoch-based reclamation.
  static void RetireFingerArray(FingerArray* fa) {
    if (fa == nullptr) return;
    size_t total_bytes = sizeof(FingerArray) + fa->size * sizeof(FingerRW);
    memory::Retire(static_cast<void*>(fa), total_bytes);
  }

  // Retire an entire subtree rooted at a FGDValue.
  // layer_id: the layer of blocks pointed to by this FGDValue's fingers.
  // layer_id == 0: fingers point to leaf blocks (Block<KeyType, ValueType>)
  // layer_id > 0: fingers point to internal blocks (Block<KeyType, FGDValueType>)
  void RetireSubtree(const FGDValueType& fgd_val, size_t layer_id) {
    FingerArray* fa = fgd_val.finger_array_;
    if (fa == nullptr) return;

    size_t finger_count = fa->size;
    const FingerRW* fingers = fa->fingers<FingerRW>();

    for (size_t i = 0; i < finger_count; i++) {
      uint64_t data = fingers[i].GetData();

      if (layer_id == 0) {
        // Leaf block
        LeafBlockType* block = FingerRW::UnpackBlock<LeafBlockType>(data);
        if (block != nullptr) {
          size_t capacity = FingerRW::UnpackCapacity(data);
          LeafBlockType::Retire(block, capacity);
        }
      } else {
        // Internal block
        InternalBlockType* block = FingerRW::UnpackBlock<InternalBlockType>(data);
        if (block != nullptr) {
          uint16_t block_size = FingerRW::UnpackSize(data);
          const size_t capacity = FingerRW::UnpackCapacity(data);

          // Recursively retire children first (read values before retiring block)
          const FGDValueType* values = block->values(capacity);
          for (size_t j = 0; j < block_size; j++) {
            RetireSubtree(values[j], layer_id - 1);
          }

          InternalBlockType::Retire(block, capacity);
        }
      }
    }

    RetireFingerArray(fa);
  }

  // Retire only the immutable root block. Descendant subtrees may be shared
  // with a newer root generation and are therefore the caller's concern.
  void RetireRootBlock(uint64_t root_data) {
    if (root_data == 0) return;

    InternalBlockType* root_block = RootHandle::Block(root_data);
    memory::Retire(
        static_cast<void*>(root_block),
        RootBlockBytes(RootHandle::Size(root_data)));
  }

  // Retire one complete immutable root generation after root_data_ no longer
  // publishes it. The caller must be inside an epoch guard.
  void RetireWholeRoot(uint64_t root_data) {
    if (root_data == 0) return;

    const InternalBlockType* root_block = RootHandle::Block(root_data);
    size_t root_size = RootHandle::Size(root_data);
    size_t root_capacity = ComputeBlockCapacity(root_size);
    const FGDValueType* root_values = root_block->values(root_capacity);
    const size_t child_layer = RootHandle::LevelsBelowRoot(root_data);
    for (size_t i = 0; i < root_size; i++) {
      RetireSubtree(root_values[i], child_layer);
    }

    RetireRootBlock(root_data);
  }

  size_t top_level_threshold_;
  size_t B_;
  size_t B_overflow_;
  fh_index::internal::Fraction finger_density_;
  size_t internal_B_;
  size_t internal_B_overflow_;
  fh_index::internal::Fraction internal_finger_density_;

  RetrainStats retrain_stats_;
  std::atomic<uint64_t> root_data_;
};

}  // namespace fh_index_rw
