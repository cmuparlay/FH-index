#ifndef FH_INDEX_INTERNAL_CONSTRUCTION_MULTI_LEVEL_SEGMENTOR_H
#define FH_INDEX_INTERNAL_CONSTRUCTION_MULTI_LEVEL_SEGMENTOR_H

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>

#include "common_internal/construction/batch_successor.h"
#include "common_internal/construction/fraction.h"
#include "common_internal/construction/min_max_segmentation.h"
#include "common_internal/construction/segment_descriptor.h"
#include "common_internal/construction/single_level_segmentor.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

class MultiLevelSegmentorTest;

namespace fh_index {
namespace internal {
namespace segmenting {

// Resource bounds for one hierarchy level.
struct LevelConfig {
  size_t B;  // Maximum entries per bucket.
  Fraction finger_density;

  LevelConfig() : B(0), finger_density() {}
  LevelConfig(size_t b, Fraction density)
      : B(b), finger_density(density) {}
};

// Logical construction output for one level. During construction, fgd_keys
// hold left boundaries; FinalizeLevels converts them to right endpoints.
template <typename KeyType>
struct LevelData {
  parlay::sequence<KeyType> fgd_keys;
  parlay::sequence<FingerGroupDescriptorValue> fgd_values;
  parlay::sequence<size_t> finger_indexes;  // Finger positions in lower input.
  parlay::sequence<size_t> finger_offsets;  // Finger range of each FGD.

  LevelData() = default;
  LevelData(parlay::sequence<KeyType> k, parlay::sequence<FingerGroupDescriptorValue> v)
      : fgd_keys(std::move(k)), fgd_values(std::move(v)) {}

  size_t size() const { return fgd_keys.size(); }
  size_t finger_count() const { return finger_indexes.size(); }
};

// MultiLevelSegmentor: builds the logical descriptor hierarchy bottom-up.
//
// Each level contains partition endpoints, AP parameters, and positions into
// the lower-level input. Physical finger arrays and blocks are allocated later
// by FHIndexRW::Construct.
//
// Stop conditions (mutually exclusive):
// - height_limit > 0: Stop after exactly height_limit levels
// - top_level_threshold > 0: Stop when fgd_size <= threshold
//
// Level numbering: level 0 = leaf, level 1+ = internal
template <typename KeyType>
class MultiLevelSegmentor {
  friend class ::MultiLevelSegmentorTest;
 public:
  // Input data
  size_t n_;
  const KeyType* keys_;

  // Configuration
  LevelConfig leaf_config_;
  LevelConfig internal_config_;

  // Stop condition (use one or the other, not both)
  size_t height_limit_;         // 0 = not used
  size_t top_level_threshold_;  // 0 = not used

  // Max key for the last FGD (used in Rebuild scenarios)
  KeyType max_key_;

  // Logical descriptor and finger-placement data for each level.
  // levels_[0] = leaf level
  // levels_[1] = first internal level
  // ...
  parlay::sequence<LevelData<KeyType>> levels_;

  MultiLevelSegmentor(size_t n, const KeyType* keys,
                      LevelConfig leaf_config, LevelConfig internal_config,
                      size_t height_limit, size_t top_level_threshold,
                      KeyType max_key = std::numeric_limits<KeyType>::max())
      : n_(n),
        keys_(keys),
        leaf_config_(leaf_config),
        internal_config_(internal_config),
        height_limit_(height_limit),
        top_level_threshold_(top_level_threshold),
        max_key_(max_key) {
    // Validate: exactly one stop condition should be set
    assert((height_limit_ > 0) != (top_level_threshold_ > 0) &&
           "Exactly one of height_limit or top_level_threshold must be set");
    assert(leaf_config_.B > 0 && "Leaf block bound must be positive");
    assert(leaf_config_.finger_density.IsPositive() &&
           internal_config_.finger_density.IsPositive() &&
           "Finger density must be positive");
    if (!leaf_config_.finger_density.IsPositive() ||
        !internal_config_.finger_density.IsPositive()) {
      std::abort();
    }
    assert(internal_config_.B >= 5 &&
           "Internal B must leave one propagation slot and a fitting bound "
           "supported by maximal covering");
    if (internal_config_.B < 5) std::abort();
  }

  void Run() {
    InitializeLevels();
    AugmentLevels();
    FinalizeLevels();
  }

  // Get total number of levels
  size_t GetLevelCount() const { return levels_.size(); }

  // Get data for a specific level (0 = leaf)
  const LevelData<KeyType>& GetLevel(size_t level) const {
    assert(level < levels_.size());
    return levels_[level];
  }

 private:
  // Initialize levels by forward construction from leaf to top.
  void InitializeLevels() {
    parlay::sequence<LevelData<KeyType>> temp_levels;

    // Build leaf level (level 0)
    bool is_leaf = true;
    auto leaf_level = BuildOneLevel(n_, keys_, leaf_config_, is_leaf);

    size_t prev_fgd_size = leaf_level.size();
    const KeyType* prev_fgd_keys = leaf_level.fgd_keys.data();
    temp_levels.push_back(std::move(leaf_level));

    // Build internal levels until stop condition
    size_t current_level = 1;
    while (!ShouldStopInternal(current_level, prev_fgd_size)) {
      is_leaf = false;

      auto internal_level = BuildOneLevel(
          prev_fgd_size, prev_fgd_keys,
          internal_config_, is_leaf);

      prev_fgd_size = internal_level.size();
      prev_fgd_keys = internal_level.fgd_keys.data();
      temp_levels.push_back(std::move(internal_level));
      current_level++;
    }

    levels_ = std::move(temp_levels);
  }

  // Augment levels by merging finger keys from upper levels into lower levels.
  // Process from top level down to level 1.
  // fgd_values are already computed in BuildOneLevel (or by previous split).
  // For each level i (i >= 1):
  //   1. Generate finger keys from existing fgd_values
  //   2. Merge into level i-1, splitting fgd_values (no recomputation)
  void AugmentLevels() {
    size_t num_levels = levels_.size();
    if (num_levels <= 1) return;

    for (size_t i = num_levels - 1; i >= 1; --i) {
      // fgd_values already computed (BuildOneLevel or previous iteration's split)
      auto finger_keys = GenerateFingerKeysFromFGD(
          levels_[i].fgd_keys, levels_[i].fgd_values);

      // Merge into lower level, splitting fgd_values
      auto [merged_keys, merged_values] = MergeKeysWithFGDValues(
          levels_[i - 1].fgd_keys, levels_[i - 1].fgd_values, finger_keys);
      levels_[i - 1].fgd_keys = std::move(merged_keys);
      levels_[i - 1].fgd_values = std::move(merged_values);
    }
  }

  // Finalize all levels by computing finger indexes and converting fgd_keys
  // to right endpoints.
// For each level i (i >= 0):
  //   1. Compute finger indexes against the lower level (or user keys at level 0)
  //   2. Record each FGD's range in the flat finger-index sequence
  //   3. Convert fgd_keys from left boundaries to inclusive right endpoints
  // IMPORTANT: minus one must be deferred until all finger_indexes are computed,
  // because upper levels use lower levels' fgd_keys for BatchSuccessor.
  void FinalizeLevels() {
    size_t num_levels = levels_.size();
    if (num_levels == 0) return;

    // Phase 1: Compute fgd_values, finger_indexes, and finger_offsets for all levels
    for (size_t i = 0; i < num_levels; ++i) {
      // Determine the keys array for computing finger_indexes
      const KeyType* lower_keys;
      size_t lower_size;
      LevelConfig config;

      if (i == 0) {
        // Level 0: use original keys
        // fgd_values already available (computed in BuildOneLevel, then split in AugmentLevels)
        lower_keys = keys_;
        lower_size = n_;
        config = leaf_config_;
      } else {
        // Level i > 0: use level i-1's fgd_keys (still in left-side definition)
        lower_keys = levels_[i - 1].fgd_keys.data();
        lower_size = levels_[i - 1].size();
        config = internal_config_;
      }

      // Generate finger keys (left boundary + AP cut points)
      auto finger_keys = GenerateFingerKeysFromFGD(levels_[i].fgd_keys, levels_[i].fgd_values);

      // Compute finger_indexes
      size_t num_fingers = finger_keys.size();
      parlay::sequence<size_t> finger_indexes(num_fingers);
      BatchSuccessor(lower_size, lower_keys, num_fingers, finger_keys.data(), finger_indexes.data());
      levels_[i].finger_indexes = std::move(finger_indexes);

      // Compute finger_offsets
      size_t num_fgds = levels_[i].fgd_keys.size();
      parlay::sequence<size_t> finger_offsets(num_fgds + 1);
      finger_offsets[0] = 0;
      for (size_t j = 0; j < num_fgds; ++j) {
        finger_offsets[j + 1] = finger_offsets[j] + levels_[i].fgd_values[j].GetAPSize() + 1;
      }
      levels_[i].finger_offsets = std::move(finger_offsets);
    }

    // Phase 2: Convert fgd_keys to right-side definition for all levels
    // For all levels: right[j] = left[j+1] - 1 for j < n-1, right[n-1] = max_key_
    for (size_t i = 0; i < num_levels; ++i) {
      size_t fgd_size = levels_[i].fgd_keys.size();
      if (fgd_size == 0) continue;

      parlay::sequence<KeyType> right_keys(fgd_size);

      parlay::parallel_for(0, fgd_size, [&](size_t j) {
        if (j + 1 < fgd_size) {
          right_keys[j] = levels_[i].fgd_keys[j + 1] - 1;
        } else {
          right_keys[j] = max_key_;
        }
      });

      levels_[i].fgd_keys = std::move(right_keys);
    }
  }

  // Verify levels after InitializeLevels (before AugmentLevels).
  // At this point fgd_keys are left-side definition.
  // For each level, verify:
  //   1. fgd_keys are strictly increasing
  //   2. Each FGD's AP cuts are strictly < next FGD's left key
  //   3. Each finger sub-interval has <= W keys in the input keys array
  void VerifyInitializedLevels() {
    size_t num_levels = levels_.size();

    for (size_t lvl = 0; lvl < num_levels; ++lvl) {
      const auto& level = levels_[lvl];
      size_t fgd_size = level.fgd_keys.size();
      const LevelConfig& config = (lvl == 0) ? leaf_config_ : internal_config_;
      // Internal fitting reserves one slot for the descriptor boundary added
      // by AugmentLevels. Leaf buckets contain user keys and are unaffected.
      size_t W = (lvl == 0) ? config.B : config.B - 1;

      // Determine input keys for this level
      const KeyType* input_keys;
      size_t input_size;
      if (lvl == 0) {
        input_keys = keys_;
        input_size = n_;
      } else {
        input_keys = levels_[lvl - 1].fgd_keys.data();
        input_size = levels_[lvl - 1].fgd_keys.size();
      }

      // 1. fgd_keys strictly increasing
      for (size_t i = 1; i < fgd_size; ++i) {
        if (level.fgd_keys[i] <= level.fgd_keys[i - 1]) {
          std::cerr << "VerifyInitializedLevels: level " << lvl
                    << " fgd_keys not strictly increasing at " << i
                    << ": " << level.fgd_keys[i - 1] << " >= " << level.fgd_keys[i]
                    << std::endl;
        }
        assert(level.fgd_keys[i] > level.fgd_keys[i - 1] &&
               "fgd_keys must be strictly increasing");
      }

      // For each FGD, generate finger keys and check sub-intervals
      for (size_t i = 0; i < fgd_size; ++i) {
        size_t ap_size = level.fgd_values[i].GetAPSize();
        uint16_t ap_step = level.fgd_values[i].GetAPStep();
        uint64_t ap_start = level.fgd_values[i].ap_start_key_;

        // 2. All AP cuts must be < next FGD's left key
        if (i + 1 < fgd_size && ap_size > 0) {
          KeyType last_cut = ap_start + ((ap_size - 1) << ap_step);
          if (last_cut >= level.fgd_keys[i + 1]) {
            std::cerr << "VerifyInitializedLevels: level " << lvl
                      << " fgd " << i << " last AP cut " << last_cut
                      << " >= next fgd_key " << level.fgd_keys[i + 1]
                      << " (ap_start=" << ap_start << " ap_size=" << ap_size
                      << " step=" << ap_step << ")" << std::endl;
          }
          assert(last_cut < level.fgd_keys[i + 1] &&
                 "AP cut must be strictly < next FGD's left key");
        }

        // 3. Each finger sub-interval has <= W keys
        // Finger keys for this FGD: [fgd_keys[i], ap_cut_0, ..., ap_cut_{n-1}]
        // These divide the input_keys range into sub-intervals.
        // Sub-interval boundaries in key space:
        //   [fgd_keys[i], ap_cut_0), [ap_cut_0, ap_cut_1), ..., [ap_cut_{n-1}, next_fgd_key)
        size_t num_fingers = ap_size + 1;
        for (size_t f = 0; f < num_fingers; ++f) {
          // Finger interval: [lo_key, hi_key) in key space
          KeyType lo_key = (f == 0) ? level.fgd_keys[i]
                                    : level.fgd_values[i].GetAPKey(f - 1);
          KeyType hi_key;
          if (f < ap_size) {
            hi_key = level.fgd_values[i].GetAPKey(f);
          } else if (i + 1 < fgd_size) {
            hi_key = level.fgd_keys[i + 1];
          } else {
            hi_key = std::numeric_limits<KeyType>::max();
          }

          // Count keys in [lo_key, hi_key) using lower_bound
          auto lo_it = std::lower_bound(input_keys, input_keys + input_size, lo_key);
          auto hi_it = std::lower_bound(input_keys, input_keys + input_size, hi_key);
          size_t count = hi_it - lo_it;

          if (count > W) {
            std::cerr << "VerifyInitializedLevels: level " << lvl
                      << " fgd " << i << " finger " << f
                      << " has " << count << " keys (W=" << W << ")"
                      << " lo_key=" << lo_key << " hi_key=" << hi_key
                      << " ap_start=" << ap_start << " ap_size=" << ap_size
                      << " step=" << ap_step << std::endl;
          }
          assert(count <= W && "Finger sub-interval exceeds W keys");
        }
      }
    }
  }

  // Verify levels after finalization.
  // 1. For each FGD, verify that finger_indexes correctly divide lower level into sub-intervals
  // 2. For each sub-interval, verify that max value exists in lower level's fgd_keys
  void VerifyLevels() {
    size_t num_levels = levels_.size();
    if (num_levels <= 1) return;

    for (size_t i = 1; i < num_levels; ++i) {
      const auto& upper_level = levels_[i];
      const auto& lower_level = levels_[i - 1];

      // Compute finger offsets for this level
      size_t num_fgds = upper_level.fgd_values.size();
      parlay::sequence<size_t> finger_offsets(num_fgds + 1);
      finger_offsets[0] = 0;
      for (size_t j = 0; j < num_fgds; ++j) {
        finger_offsets[j + 1] = finger_offsets[j] + upper_level.fgd_values[j].GetAPSize() + 1;
      }

      size_t total_fingers = finger_offsets[num_fgds];
      assert(total_fingers == upper_level.finger_indexes.size() &&
             "Finger count mismatch");

      // For each FGD, verify sub-intervals
      KeyType fgd_interval_start_key = 0;
      for (size_t j = 0; j < num_fgds; ++j) {
        // Handle overflow: when fgd_key is UINT64_MAX, end_key would overflow to 0
        // Use a flag to indicate "no upper bound" for the last FGD
        bool fgd_has_max_bound = (upper_level.fgd_keys[j] != std::numeric_limits<KeyType>::max());
        KeyType fgd_interval_end_key = fgd_has_max_bound ? upper_level.fgd_keys[j] + 1 : 0;
        // All keys in this FGD should be in [fgd_interval_start_key, fgd_interval_end_key)

        size_t ap_size = upper_level.fgd_values[j].GetAPSize();
        size_t num_subintervals = ap_size + 1;

        // Check if this is the last sub-interval of the last FGD (no upper bound)
        bool is_last_fgd = (j == num_fgds - 1);
        bool last_fgd_unbounded = is_last_fgd && !fgd_has_max_bound;

        for (size_t k = 0; k < num_subintervals; ++k) {
          KeyType finger_interval_start_key = (k == 0) ? fgd_interval_start_key
                                                                      : upper_level.fgd_values[j].GetAPKey(k - 1);
          bool is_last_subinterval = (k == num_subintervals - 1);
          // For the last sub-interval of last FGD with UINT64_MAX key, skip upper bound check
          bool finger_unbounded = is_last_subinterval && last_fgd_unbounded;
          KeyType finger_interval_end_key = is_last_subinterval ? fgd_interval_end_key
                                                                      : upper_level.fgd_values[j].GetAPKey(k);
          // All keys in this sub-interval should be in [finger_interval_start_key, finger_interval_end_key)

          size_t finger_idx = finger_offsets[j] + k;
          size_t start_pos = upper_level.finger_indexes[finger_idx];

          // End position: next finger or end of FGD's range
          size_t end_pos;
          if (is_last_subinterval && is_last_fgd) {
            end_pos = lower_level.size();
          } else {
            end_pos = upper_level.finger_indexes[finger_idx + 1];
          }

          // Internal fitting uses config.B - 1. AugmentLevels may add one
          // descriptor to this interval, so the materialized size is <= B.
          size_t W_upper = internal_config_.B;

          assert(end_pos >= start_pos && "Invalid interval: end < start");
          assert(end_pos - start_pos <= W_upper &&
                 "Sub-interval too large");

          // Verify: keys[start_pos - 1] < finger_interval_start_key (if start_pos > 0)
          if (start_pos > 0) {
            // lower_level.fgd_keys is right-side (after minus one), so fgd_keys[pos] + 1 is the left boundary
            KeyType prev_key = lower_level.fgd_keys[start_pos - 1];
            assert(prev_key < finger_interval_start_key &&
                   "Key before start_pos should be < finger_interval_start_key");
          }

          // Verify: keys[end_pos] >= finger_interval_end_key (if end_pos < size and bounded)
          if (end_pos < lower_level.size() && !finger_unbounded) {
            // fgd_keys[end_pos] is the right boundary of interval end_pos, which is >= finger_interval_end_key - 1
            // So fgd_keys[end_pos] + 1 >= finger_interval_end_key
            KeyType end_key = lower_level.fgd_keys[end_pos];
            assert(end_key >= finger_interval_end_key &&
                   "Key at end_pos should be >= finger_interval_end_key - 1");
          }

          // Verify: if i > 1 (lower level is internal), then keys[end_pos - 1] == finger_interval_end_key - 1
          // Skip for unbounded intervals
          if (i > 1 && end_pos > start_pos && !finger_unbounded) {
            KeyType last_key = lower_level.fgd_keys[end_pos - 1];
            assert(last_key == finger_interval_end_key - 1 &&
                   "Last key in sub-interval should equal finger_interval_end_key - 1");
          }

          // Verify: all keys in [start_pos, end_pos) are within [finger_interval_start_key, finger_interval_end_key)
          for (size_t pos = start_pos; pos < end_pos; ++pos) {
            // fgd_keys[pos] is right-side, represents keys <= fgd_keys[pos]
            // The actual key range for this FGD is [prev_fgd_key + 1, fgd_keys[pos] + 1)
            // We check that fgd_keys[pos] is within the expected range
            KeyType key = lower_level.fgd_keys[pos];
            // Check key >= finger_interval_start_key - 1 without overflow
            // When start_key > 0: check key >= start_key - 1
            // When start_key == 0: any key is valid
            bool key_too_small = (finger_interval_start_key > 0) &&
                                 (key < finger_interval_start_key - 1);
            assert(!key_too_small && "Key too small for sub-interval");
            // Skip upper bound check for unbounded intervals
            if (!finger_unbounded) {
              assert(key < finger_interval_end_key &&
                     "Key too large for sub-interval");
            }
          }
        }

        // Update start key for next FGD, handle overflow for UINT64_MAX
        fgd_interval_start_key = fgd_has_max_bound ? upper_level.fgd_keys[j] + 1 : 0;
      }
    }
  }

  // Generate FGD values from fgd_keys (left-side definition).
  // fgd_keys[i] defines the left boundary of segment i.
  // Computes L/R from fgd_keys using BatchSuccessor, then generates fgd_values.
  parlay::sequence<FingerGroupDescriptorValue> GenerateFGDFromFGDKeys(
      const parlay::sequence<KeyType>& fgd_keys,
      const KeyType* keys,
      size_t num_keys,
      const LevelConfig& config) {
    size_t num_segments = fgd_keys.size();
    if (num_segments == 0) return parlay::sequence<FingerGroupDescriptorValue>();

    const Fraction final_finger_density = config.finger_density;

    // Compute L[i] from fgd_keys using BatchSuccessor
    parlay::sequence<size_t> L(num_segments);
    BatchSuccessor(num_keys, keys, num_segments, fgd_keys.data(), L.data());

    // Generate FGD values for each segment
    // R[i] = L[i+1] for i < num_segments-1, R[last] = num_keys
    parlay::sequence<FingerGroupDescriptorValue> fgd_values(num_segments);
    parlay::parallel_for(0, num_segments, [&](size_t i) {
      size_t r = (i + 1 < num_segments) ? L[i + 1] : num_keys;
      fgd_values[i] = GenerateFGDValueFromLR(
          L[i], r, keys, config.B, final_finger_density);
    });

    return fgd_values;
  }


  // // Compute (finger_key - 1) values.
  // // Precondition: finger_keys[0] == 0, finger_keys[i] > 0 for i > 0
  // // Result: [finger_keys[1]-1, finger_keys[2]-1, ..., MAX]
  // parlay::sequence<KeyType> ComputeFingerKeysMinusOne(
  //     const parlay::sequence<KeyType>& finger_keys) {
  //   if (finger_keys.empty()) return parlay::sequence<KeyType>();
  //
  //   assert(finger_keys[0] == 0 && "First finger key must be 0");
  //
  //   // Result has same size: skip index 0, add MAX at end
  //   size_t result_size = finger_keys.size();
  //   parlay::sequence<KeyType> result(result_size);
  //
  //   parlay::parallel_for(0, result_size, [&](size_t i) {
  //     if (i + 1 < result_size) {
  //       assert(finger_keys[i + 1] > 0 && "Finger keys after first must be non-zero");
  //       result[i] = finger_keys[i + 1] - 1;
  //     } else {
  //       result[i] = std::numeric_limits<KeyType>::max();
  //     }
  //   });
  //
  //   return result;
  // }

  // Split a FingerGroupDescriptorValue at key K.
  // AP cuts are at ap_start + j*(1<<step) for j=0..ap_size-1.
  // Returns {left, right} where:
  //   left:  cuts before K  (ap_start, j, step)
  //   right: cuts at/after K (ap_start + j*(1<<step), ap_size - j, step)
  static std::pair<FingerGroupDescriptorValue, FingerGroupDescriptorValue>
  SplitFGDValue(const FingerGroupDescriptorValue& val, KeyType K) {
    size_t ap_size = val.GetAPSize();
    uint16_t step = val.GetAPStep();
    uint64_t ap_start = val.ap_start_key_;

    if (ap_size == 0) {
      // No cuts: both sub-segments get ap_size = 0
      return {FingerGroupDescriptorValue(ap_start, 0, 0, nullptr),
              FingerGroupDescriptorValue(ap_start, 0, 0, nullptr)};
    }

    // Find first j where cut[j] >= K
    // cut[j] = ap_start + j*(1<<step), so j = ceil((K - ap_start) / (1<<step))
    size_t j;
    if (K <= ap_start) {
      j = 0;
    } else {
      uint64_t diff = K - ap_start;
      j = (diff + (1ULL << step) - 1) >> step;
    }
    j = std::min(j, ap_size);

    // Left: j cuts before K
    FingerGroupDescriptorValue left_val(ap_start, j, step, nullptr);

    // Right: ap_size - j cuts at/after K
    uint64_t right_start = (j < ap_size) ? val.GetAPKey(j) : ap_start;
    FingerGroupDescriptorValue right_val(right_start, ap_size - j, step, nullptr);

    return {left_val, right_val};
  }

  // Merge two sorted key sequences, removing duplicates
  parlay::sequence<KeyType> MergeKeys(
      const parlay::sequence<KeyType>& a_keys,
      const parlay::sequence<KeyType>& b_keys) {
    if (b_keys.empty()) return a_keys;
    if (a_keys.empty()) return b_keys;

    auto merged = parlay::merge(a_keys, b_keys);

    // Remove duplicates
    auto unique_flags = parlay::delayed_tabulate(merged.size(), [&](size_t i) {
      return i == 0 || merged[i] != merged[i - 1];
    });

    return parlay::pack(merged, unique_flags);
  }

  // Merge finger_keys into fgd_keys + fgd_values by splitting fgd_values.
  // For each segment [fgd_keys[i], fgd_keys[i+1]):
  //   - For each finger_key K in (fgd_keys[i], fgd_keys[i+1]):
  //       split current fgd_value at K, emit left portion, continue with right
  //   - Skip duplicates (finger_key == fgd_key)
  //   - Emit remaining portion
  std::pair<parlay::sequence<KeyType>, parlay::sequence<FingerGroupDescriptorValue>>
  MergeKeysWithFGDValues(
      const parlay::sequence<KeyType>& fgd_keys,
      const parlay::sequence<FingerGroupDescriptorValue>& fgd_values,
      const parlay::sequence<KeyType>& finger_keys) {
    size_t n_fgd = fgd_keys.size();
    size_t n_finger = finger_keys.size();

    if (n_finger == 0) {
      return {fgd_keys, fgd_values};
    }

    // Step 1: For each segment, binary search for the range of finger_keys
    // that fall strictly within (fgd_keys[i], fgd_keys[i+1])
    parlay::sequence<size_t> fi_start(n_fgd);
    parlay::sequence<size_t> fi_end(n_fgd);

    parlay::parallel_for(0, n_fgd, [&](size_t i) {
      fi_start[i] = std::lower_bound(
          finger_keys.begin(), finger_keys.end(), fgd_keys[i]) - finger_keys.begin();
      if (i + 1 < n_fgd) {
        fi_end[i] = std::lower_bound(
            finger_keys.begin(), finger_keys.end(), fgd_keys[i + 1]) - finger_keys.begin();
      } else {
        fi_end[i] = n_finger;
      }
    });

    // Step 2: Each segment produces output entries.
    // If the first finger_key == fgd_keys[i], the finger_key replaces the fgd entry (no extra +1).
    auto output_counts = parlay::tabulate(n_fgd, [&](size_t i) -> size_t {
      size_t n_fingers_in_range = fi_end[i] - fi_start[i];
      if (n_fingers_in_range > 0 && finger_keys[fi_start[i]] == fgd_keys[i]) {
        return n_fingers_in_range;  // finger_key replaces fgd_key, no +1
      }
      return n_fingers_in_range + 1;
    });

    // Step 3: Prefix sum for output offsets
    auto [output_offsets, total_output] = parlay::scan(output_counts);

    // Step 4: Fill output in parallel
    parlay::sequence<KeyType> merged_keys(total_output);
    parlay::sequence<FingerGroupDescriptorValue> merged_values(total_output);

    // When ap_start == fgd_key, the first AP cut is redundant (same as fgd_key boundary).
    // Remove it: the sub-interval [key, key) was empty, so merging with next doesn't exceed W.
    auto fix_ap = [&](const auto& key, auto& value) {
      if (value.GetAPSize() > 0 && value.ap_start_key_ == static_cast<uint64_t>(key)) {
        uint64_t new_start = value.ap_start_key_ + (1ULL << value.GetAPStep());
        size_t new_size = value.GetAPSize() - 1;
        uint16_t step = value.GetAPStep();
        value = FingerGroupDescriptorValue(new_start, new_size, step, nullptr);
      }
    };

    parlay::parallel_for(0, n_fgd, [&](size_t i) {
      size_t out_pos = output_offsets[i];
      FingerGroupDescriptorValue current_val = fgd_values[i];
      KeyType current_key = fgd_keys[i];

      for (size_t fi = fi_start[i]; fi < fi_end[i]; fi++) {
        KeyType K = finger_keys[fi];
        if (K == current_key) {
          continue;  // Skip duplicate finger_key
        }

        auto [left_val, right_val] = SplitFGDValue(current_val, K);

        fix_ap(current_key, left_val);

        merged_keys[out_pos] = current_key;
        merged_values[out_pos] = left_val;
        out_pos++;

        current_key = K;
        current_val = right_val;
      }

      fix_ap(current_key, current_val);
      // Emit remaining portion
      merged_keys[out_pos] = current_key;
      merged_values[out_pos] = current_val;

      assert((out_pos == output_offsets[i] + output_counts[i] - 1) &&
             "Output position mismatch");
    });

    return {std::move(merged_keys), std::move(merged_values)};
  }

  bool ShouldStopInternal(size_t next_level, size_t prev_fgd_size) const {
    if (height_limit_ > 0) {
      // Stop after height_limit levels
      return next_level >= height_limit_;
    } else {
      // Stop when top level is small enough
      return prev_fgd_size <= top_level_threshold_;
    }
  }

  // The RW boundary-propagation contract reserves one descriptor slot in
  // every internal bucket. Propagation does not add user keys to leaf buckets.
  static size_t FittingBlockBound(const LevelConfig& config, bool is_leaf) {
    if (is_leaf) {
      return config.B;
    }
    assert(config.B >= 5 &&
           "Internal B must leave one propagation slot and a fitting bound "
           ">= 4");
    return config.B - 1;
  }

  // Build one level of segmentation using static functions from segmenting.h
  LevelData<KeyType> BuildOneLevel(size_t size, const KeyType* keys,
                                   const LevelConfig& config, bool is_leaf) {
    if (size == 0) {
      return LevelData<KeyType>();
    }

    size_t fitting_B = FittingBlockBound(config, is_leaf);

    // The overlapping-cover phase uses half the final finger-density
    // budget, equivalently twice as many keys per finger. Trimming the cover
    // into a partition may then at most double finger density.
    const Fraction final_finger_density = config.finger_density;
    const Fraction covering_finger_density = final_finger_density.Halve();

    // Compose segmentation pipeline: build -> select -> adjust
    auto segments = BuildMaximalCoveringStatic(
        size, keys, fitting_B, covering_finger_density);
    if (segments.empty()) {
      return LevelData<KeyType>();
    }
    auto valid_segments = SegmentSelectionStatic(segments);
    SegmentAdjustmentStatic(valid_segments);

    assert(valid_segments.size() > 0 && "Segmentation should produce at least one segment");

    // Generate fgd_keys from valid_segments
    size_t fgd_size = valid_segments.size();
    parlay::sequence<KeyType> fgd_keys(fgd_size);

    // Generate fgd_keys using left-side definition
    GenerateFGDKeysLeftSide(valid_segments, size, keys, max_key_, is_leaf,
                            fgd_size, fgd_keys.data());

    // Compute fgd_values immediately while valid_segments are available.
    // Internal levels fit to B-1, reserving one descriptor for propagation.
    // Leaf levels fit to B because propagation never adds user keys.
    parlay::sequence<FingerGroupDescriptorValue> fgd_values(fgd_size);
    parlay::parallel_for(0, fgd_size, [&](size_t i) {
      fgd_values[i] = GenerateFGDValueFromLR(
          valid_segments[i].l, valid_segments[i].r,
          keys, fitting_B, final_finger_density);
    });
    return LevelData<KeyType>(std::move(fgd_keys), std::move(fgd_values));
  }
};

}  // namespace segmenting
}  // namespace internal
}  // namespace fh_index

#endif  // FH_INDEX_INTERNAL_CONSTRUCTION_MULTI_LEVEL_SEGMENTOR_H
