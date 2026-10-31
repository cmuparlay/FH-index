#ifndef FH_INDEX_INTERNAL_CONSTRUCTION_SINGLE_LEVEL_SEGMENTOR_H
#define FH_INDEX_INTERNAL_CONSTRUCTION_SINGLE_LEVEL_SEGMENTOR_H

#include <cstdint>
#include <cstdlib>
#include <utility>

#include "common_internal/construction/batch_successor.h"
#include "common_internal/construction/fraction.h"
#include "common_internal/construction/maximal_covering_builder.h"
#include "common_internal/construction/min_max_segmentation.h"
#include "common_internal/construction/segment_descriptor.h"
#include "common_internal/representation/finger_group_descriptor.h"
#include "common_internal/utility/common.h"
#include "common_internal/utility/math.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "interval_covering.h"

namespace fh_index {
namespace internal {
namespace segmenting {

// Use FingerGroupDescriptorValue from parent namespace
using FingerGroupDescriptorValue = fh_index::internal::FingerGroupDescriptorValue;

static constexpr size_t kParallelThreshold = 1024;

struct LinkListNode {
  bool valid;
  size_t nxt;
};

// ============================================================================
// Static segmentation functions (reusable by MultiLevelSegmentor)
// ============================================================================

// Build maximal covering segments for the given keys.
// Returns sequence of overlapping segments.
template <typename KeyType>
parlay::sequence<SegmentDescriptor> BuildMaximalCoveringStatic(
    size_t n, const KeyType* keys, size_t B, Fraction finger_density) {
  MaximalCoveringBuilder<KeyType> builder(n, keys, B, finger_density);
  builder.Build();
  parlay::sequence<SegmentDescriptor> segments;
  builder.TakeOwnership(segments);
  return segments;
}

// Select minimum set of segments that cover the entire range using interval covering algorithm.
// Returns sequence of selected segments (may still overlap, will be adjusted later).
inline parlay::sequence<SegmentDescriptor> SegmentSelectionStatic(
    const parlay::sequence<SegmentDescriptor>& segments) {
  if (segments.empty()) {
    return parlay::sequence<SegmentDescriptor>();
  }

  size_t segment_count = segments.size();
  auto L = [&](size_t i) { return segments[i].l; };
  auto R = [&](size_t i) { return segments[i].r; };

  IntervalCovering ic(segment_count, L, R);
  ic.Run();
  return parlay::pack(segments, ic.valid);
}

// Adjust segments to ensure no overlap by splitting at midpoints.
// Modifies segments in place.
inline void SegmentAdjustmentStatic(parlay::sequence<SegmentDescriptor>& segments) {
  if (segments.size() <= 1) return;

  parlay::parallel_for(1, segments.size(), [&](size_t i) {
    if (segments[i].l < segments[i - 1].r) {
      size_t mid = (segments[i].l + segments[i - 1].r) >> 1;
      segments[i].l = segments[i - 1].r = mid;
    }
  });
}

// ============================================================================
// FGD Key Generation: Fill fgd_keys array from segments
// ============================================================================
//
// Relationship: left[i] = right[i-1] + 1 for i > 0
//
// - right[i]: upper bound key for segment i (keys <= right[i] belong to segment i)
// - left[i]: lower bound key for segment i (keys >= left[i] belong to segment i)
//
// For a key k to belong to segment i:
//   left[i] <= k <= right[i]
//   equivalently: right[i-1] < k <= right[i]

// Compute the boundary key between segment (i-1) and segment i.
// This equals right[i-1] (upper bound of segment i-1).
// Precondition: 0 < i <= num_segments
template <typename KeyType>
inline KeyType ComputeSegmentBoundary(
    size_t i,
    const parlay::sequence<SegmentDescriptor>& segments,
    size_t n,
    const KeyType* keys,
    KeyType max_key,
    bool is_leaf_layer) {
  assert(i > 0);
  size_t r = segments[i - 1].r;
  if (r == n) {
    return max_key;
  } else if (is_leaf_layer) {
    return keys[r - 1] + (keys[r] - keys[r - 1]) / 2;
  } else {
    return keys[r - 1];
  }
}

// Generate fgd_keys using left-side definition (for finger key generation)
// left[0] = 0
// left[i] = right[i-1] + 1 = ComputeSegmentBoundary(i) + 1 for i > 0
// Output: fills fgd_keys[0..num_segments-1]
template <typename KeyType>
inline void GenerateFGDKeysLeftSide(
    const parlay::sequence<SegmentDescriptor>& segments,
    size_t n,
    const KeyType* keys,
    KeyType max_key,
    bool is_leaf_layer,
    size_t num_segments,
    KeyType* fgd_keys) {
  if (num_segments == 0) return;

  parlay::parallel_for(0, num_segments, [&](size_t i) {
    if (i == 0) {
      fgd_keys[i] = 0;
    } else {
      fgd_keys[i] = ComputeSegmentBoundary(i, segments, n, keys, max_key, is_leaf_layer) + 1;
    }
  });
}

// Generate fgd_keys using right-side definition (for output/search)
// right[i] = ComputeSegmentBoundary(i+1)
// Output: fills fgd_keys[0..num_segments-1]
template <typename KeyType>
inline void GenerateFGDKeysRightSide(
    const parlay::sequence<SegmentDescriptor>& segments,
    size_t n,
    const KeyType* keys,
    KeyType max_key,
    bool is_leaf_layer,
    size_t num_segments,
    KeyType* fgd_keys) {
  if (num_segments == 0) return;

  parlay::parallel_for(0, num_segments, [&](size_t i) {
    fgd_keys[i] = ComputeSegmentBoundary(i + 1, segments, n, keys, max_key, is_leaf_layer);
  });
}

// ============================================================================
// FGD Value Generation: Generate AP parameters from segment boundaries
// ============================================================================

// Generate FingerGroupDescriptorValue for a single segment.
// Input: l, r (segment boundaries), keys, B (max bucket size bound),
// finger_density
// Output: FingerGroupDescriptorValue with AP parameters (finger_array_ = nullptr)
template <typename KeyType>
FingerGroupDescriptorValue GenerateFGDValueFromLR(
    size_t l, size_t r, const KeyType* keys, size_t B,
    Fraction finger_density) {
  size_t segment_size = r - l;

  // Handle small segments that don't need full MinMaxSegmentation
  if (segment_size <= B) {
    // Single finger covers all keys (ap_size = 0 means 0 cuts = 1 finger)
    return FingerGroupDescriptorValue(keys[l], 0, 0, nullptr);
  }

  if (segment_size < 2 * B) {
    // Need 2 fingers (1 cut at midpoint) to ensure each finger covers <= B keys
    size_t mid = l + segment_size / 2;
    return FingerGroupDescriptorValue(keys[mid], 1, 0, nullptr);
  }

  const size_t max_fingers = finger_density.FloorMultiply(segment_size);
  assert(max_fingers > 0);
  if (max_fingers == 0) std::abort();
  const size_t max_cuts = max_fingers - 1;

  // Use SolveMinMaxSegmentation to find optimal (X, t)
  const KeyType* segment_keys = keys + l;
  auto result = SolveMinMaxSegmentation<KeyType>(
      segment_size, segment_keys, max_cuts, B);

  // Extract AP parameters from result
  uint16_t ap_step = result.x;
  KeyType ap_start_key = static_cast<KeyType>(result.t) << ap_step;
  size_t ap_size = result.num_cuts;

  assert(result.max_segment_size <= B);
  assert(result.num_cuts + 1 <= max_fingers);

  return FingerGroupDescriptorValue(ap_start_key, ap_size, ap_step, nullptr);
}

// ============================================================================
// Finger Key Generation: Generate finger keys from FGD (left-side logic)
// ============================================================================

// Generate finger keys from fgd_keys and fgd_values.
// First finger of FGD i = fgd_keys[i] (left boundary)
// Remaining fingers = AP cut points from fgd_values[i]
template <typename KeyType>
parlay::sequence<KeyType> GenerateFingerKeysFromFGD(
    const parlay::sequence<KeyType>& fgd_keys,
    const parlay::sequence<FingerGroupDescriptorValue>& fgd_values) {
  size_t fgd_size = fgd_keys.size();
  if (fgd_size == 0) {
    return parlay::sequence<KeyType>();
  }

  // Count total finger keys
  auto finger_counts = parlay::delayed_tabulate(
      fgd_size, [&](size_t i) { return fgd_values[i].GetAPSize() + 1; });

  auto [finger_offsets, total_fingers] = parlay::scan(finger_counts);

  parlay::sequence<KeyType> finger_keys(total_fingers);

  parlay::parallel_for(0, fgd_size, [&](size_t i) {
    size_t offset = finger_offsets[i];
    size_t ap_count = fgd_values[i].GetAPSize();

    // First finger = fgd_keys[i] (left boundary of segment)
    finger_keys[offset] = fgd_keys[i];

    // AP cut points
    for (size_t j = 0; j < ap_count; ++j) {
      finger_keys[offset + 1 + j] = fgd_values[i].GetAPKey(j);
    }
  });

  return finger_keys;
}

template <typename KeyType>
class Segmentor {
 public:
  // input
  size_t n;
  const KeyType* keys;
  size_t B;
  Fraction covering_finger_density;
  KeyType max_key;      // max key for last FGD (UINT64_MAX if full dataset)
  bool is_leaf_layer;   // true: use midpoint for fgd_key; false: use keys[r-1]

  // output
  size_t fgd_size = 0, fgd_array_size = 0;
  KeyType* fgd_keys = nullptr;
  FingerGroupDescriptorValue* fgd_values = nullptr;

  Segmentor(size_t n, const KeyType* keys, size_t B,
            Fraction covering_finger_density,
            KeyType max_key = std::numeric_limits<KeyType>::max(), bool is_leaf_layer = true)
      : n(n), keys(keys), B(B),
        covering_finger_density(covering_finger_density),
        max_key(max_key), is_leaf_layer(is_leaf_layer) {
    assert(covering_finger_density.IsPositive());
    if (!covering_finger_density.IsPositive()) std::abort();
    // Arrays are allocated by MaximalCoveringBuilder and transferred via TakeOwnership
  }

  Segmentor(const Segmentor&) = delete;
  Segmentor& operator=(const Segmentor&) = delete;
  Segmentor(Segmentor&&) = delete;
  Segmentor& operator=(Segmentor&&) = delete;
  ~Segmentor() {
    delete[] fgd_keys;
    delete[] fgd_values;
  }

  // Transfer both output arrays to the next RO layer. Otherwise this segmentor
  // owns them, including in standalone analysis and benchmark programs.
  auto ReleaseFGDArrays() {
    return std::make_pair(std::exchange(fgd_keys, nullptr),
                          std::exchange(fgd_values, nullptr));
  }

  using Segment = SegmentDescriptor;

  // overlapping segments (output of MaximalCoveringBuilder)
  parlay::sequence<SegmentDescriptor> segments;
  void BuildMaximalCovering() {
    segments = BuildMaximalCoveringStatic(
        n, keys, B, covering_finger_density);
  }

  parlay::sequence<Segment> valid_segments;
  void SegmentSelection() {
    valid_segments = SegmentSelectionStatic(segments);
  }

  // Adjust segments to ensure no overlap
  void SegmentAdjustment() {
    SegmentAdjustmentStatic(valid_segments);
  }

  size_t total_finger_count;
  parlay::sequence<KeyType> finger_keys;
  parlay::sequence<size_t> finger_offsets;

  // Allocate arrays and generate FGD using left-side keys
  void GenerateFGD() {
    delete[] fgd_keys;
    delete[] fgd_values;
    fgd_keys = nullptr;
    fgd_values = nullptr;
    fgd_size = valid_segments.size();
    fgd_array_size = fh_index::internal::RoundUp(fgd_size, kArrayPaddingSize);

    // Allocate fgd_keys and fill with left-side definition
    fgd_keys = new KeyType[fgd_array_size];
    GenerateFGDKeysLeftSide(valid_segments, n, keys, max_key, is_leaf_layer,
                            fgd_size, fgd_keys);

    // Fill padding with sentinels
    for (size_t i = fgd_size; i < fgd_array_size; i++) {
      fgd_keys[i] = std::numeric_limits<KeyType>::max();
    }

    // Allocate and fill fgd_values
    fgd_values = new FingerGroupDescriptorValue[fgd_array_size];
    // Segment adjustment may double finger density, so the covering phase
    // uses half the final density.
    const Fraction final_finger_density = covering_finger_density.Double();
    parlay::parallel_for(0, fgd_size, [&](size_t i) {
      fgd_values[i] = GenerateFGDValueFromLR(
          valid_segments[i].l, valid_segments[i].r,
          keys, B, final_finger_density);
    });

    // Fill padding with default values
    for (size_t i = fgd_size; i < fgd_array_size; i++) {
      fgd_values[i] = FingerGroupDescriptorValue();
    }
  }

  // Generate finger keys using left-side fgd_keys
  // First finger of FGD i = fgd_keys[i]
  void GenerateFingerKeys() {
    auto finger_counts = parlay::delayed_tabulate(
        fgd_size, [&](size_t i) { return fgd_values[i].GetAPSize() + 1; });

    std::tie(finger_offsets, total_finger_count) = parlay::scan(finger_counts);

    finger_keys = parlay::sequence<KeyType>(total_finger_count);

    parlay::parallel_for(0, fgd_size, [&](size_t i) {
      size_t offset = finger_offsets[i];
      size_t ap_count = fgd_values[i].GetAPSize();

      // First finger = fgd_keys[i] (left boundary)
      finger_keys[offset] = fgd_keys[i];

      // AP cut points
      for (size_t j = 0; j < ap_count; ++j) {
        finger_keys[offset + 1 + j] = fgd_values[i].GetAPKey(j);
      }
    });
  }

  // Convert fgd_keys from left-side to right-side for output
  void ConvertFGDKeysToRightSide() {
    GenerateFGDKeysRightSide(valid_segments, n, keys, max_key, is_leaf_layer,
                             fgd_size, fgd_keys);
  }

  parlay::sequence<size_t> finger_indexes;
  // Helper function: Compute finger positions using BatchSuccessor
  void ComputeFingerPositions() {
    finger_indexes = parlay::sequence<size_t>(finger_keys.size());
    BatchSuccessor(n, keys, finger_keys.size(), finger_keys.data(),
                   finger_indexes.data());
  }

  void Run() {
    // Step 1: Segmentation
    BuildMaximalCovering();
    SegmentSelection();
    SegmentAdjustment();

    // Step 2: Generate FGD with left-side keys
    GenerateFGD();

    // Step 3: Generate finger keys using left-side definition
    GenerateFingerKeys();

    // Step 4: Compute finger positions
    ComputeFingerPositions();

    // Step 5: Convert fgd_keys to right-side (old definition) for output
    ConvertFGDKeysToRightSide();
  }

};  // class Segmentor

}  // namespace segmenting
}  // namespace internal
}  // namespace fh_index

#endif  // FH_INDEX_INTERNAL_CONSTRUCTION_SINGLE_LEVEL_SEGMENTOR_H
