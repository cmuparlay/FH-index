#ifndef FH_INDEX_INTERNAL_CONSTRUCTION_MIN_MAX_SEGMENTATION_H
#define FH_INDEX_INTERNAL_CONSTRUCTION_MIN_MAX_SEGMENTATION_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "common_internal/utility/math.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

namespace fh_index {
namespace internal {
namespace segmenting {

// ============================================================================
// MinMaxSegmentation: Find optimal cut points for a segment
// ============================================================================
//
// Problem: Given n sorted keys, find at most k cut points that divide
// the keys into segments, minimizing the maximum segment size.
//
// Solution: Cut points form an Arithmetic Progression (AP) defined by (x, t):
//   - x: step exponent, AP step size = 2^x
//   - t: starting bucket index
//   - Cut points: t*2^x, (t+1)*2^x, (t+2)*2^x, ..., (t+num_cuts-1)*2^x
//
// The algorithm:
//   1. Enumerate x from small to large (fine to coarse granularity)
//   2. For each x, binary search on W (target max segment size)
//   3. Find the smallest W such that we can cover all keys with <= k cuts
//
// Parameters:
//   - n: number of keys
//   - keys: sorted key array
//   - k: maximum allowed number of cuts (= number of fingers - 1)
//   - W_upper: upper bound on segment size (typically B, the block size)
//
// Result:
//   - x: step exponent (AP step = 2^x)
//   - t: first cut bucket (first cut point = t << x)
//   - num_cuts: actual number of cuts used
//   - max_segment_size: achieved maximum segment size
//
// ============================================================================

template <typename KeyType>
struct MinMaxSegmentationResult {
  uint8_t x;      // Step exponent: AP step size = 2^x
  KeyType t;      // First cut bucket: first cut point = t << x
  size_t num_cuts;        // Number of AP cut points
  size_t max_segment_size; // Maximum segment size achieved

  // Get the i-th cut point (0-indexed)
  // Cut points: t*2^x, (t+1)*2^x, ..., (t+num_cuts-1)*2^x
  KeyType GetCutPoint(size_t i) const {
    return (static_cast<KeyType>(t) + i) << x;
  }
};

template <typename KeyType>
class MinMaxSegmentation {
 public:
  using HistogramType = parlay::sequence<std::pair<KeyType, size_t>>;

  MinMaxSegmentation(size_t n, const KeyType* keys, size_t k, size_t W_upper)
      : n_(n), keys_(keys), k_(k), W_upper_(W_upper) {
    if (n > 0) {
      max_key_ = keys[n - 1];
    } else {
      max_key_ = 0;
    }
  }

  MinMaxSegmentationResult<KeyType> Solve() {
    if (n_ == 0) return {0, 0, 0, 0};
    if (k_ == 0) return {0, 0, 0, n_};

    MinMaxSegmentationResult<KeyType> best;
    best.max_segment_size = n_;
    best.x = 0;
    best.t = 0;
    best.num_cuts = 0;

    auto [x_min, max_bits, histogram, valid] = ComputeXRangeAndInitialHistogram();
    if (!valid) return best;

    // Outer loop: enumerate X from x_min to max_bits
    for (uint8_t x = x_min; x <= max_bits; x++) {
      // Inner loop: binary search on W for this X
      auto result = BinarySearchWWithHistogram(x, histogram);
      if (result.max_segment_size < best.max_segment_size) {
        best = result;
        if (best.max_segment_size <= (n_ + k_) / (k_ + 1)) {
          return best;  // Optimal, stop early
        }
      }

      // Merge histogram for next x
      if (x < max_bits) {
        histogram = MergeHistogramForNextX(histogram);
      }
    }

    return best;
  }

 private:
  size_t n_;
  const KeyType* keys_;
  size_t k_;
  size_t W_upper_;
  KeyType max_key_;

  // Compute valid X range and build initial histogram at x_min
  // Returns {x_min, max_bits, histogram}
  // If x_min > max_bits, no feasible X exists
  auto ComputeXRangeAndInitialHistogram() {
    struct Result {
      uint8_t x_min;
      uint8_t max_bits;
      HistogramType histogram;
      bool valid;
    };

    uint8_t max_bits = (max_key_ > 0)
        ? static_cast<uint8_t>(fh_index::internal::Log2(max_key_) + 1)
        : 0;

    // Find minimum X by checking cut count constraint with W = W_upper
    uint8_t x_min = 0;
    if (W_upper_ < n_) {
      // Enumerate x to find the first valid one
      for (uint8_t x = 0; x <= max_bits; x++) {
        auto [t_first, t_last] = ComputeFirstLast(W_upper_, x);
        auto [num_cuts, valid] = CheckCutCount(t_first, t_last);
        if (valid) {
          x_min = x;
          break;
        }
        if (x == max_bits) {
          // No valid x found
          auto empty_ids = parlay::tabulate(size_t{0}, [](size_t) -> KeyType { return 0; });
          return Result{static_cast<uint8_t>(max_bits + 1), max_bits,
                        parlay::histogram_by_key(empty_ids), false};
        }
      }
    }

    // Build histogram directly at x_min level
    auto bucket_ids = parlay::tabulate(n_, [&](size_t i) {
      return keys_[i] >> x_min;
    });
    return Result{x_min, max_bits, parlay::histogram_by_key(bucket_ids), true};
  }

  // Binary search on W for a fixed X with precomputed histogram
  template <typename Histogram>
  MinMaxSegmentationResult<KeyType> BinarySearchWWithHistogram(
      uint8_t x, const Histogram& histogram) {
    MinMaxSegmentationResult<KeyType> best;
    best.max_segment_size = n_;
    best.x = x;
    best.t = 0;
    best.num_cuts = 0;

    // Get max bucket size from histogram - this is a hard lower bound for W
    size_t max_bucket = parlay::reduce(
        parlay::delayed_map(histogram, [](const auto& p) { return p.second; }),
        parlay::maximum<size_t>());

    // W_lower: max of theoretical bound and max bucket size
    size_t W_lower = std::max((n_ + k_) / (k_ + 1), max_bucket);
    size_t W_upper = W_upper_;

    // Early exit: if max bucket > W_upper, infeasible for this x
    if (max_bucket > W_upper) {
      return best;
    }

    while (W_lower < W_upper) {
      size_t W_mid = W_lower + (W_upper - W_lower) / 2;
      auto result = TryW(W_mid, x);
      if (result.max_segment_size <= W_mid) {
        best = result;
        W_upper = W_mid;
      } else {
        W_lower = W_mid + 1;
      }
    }

    // Final check
    auto result = TryW(W_lower, x);
    if (result.max_segment_size <= W_lower &&
        result.max_segment_size < best.max_segment_size) {
      best = result;
    }

    return best;
  }

  // Merge adjacent bucket pairs: bucket[b] = bucket[2b] + bucket[2b+1]
  template <typename Histogram>
  auto MergeHistogramForNextX(const Histogram& hist) {
    // Group by (bucket_id >> 1) and sum counts
    auto shifted = parlay::delayed_map(hist, [](const auto& p) {
      return std::make_pair(p.first >> 1, p.second);
    });
    return parlay::reduce_by_key(shifted, parlay::plus<size_t>());
  }

  // Compute t_first and t_last from W and x
  // t_first: first cut bucket (prefix has keys[0..W-1])
  // t_last: last cut bucket + 1 (suffix has keys[n-W..n-1])
  struct FirstLast {
    KeyType t_first;
    KeyType t_last;
  };

  FirstLast ComputeFirstLast(size_t W, uint8_t x) {
    KeyType t_first = keys_[W] >> x;
    KeyType t_last = (keys_[n_ - W - 1] >> x) + 1;
    return {t_first, t_last};
  }

  // Check if num_cuts <= k
  // Returns {num_cuts, valid}
  struct CutCountResult {
    size_t num_cuts;
    bool valid;
  };

  CutCountResult CheckCutCount(KeyType t_first, KeyType t_last) {
    if (t_last <= t_first) {
      // Single cut suffices
      return {1, true};
    }
    size_t num_cuts = t_last - t_first + 1;
    return {num_cuts, num_cuts <= k_};
  }

  // Try to achieve max_segment_size <= W with specific X
  // Note: max_bucket constraint is already satisfied since W >= W_lower >= max_bucket
  MinMaxSegmentationResult<KeyType> TryW(size_t W, uint8_t x) {
    MinMaxSegmentationResult<KeyType> result;
    result.max_segment_size = n_;
    result.x = x;
    result.t = 0;
    result.num_cuts = 0;

    // Edge case: W >= n means no cuts needed
    if (W >= n_) {
      return result;
    }

    // Step 1: Compute cut boundaries
    auto [t_first, t_last] = ComputeFirstLast(W, x);

    // Step 2: Check cut count constraint
    auto [num_cuts, valid] = CheckCutCount(t_first, t_last);
    if (!valid) {
      return result;  // Too many cuts
    }

    // Feasible: max_segment_size = W (since W >= max_bucket is guaranteed)
    result.max_segment_size = W;
    result.t = t_first;
    result.num_cuts = num_cuts;
    return result;
  }
};

template <typename KeyType>
MinMaxSegmentationResult<KeyType> SolveMinMaxSegmentation(
    size_t n, const KeyType* keys, size_t k, size_t W_upper) {
  MinMaxSegmentation<KeyType> solver(n, keys, k, W_upper);
  return solver.Solve();
}

}  // namespace segmenting
}  // namespace internal
}  // namespace fh_index

#endif  // FH_INDEX_INTERNAL_CONSTRUCTION_MIN_MAX_SEGMENTATION_H
