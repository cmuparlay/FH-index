#ifndef FH_INDEX_INTERNAL_CONSTRUCTION_MAXIMAL_COVERING_BUILDER_H
#define FH_INDEX_INTERNAL_CONSTRUCTION_MAXIMAL_COVERING_BUILDER_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <tuple>
#include <utility>

#include "common_internal/construction/fraction.h"
#include "common_internal/construction/segment_descriptor.h"
#include "common_internal/utility/math.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/slice.h"

namespace fh_index {
namespace internal {
namespace segmenting {

static constexpr size_t kMaxX = 64;

inline uint64_t GetX(uint64_t xi, uint64_t xi_B) {
  return fh_index::internal::Log2(xi ^ xi_B);
}

// MaximalCoveringBuilder: Responsible for building r_pos, X, X_of_start arrays
// and generating overlapping segments from r_pos
// Verified through VerifyRPos()
template <typename KeyType>
class MaximalCoveringBuilder {
 public:
  using Segment = SegmentDescriptor;

  // input
  size_t n;
  const KeyType* keys;
  size_t B;
  Fraction finger_density;

  // output: overlapping maximal segments
  parlay::sequence<Segment> segments;

  // internal arrays (owned by this class)
  size_t* r_pos;
  uint8_t* X;
  uint8_t* X_of_start;
  int64_t* L;
  int64_t* R;


  MaximalCoveringBuilder(size_t n, const KeyType* keys, size_t B,
                         Fraction finger_density)
      : n(n),
        keys(keys),
        B(B),
        finger_density(finger_density) {
    assert(finger_density.IsPositive());
    if (!finger_density.IsPositive()) std::abort();
    // Allocate arrays
    r_pos = new size_t[n + 1];
    X = new uint8_t[n + 1];
    X_of_start = new uint8_t[n + 1];
    L = new int64_t[n + 1];
    R = new int64_t[n + 1];
  }

  MaximalCoveringBuilder(const MaximalCoveringBuilder&) = delete;
  MaximalCoveringBuilder& operator=(const MaximalCoveringBuilder&) = delete;
  MaximalCoveringBuilder(MaximalCoveringBuilder&&) = delete;
  MaximalCoveringBuilder& operator=(MaximalCoveringBuilder&&) = delete;

  ~MaximalCoveringBuilder() {
    delete[] r_pos;
    delete[] X;
    delete[] X_of_start;
    delete[] L;
    delete[] R;
  }

  // Transfer ownership of arrays to caller
  // After calling this, MaximalCoveringBuilder no longer owns the arrays
  void TakeOwnership(parlay::sequence<Segment>& out_segments) {
    out_segments = std::move(segments);
  }

  // Helper function: Check if L[l] >= R[r] (used in BuildRPosInternal)
  inline bool ValidLR_RPos(size_t l, size_t r) const {
    return R[r] <= L[l];
  }

  // ==================== Build X ====================

  void BuildX() {
    parlay::parallel_for(0, n - 1, [&](size_t i) {
      KeyType key_l = keys[i];
      KeyType key_r = (i + B >= n) ? std::numeric_limits<KeyType>::max() : keys[i + B];
      // X[i] = max{x | exist k, (k * 2^X[i]) in (key_l, key_r]}
      X[i] = fh_index::internal::Log2(key_l ^ key_r);
    });
  }

  // ==================== Build RPos Internal ====================

  void LinkRPos(size_t prefix_pos, size_t suffix_pos, uint8_t min_X) {
    size_t l = prefix_pos - B;
    size_t r = suffix_pos + B + 1;
    if (r_pos[l] < r) {
      r_pos[l] = r;
      X_of_start[l] = min_X;
    }
    assert(ValidLR_Segment(l, r, min_X));
  }


  using SizeTSlice = parlay::slice<size_t*, size_t*>;

  void FindMatchingPrefixSuffixPosParallel(SizeTSlice prefix_positions, SizeTSlice suffix_positions, SizeTSlice matching_suffix_pos, size_t suffix_base_pos) {
    size_t prefix_n = prefix_positions.size();
    size_t suffix_n = suffix_positions.size();
    if (prefix_n + suffix_n <= parallel_block_size) {
      FindMatchingPrefixSuffixPosSerial(prefix_positions, suffix_positions, matching_suffix_pos, suffix_base_pos);
      return;
    }

    size_t prefix_mid = prefix_n >> 1;
    size_t l = 0, r = suffix_n;
    // Upper bound of the matching range; it may be empty. Recursive calls
    // use suffix_base_pos to retain a previously established match.
    while (l < r) {
      size_t mid = (l + r) >> 1;
      if (ValidLR_RPos(prefix_positions[prefix_mid], suffix_positions[mid])) {
        l = mid + 1;
      } else {
        r = mid;
      }
    }

    matching_suffix_pos[prefix_mid] =
        l + suffix_base_pos == 0 ? std::numeric_limits<size_t>::max()
                                : l + suffix_base_pos - 1;

    parlay::par_do(
        [&]() {
          if (prefix_mid > 0) {
            FindMatchingPrefixSuffixPosParallel(prefix_positions.cut(0, prefix_mid), suffix_positions.cut(0, l), matching_suffix_pos.cut(0, prefix_mid), suffix_base_pos);
          }
        },
        [&]() {
          if (prefix_mid + 1 < prefix_n) {
            FindMatchingPrefixSuffixPosParallel(prefix_positions.cut(prefix_mid + 1, prefix_n), suffix_positions.cut(l, suffix_n), matching_suffix_pos.cut(prefix_mid + 1, prefix_n), suffix_base_pos + l);
          }
        });
  }

  void FindMatchingPrefixSuffixPosSerial(SizeTSlice prefix_positions, SizeTSlice suffix_positions, SizeTSlice matching_suffix_pos, size_t suffix_base_pos) {
    size_t prefix_n = prefix_positions.size();
    size_t suffix_n = suffix_positions.size();
    for (size_t i = 0, j = 0; i < prefix_n; i++) {
      while (j < suffix_n && ValidLR_RPos(prefix_positions[i], suffix_positions[j])) {
        j++;
      }
      matching_suffix_pos[i] =
          j + suffix_base_pos == 0 ? std::numeric_limits<size_t>::max()
                                  : j + suffix_base_pos - 1;
    }
  }

  auto FindMatchingPrefixSuffixPos(SizeTSlice prefix_pos, SizeTSlice suffix_pos) {
    size_t prefix_n = prefix_pos.size();
    parlay::sequence<size_t> matching_suffix_pos(prefix_n);
    FindMatchingPrefixSuffixPosParallel(
        prefix_pos, suffix_pos, parlay::make_slice(matching_suffix_pos), 0);

    return matching_suffix_pos;
  }

  void BuildRPosInternal(uint8_t min_X, SizeTSlice prefix_pos, SizeTSlice suffix_pos) {
    auto matching_suffix_pos = FindMatchingPrefixSuffixPos(prefix_pos, suffix_pos);
    parlay::parallel_for(0, prefix_pos.size(), [&](size_t i) {
      const size_t match = matching_suffix_pos[i];
      if (match != std::numeric_limits<size_t>::max()) {
        LinkRPos(prefix_pos[i], suffix_pos[match], min_X);
      }
    });
  }

  // ==================== Build Prefix/Suffix ====================

  inline auto BuildPrefixSuffixParallel(size_t ll, size_t rr) {
    size_t block_size = parlay::internal::_block_size;
    size_t nn = rr - ll;
    size_t block_count = parlay::internal::num_blocks(nn, block_size);

    parlay::sequence<int64_t> l_sketch(block_count);
    parlay::sequence<int64_t> r_sketch(block_count);
    parlay::sequence<int64_t> ascending_l_count(block_count);
    parlay::sequence<int64_t> ascending_r_count(block_count);

    parlay::internal::sliced_for(
        nn, block_size, [&](size_t i, size_t s, size_t e) {
          s += ll;
          e += ll;
          int64_t l_val = std::numeric_limits<int64_t>::min();
          int64_t r_val = std::numeric_limits<int64_t>::max();
          for (size_t j = s; j < e; j++) {
            l_val = std::max(l_val, L[j]);
            r_val = std::min(r_val, R[j]);
          }
          l_sketch[i] = l_val;
          r_sketch[i] = r_val;
        });

    for (size_t i = 1; i < block_count; i++) {
      l_sketch[i] = std::max(l_sketch[i], l_sketch[i - 1]);
    }
    for (int64_t i = block_count - 2; i >= 0; i--) {
      r_sketch[i] = std::min(r_sketch[i], r_sketch[i + 1]);
    }

    parlay::internal::sliced_for(
        nn, block_size, [&](size_t i, size_t s, size_t e) {
          s += ll;
          e += ll;
          int64_t ascending_l = 0;
          int64_t ascending_r = 0;

          int64_t l_val =
              (i > 0) ? l_sketch[i - 1] : std::numeric_limits<int64_t>::min();
          int64_t r_val = (i < block_count - 1)
                              ? r_sketch[i + 1]
                              : std::numeric_limits<int64_t>::max();

          for (size_t j = s; j < e; j++) {
            if (L[j] > l_val) {
              ascending_l++;
              l_val = L[j];
            }
          }

          for (int64_t j = (int64_t)e - 1; j >= (int64_t)s; j--) {
            if (R[j] < r_val) {
              ascending_r++;
              r_val = R[j];
            }
          }

          ascending_l_count[i] = ascending_l;
          ascending_r_count[i] = ascending_r;
        });

    for (size_t i = 1; i < block_count; i++) {
      ascending_l_count[i] = ascending_l_count[i - 1] + ascending_l_count[i];
      ascending_r_count[i] = ascending_r_count[i - 1] + ascending_r_count[i];
    }

    size_t prefix_count = ascending_l_count[block_count - 1];
    size_t suffix_count = ascending_r_count[block_count - 1];
    parlay::sequence<size_t> prefix_pos(prefix_count);
    parlay::sequence<size_t> suffix_pos(suffix_count);

    parlay::internal::sliced_for(
        nn, block_size, [&](size_t i, size_t s, size_t e) {
          s += ll;
          e += ll;
          int64_t ascending_l = (i == 0) ? 0 : ascending_l_count[i - 1];
          int64_t ascending_r = ascending_r_count[i];

          int64_t l_val = std::numeric_limits<int64_t>::min();
          int64_t r_val = std::numeric_limits<int64_t>::max();
          if (i > 0) {
            l_val = std::max(l_val, l_sketch[i - 1]);
          }
          if (i < block_count - 1) {
            r_val = std::min(r_val, r_sketch[i + 1]);
          }
          for (size_t j = s; j < e; j++) {
            if (L[j] > l_val) {
              prefix_pos[ascending_l++] = j;
              l_val = L[j];
            }
          }

          for (int64_t j = (int64_t)e - 1; j >= (int64_t)s; j--) {
            if (R[j] < r_val) {
              suffix_pos[--ascending_r] = j;
              r_val = R[j];
            }
          }
        });

    return std::make_tuple(prefix_count, prefix_pos, suffix_count, suffix_pos);
  }

  std::pair<parlay::sequence<size_t>, parlay::sequence<size_t>> BuildPrefixSuffix(size_t ll, size_t rr) {
    auto ret = BuildPrefixSuffixParallel(ll, rr);
    return {std::move(std::get<1>(ret)), std::move(std::get<3>(ret))};
  }

  // ==================== Build RPos Prefix Max ====================

  void BuildRPosPrefixMaxParallel(size_t l, size_t r, size_t* r_pos) {
    auto slice = parlay::make_slice(r_pos + l, r_pos + r);
    parlay::scan_inclusive_inplace(slice, parlay::maximum<size_t>());
  }

  void BuildRPosPrefixMax(size_t l, size_t r, size_t* r_pos) {
    BuildRPosPrefixMaxParallel(l, r, r_pos);
  }

  // ==================== Build L/R and MinX ====================

  // Compute L and R arrays for the given range. For the provisional finger
  // density rho = p/q, the boundary coordinate is
  // key / rho - i = (key * q - i * p) / p. Because p is positive, we store
  // the numerator key * q - i * p; removing the common denominator preserves
  // all subsequent comparisons.
  uint8_t BuildLRMinX(size_t l, size_t l_covering_end, size_t r_covering_end) {
    uint8_t min_X = *parlay::min_element(parlay::make_slice(&X[l], &X[r_covering_end]));
    assert((1ull << min_X) >= 2);

    const uint64_t numerator = finger_density.numerator();
    const uint64_t denominator = finger_density.denominator();
    constexpr uint64_t kInt64Max =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    constexpr uint64_t kInt64Magnitude = kInt64Max + 1;

    // Check the extrema once so the per-entry loop can use signed 64-bit
    // arithmetic without overflow checks. The largest key coordinate occurs
    // at the end of this sorted range. The smallest possible coordinate is
    // -2, so reserve two denominators below zero as well.
    const uint64_t max_scaled_key =
        (keys[r_covering_end - 1] >> min_X) + 1;
    if (numerator > kInt64Max || denominator > kInt64Max ||
        max_scaled_key > kInt64Max / denominator ||
        denominator > kInt64Magnitude / 2 ||
        n > (kInt64Magnitude - 2 * denominator) / numerator) {
      std::abort();
    }

    const int64_t signed_numerator = static_cast<int64_t>(numerator);
    const int64_t signed_denominator = static_cast<int64_t>(denominator);
    auto ScaleForDensity = [&](int64_t key, size_t i) -> int64_t {
      return key * signed_denominator -
             static_cast<int64_t>(i) * signed_numerator;
    };

    parlay::parallel_for(l_covering_end, r_covering_end, [&](size_t i) {
      const int64_t scaled_key =
          static_cast<int64_t>(keys[i] >> min_X);
      L[i] = ScaleForDensity(scaled_key - 2, i - B);
      R[i] = ScaleForDensity(scaled_key + 1, i + B + 1);
    });

    return min_X;
  }

  // ==================== Build RPos Recursive ====================

  // build r_pos in subrange of range [l, r)
  inline void BuildRPosRecursive(size_t l, size_t r) {
    if (r <= r_pos[l]) {
      return;
    }

    size_t l_covering_end = l + B;
    size_t r_covering_end = r - B;

    uint8_t min_X = BuildLRMinX(l, l_covering_end, r_covering_end);

    auto [prefix_pos, suffix_pos] = BuildPrefixSuffix(l_covering_end, r_covering_end);

    BuildRPosInternal(min_X, parlay::make_slice(prefix_pos), parlay::make_slice(suffix_pos));

    BuildRPosPrefixMax(l, r_covering_end, r_pos);

    // Build Child
    size_t nn = r_covering_end - l;
    auto is_min_X = [&](size_t i) -> bool { return X[l + i] == min_X; };
    auto min_X_pos = parlay::pack(
        parlay::delayed_tabulate(nn, [&](size_t i) { return l + i; }),
        parlay::delayed_tabulate(nn, is_min_X));

    size_t min_X_count = min_X_pos.size();
    size_t sub_array_count = min_X_count + 1;

    // sub_array_l[i], sub_array_r[i]: range of i-th sub-array, will not cover min_X_pos[i] or min_X_pos[i - 1]
    auto sub_array_l = [&](size_t i) -> size_t {
      return (i == 0) ? l : min_X_pos[i - 1] + 1;
    };

    auto sub_array_r = [&](size_t i) -> size_t {
      return (i == sub_array_count - 1) ? r : min_X_pos[i] + B;
    };

    parlay::parallel_for(0, sub_array_count, [&](size_t i) {
      size_t sub_l = sub_array_l(i);
      size_t sub_r = sub_array_r(i);
      assert(sub_l > l || sub_r < r);
      assert(sub_l >= l && sub_r <= r);
      if (sub_r > r_pos[sub_l]) {
        BuildRPosRecursive(sub_l, sub_r);
      }
    });
  }

  // ==================== Main Entry Point ====================

  void BuildRPos() {
    parlay::parallel_for(0, n,
                         [&](size_t i) { r_pos[i] = std::min(i + B * 2, n); });

    // Early exit if entire range [0, n) is already covered by initial r_pos[0]
    // This handles n <= 2*B case where no recursive building is needed
    if (n <= r_pos[0]) {
      X_of_start[0] = 0;  // Default value for small inputs
      return;
    }

    BuildRPosRecursive(0, n);
  }

  void Build() {
    BuildX();
    BuildRPos();
    MaximalCoveringFromRPos();
  }

  // ==================== Overlapping Segmenting ====================

  void MaximalCoveringFromRPos() {
    auto is_r_pos_start = parlay::delayed_tabulate(
        n, [&](size_t i) { return (i == 0) || (r_pos[i] != r_pos[i - 1]); });

    parlay::sequence<size_t> important_pos = parlay::pack_index(is_r_pos_start);
    size_t segment_count = important_pos.size();

    segments = parlay::tabulate<Segment>(segment_count, [&](size_t i) {
      size_t l = important_pos[i];
      size_t r = r_pos[l];
      assert(r > l);
      return Segment{l, r};
    });

  }

  // ==================== Verification ====================

  size_t GetAPCutCount(size_t l, size_t r, size_t X) const {
    assert(l < r && r <= n);
    
    if (r <= l + B) {
      return 0;
    } else if (r <= l + 2 * B) {
      return 1;
    } else {
      size_t l_pos = l + B;
      size_t r_pos_local = r - B - 1;

      size_t l_num = keys[l_pos] >> X;
      size_t r_num = keys[r_pos_local] >> X;
      return r_num - l_num + 2;
    }
  }

  bool ValidLR_Segment(size_t l, size_t r, size_t X) const {
    const size_t required_finger_count = GetAPCutCount(l, r, X) + 1;
    return required_finger_count <=
           finger_density.FloorMultiply(r - l);
  }

  static constexpr size_t parallel_block_size = 2000;

};  // class MaximalCoveringBuilder

}  // namespace segmenting
}  // namespace internal
}  // namespace fh_index

#endif  // FH_INDEX_INTERNAL_CONSTRUCTION_MAXIMAL_COVERING_BUILDER_H
