#ifndef FH_INDEX_INTERNAL_CONSTRUCTION_BATCH_SUCCESSOR_H
#define FH_INDEX_INTERNAL_CONSTRUCTION_BATCH_SUCCESSOR_H

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "parlay/parallel.h"
#include "parlay/sequence.h"

namespace fh_index {
namespace internal {
namespace segmenting {

// BatchSuccessor: Find the successor (first element >= query) for each query key
// in a sorted array.
//
// Given:
//   - keys: a sorted array of n elements
//   - queries: a sorted array of m query keys
//
// Returns:
//   - result: for each query[i], result[i] is the smallest index j such that
//     keys[j] >= queries[i]. If no such index exists, result[i] = n.
//
// This is equivalent to std::lower_bound for each query, but optimized for
// batch processing with parallel execution using divide-and-conquer.

template <typename KeyType>
class BatchSuccessorComputer {
 public:
  static constexpr size_t kParallelMergeThreshold = 2000;

  BatchSuccessorComputer(size_t n, const KeyType* keys, size_t m,
                         const KeyType* queries, size_t* result)
      : n_(n), keys_(keys), m_(m), queries_(queries), result_(result) {}

  void Compute() {
    if (m_ == 0) return;
    ComputeCoreParallel(0, m_ - 1, 0, n_);
  }

  void ComputeSerial() {
    if (m_ == 0) return;
    ComputeCoreSerial(0, m_ - 1, 0, n_);
  }

 private:
  size_t n_;
  const KeyType* keys_;
  size_t m_;
  const KeyType* queries_;
  size_t* result_;

  // Serial implementation: linear scan
  // queries[ql..qr] (inclusive), keys in range [kl, kr) (exclusive right)
  void ComputeCoreSerial(size_t ql, size_t qr, size_t kl, size_t kr) {
    size_t key_pos = kl;
    for (size_t i = ql; i <= qr; i++) {
      KeyType query = queries_[i];
      while (key_pos < kr && keys_[key_pos] < query) {
        key_pos++;
      }
      result_[i] = key_pos;
    }
  }

  // Parallel implementation: divide-and-conquer
  // queries[ql..qr] (inclusive), keys in range [kl, kr) (exclusive right)
  void ComputeCoreParallel(size_t ql, size_t qr, size_t kl, size_t kr) {
    // Use serial for small ranges
    if (qr - ql + 1 + kr - kl <= kParallelMergeThreshold) {
      ComputeCoreSerial(ql, qr, kl, kr);
      return;
    }

    // Find midpoint of query range
    size_t qmid = (ql + qr) >> 1;
    KeyType query_mid = queries_[qmid];

    // Binary search for query_mid in keys[kl, kr)
    size_t l = kl, r = kr;
    while (l < r) {
      size_t mid = (l + r) >> 1;
      if (keys_[mid] < query_mid) {
        l = mid + 1;
      } else {
        r = mid;
      }
    }
    result_[qmid] = l;

    // Recursively process left and right halves in parallel
    parlay::par_do(
        [&]() {
          if (ql < qmid) {
            ComputeCoreParallel(ql, qmid - 1, kl, l + 1);
          }
        },
        [&]() {
          if (qmid < qr) {
            ComputeCoreParallel(qmid + 1, qr, l, kr);
          }
        });
  }
};

// Main interface function
template <typename KeyType>
void BatchSuccessor(size_t n, const KeyType* keys, size_t m,
                    const KeyType* queries, size_t* result) {
  BatchSuccessorComputer<KeyType> computer(n, keys, m, queries, result);
  computer.Compute();
}

// Convenience wrapper using parlay sequences
template <typename KeyType>
parlay::sequence<size_t> BatchSuccessor(size_t n, const KeyType* keys,
                                        const parlay::sequence<KeyType>& queries) {
  size_t m = queries.size();
  parlay::sequence<size_t> result(m);
  BatchSuccessor(n, keys, m, queries.data(), result.data());
  return result;
}

// Serial-only version for testing/verification
template <typename KeyType>
void BatchSuccessorSerial(size_t n, const KeyType* keys, size_t m,
                          const KeyType* queries, size_t* result) {
  BatchSuccessorComputer<KeyType> computer(n, keys, m, queries, result);
  computer.ComputeSerial();
}

// Verify correctness of batch successor results
template <typename KeyType>
bool VerifyBatchSuccessor(size_t n, const KeyType* keys, size_t m,
                          const KeyType* queries, const size_t* result) {
  for (size_t i = 0; i < m; i++) {
    size_t pos = result[i];
    KeyType query = queries[i];

    // Check: keys[pos] >= query (or pos == n if no successor exists)
    if (pos < n && keys[pos] < query) {
      return false;
    }

    // Check: pos is the smallest such index
    if (pos > 0 && keys[pos - 1] >= query) {
      return false;
    }

    // Check: result is monotonically non-decreasing for sorted queries
    if (i + 1 < m && queries[i] <= queries[i + 1] && result[i] > result[i + 1]) {
      return false;
    }
  }
  return true;
}

}  // namespace segmenting
}  // namespace internal
}  // namespace fh_index

#endif  // FH_INDEX_INTERNAL_CONSTRUCTION_BATCH_SUCCESSOR_H
