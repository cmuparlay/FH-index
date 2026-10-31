#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <cassert>
#include <random>
#include <utility>
#include "parlay/sequence.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

namespace test_utils {

// Generate n intervals with guaranteed properties:
// - L(i) <= L(i+1) && R(i) <= R(i+1) (monotonically non-decreasing)
// - L(i) < R(i) (valid intervals)
// - L(i+1) <= R(i) (no gaps between consecutive intervals)
//
// Note: This generator produces strictly increasing intervals for robustness,
// which satisfies the weaker non-decreasing requirement.
//
// Uses parallel construction for efficiency.
//
// Parameters:
//   n: Number of intervals to generate
//   seed: Random seed for reproducibility (default: 42)
//   step_min: Minimum step between left endpoints (default: 5)
//   step_max: Maximum step between left endpoints (default: 15)
//   len_min: Minimum interval length (default: 20)
//   len_max: Maximum interval length (default: 24)
//
// CRITICAL CONSTRAINTS (must hold for correctness):
//   1. step_max <= len_min (guarantees no gaps: L(i+1) <= R(i))
//   2. step_min > len_max - len_min (guarantees R(i) < R(i+1))
inline parlay::sequence<std::pair<int, int>> generate_intervals(
    size_t n,
    int seed = 42,
    int step_min = 5,
    int step_max = 15,
    int len_min = 20,
    int len_max = 24) {

  // Validate parameters to guarantee all interval constraints
  assert(step_max <= len_min && "step_max must be <= len_min to prevent gaps");
  assert(step_min > len_max - len_min && "step_min must be > len_max - len_min to guarantee R(i) strictly increasing");

  // Use parlay::random for parallel random number generation with reproducibility
  parlay::random rng(seed);

  // Generate random steps in parallel
  parlay::sequence<int> steps = parlay::tabulate(n, [&](size_t i) {
    auto rand_val = rng.ith_rand(2 * i);
    return static_cast<int>(step_min + (rand_val % (step_max - step_min + 1)));
  });

  // Generate random lengths in parallel
  parlay::sequence<int> lens = parlay::tabulate(n, [&](size_t i) {
    auto rand_val = rng.ith_rand(2 * i + 1);
    return static_cast<int>(len_min + (rand_val % (len_max - len_min + 1)));
  });

  // Compute left endpoints via prefix sum (parallel)
  parlay::sequence<int> lefts = steps;
  parlay::scan_inplace(lefts);

  // Generate right endpoints in parallel
  parlay::sequence<int> rights = parlay::tabulate(n, [&](size_t i) {
    return lefts[i] + lens[i];
  });

  // Combine into interval pairs (parallel)
  parlay::sequence<std::pair<int, int>> intervals = parlay::tabulate(n, [&](size_t i) {
    return std::make_pair(lefts[i], rights[i]);
  });

  return intervals;
}

} // namespace test_utils

#endif // TEST_UTILS_H
