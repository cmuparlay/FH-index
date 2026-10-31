#ifndef UTIL_DATASET_GENERATOR_H
#define UTIL_DATASET_GENERATOR_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <vector>

#include "parlay/parallel.h"

namespace fh_index {
namespace util {

// Distribution types for key generation
enum class Distribution {
  UNIFORM,      // Uniform random gaps
  EXPONENTIAL,  // Exponential gaps (many small, few large)
  NORMAL,       // Normal/Gaussian gaps
  CLUSTERED,    // Keys clustered in groups
  DENSE,        // Very small gaps
  SPARSE,       // Very large gaps
  MIXED         // Mix of dense and sparse regions
};

inline const char* DistributionName(Distribution dist) {
  switch (dist) {
    case Distribution::UNIFORM: return "UNIFORM";
    case Distribution::EXPONENTIAL: return "EXPONENTIAL";
    case Distribution::NORMAL: return "NORMAL";
    case Distribution::CLUSTERED: return "CLUSTERED";
    case Distribution::DENSE: return "DENSE";
    case Distribution::SPARSE: return "SPARSE";
    case Distribution::MIXED: return "MIXED";
    default: return "UNKNOWN";
  }
}

// Parse distribution name from string (case-insensitive)
inline Distribution ParseDistribution(const std::string& name) {
  std::string upper = name;
  std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);

  if (upper == "UNIFORM" || upper == "GEN") return Distribution::UNIFORM;
  if (upper == "EXPONENTIAL") return Distribution::EXPONENTIAL;
  if (upper == "NORMAL") return Distribution::NORMAL;
  if (upper == "CLUSTERED" || upper == "GAPPED") return Distribution::CLUSTERED;
  if (upper == "DENSE") return Distribution::DENSE;
  if (upper == "SPARSE") return Distribution::SPARSE;
  if (upper == "MIXED") return Distribution::MIXED;

  // Default to UNIFORM if unknown
  return Distribution::UNIFORM;
}

// Generate sorted random keys with different distributions
inline std::vector<uint64_t> GenerateSortedKeys(
    size_t n, Distribution dist = Distribution::UNIFORM, uint64_t seed = 42) {
  std::vector<uint64_t> keys(n);

  std::mt19937_64 rng(seed);

  keys[0] = 1000;  // Start with some offset

  switch (dist) {
    case Distribution::UNIFORM: {
      // Uniform random gaps between 1 and 1000
      std::uniform_int_distribution<uint64_t> gap_dist(1, 1000);
      for (size_t i = 1; i < n; i++) {
        keys[i] = keys[i - 1] + gap_dist(rng);
      }
      break;
    }

    case Distribution::EXPONENTIAL: {
      // Exponential gaps - many small gaps, few large gaps
      std::exponential_distribution<double> exp_dist(0.01);
      for (size_t i = 1; i < n; i++) {
        uint64_t gap = std::max((uint64_t)1, (uint64_t)(exp_dist(rng) * 100));
        keys[i] = keys[i - 1] + gap;
      }
      break;
    }

    case Distribution::NORMAL: {
      // Normal/Gaussian gaps centered around 500
      std::normal_distribution<double> normal_dist(500.0, 200.0);
      for (size_t i = 1; i < n; i++) {
        uint64_t gap = std::max((uint64_t)1, (uint64_t)std::abs(normal_dist(rng)));
        keys[i] = keys[i - 1] + gap;
      }
      break;
    }

    case Distribution::CLUSTERED: {
      // Keys clustered in groups with large gaps between clusters
      size_t cluster_size = std::max((size_t)100, n / 20);
      std::uniform_int_distribution<uint64_t> small_gap(1, 50);
      std::uniform_int_distribution<uint64_t> large_gap(10000, 100000);

      for (size_t i = 1; i < n; i++) {
        if (i % cluster_size == 0) {
          keys[i] = keys[i - 1] + large_gap(rng);
        } else {
          keys[i] = keys[i - 1] + small_gap(rng);
        }
      }
      break;
    }

    case Distribution::DENSE: {
      // Very small gaps (consecutive or near-consecutive)
      std::uniform_int_distribution<uint64_t> gap_dist(1, 10);
      for (size_t i = 1; i < n; i++) {
        keys[i] = keys[i - 1] + gap_dist(rng);
      }
      break;
    }

    case Distribution::SPARSE: {
      // Very large gaps
      std::uniform_int_distribution<uint64_t> gap_dist(10000, 1000000);
      for (size_t i = 1; i < n; i++) {
        keys[i] = keys[i - 1] + gap_dist(rng);
      }
      break;
    }

    case Distribution::MIXED: {
      // Alternating dense and sparse regions
      std::uniform_int_distribution<uint64_t> small_gap(1, 20);
      std::uniform_int_distribution<uint64_t> large_gap(5000, 50000);
      size_t region_size = std::max((size_t)50, n / 40);

      for (size_t i = 1; i < n; i++) {
        bool is_dense = ((i / region_size) % 2) == 0;
        if (is_dense) {
          keys[i] = keys[i - 1] + small_gap(rng);
        } else {
          keys[i] = keys[i - 1] + large_gap(rng);
        }
      }
      break;
    }
  }

  // Verify keys are sorted and unique
  for (size_t i = 1; i < n; i++) {
    assert(keys[i] > keys[i - 1] && "Keys must be strictly increasing");
  }

  return keys;
}

}  // namespace util
}  // namespace fh_index

#endif  // UTIL_DATASET_GENERATOR_H
