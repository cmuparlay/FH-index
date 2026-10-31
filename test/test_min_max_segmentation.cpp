#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "common_internal/construction/batch_successor.h"
#include "common_internal/construction/min_max_segmentation.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"

using namespace std;
using namespace fh_index::internal::segmenting;

// Test result structure
struct TestResult {
  bool passed;
  string message;

  TestResult(bool p, const string& msg) : passed(p), message(msg) {}
};

void PrintTestResult(const string& test_name, const TestResult& result) {
  cout << "[" << (result.passed ? "PASS" : "FAIL") << "] " << test_name;
  if (!result.message.empty()) {
    cout << ": " << result.message;
  }
  cout << endl;
}

// Distribution types for key generation
enum class Distribution {
  UNIFORM,
  EXPONENTIAL,
  CLUSTERED,
  DENSE,
  SPARSE
};

const char* DistributionName(Distribution dist) {
  switch (dist) {
    case Distribution::UNIFORM: return "UNIFORM";
    case Distribution::EXPONENTIAL: return "EXPONENTIAL";
    case Distribution::CLUSTERED: return "CLUSTERED";
    case Distribution::DENSE: return "DENSE";
    case Distribution::SPARSE: return "SPARSE";
    default: return "UNKNOWN";
  }
}

// Generate sorted random keys
uint64_t* GenerateSortedKeys(size_t n, Distribution dist, uint64_t seed = 42) {
  constexpr size_t kAlignment = 64;
  const size_t bytes = sizeof(uint64_t) * n;
  const size_t aligned_bytes =
      ((bytes + kAlignment - 1) / kAlignment) * kAlignment;
  uint64_t* keys =
      static_cast<uint64_t*>(aligned_alloc(kAlignment, aligned_bytes));
  assert(keys != nullptr);
  std::mt19937_64 rng(seed);

  keys[0] = 1000;

  switch (dist) {
    case Distribution::UNIFORM: {
      std::uniform_int_distribution<uint64_t> gap_dist(1, 1000);
      for (size_t i = 1; i < n; i++) {
        keys[i] = keys[i - 1] + gap_dist(rng);
      }
      break;
    }

    case Distribution::EXPONENTIAL: {
      std::exponential_distribution<double> exp_dist(0.01);
      for (size_t i = 1; i < n; i++) {
        uint64_t gap = std::max((uint64_t)1, (uint64_t)(exp_dist(rng) * 100));
        keys[i] = keys[i - 1] + gap;
      }
      break;
    }

    case Distribution::CLUSTERED: {
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
      std::uniform_int_distribution<uint64_t> gap_dist(1, 10);
      for (size_t i = 1; i < n; i++) {
        keys[i] = keys[i - 1] + gap_dist(rng);
      }
      break;
    }

    case Distribution::SPARSE: {
      std::uniform_int_distribution<uint64_t> gap_dist(10000, 1000000);
      for (size_t i = 1; i < n; i++) {
        keys[i] = keys[i - 1] + gap_dist(rng);
      }
      break;
    }
  }

  return keys;
}

// Verify that the result is correct
TestResult VerifyResult(size_t n, const uint64_t* keys, size_t k,
                        const MinMaxSegmentationResult<uint64_t>& result) {
  if (n == 0) {
    if (result.max_segment_size == 0) {
      return TestResult(true, "Empty input handled correctly");
    }
    return TestResult(false, "Empty input should have max_segment_size = 0");
  }

  if (result.num_cuts > k) {
    return TestResult(false, "Used more cuts than allowed: " +
                      to_string(result.num_cuts) + " > " + to_string(k));
  }

  // Precompute cut points
  vector<uint64_t> cut_points(result.num_cuts);
  for (size_t j = 0; j < result.num_cuts; j++) {
    cut_points[j] = result.GetCutPoint(j);
  }

  // Use cuts as queries, keys as sorted array - more efficient
  // lower_bound gives the first key >= cut, which is the segment boundary
  vector<size_t> boundaries(result.num_cuts);
  BatchSuccessor(n, keys, result.num_cuts, cut_points.data(), boundaries.data());

  // Compute segment sizes in parallel
  // boundaries[i] is the start of segment i+1
  // Segment sizes: boundaries[0], boundaries[1]-boundaries[0], ..., n-boundaries[num_cuts-1]
  auto segment_sizes = parlay::tabulate(result.num_cuts + 1, [&](size_t i) {
    if (i == 0) return boundaries[0];
    if (i == result.num_cuts) return n - boundaries[i - 1];
    return boundaries[i] - boundaries[i - 1];
  });

  size_t actual_max = parlay::reduce(segment_sizes, parlay::maximum<size_t>());

  if (actual_max != result.max_segment_size) {
    return TestResult(false, "Reported max_segment_size=" +
                      to_string(result.max_segment_size) +
                      " but actual=" + to_string(actual_max));
  }

  // Verify theoretical lower bound
  size_t theoretical_lower = (n + k) / (k + 1);
  if (result.max_segment_size < theoretical_lower) {
    return TestResult(false, "Result violates lower bound: " +
                      to_string(result.max_segment_size) + " < " +
                      to_string(theoretical_lower));
  }

  return TestResult(true, "max_segment=" + to_string(result.max_segment_size) +
                    ", x=" + to_string((int)result.x) +
                    ", t=" + to_string(result.t) +
                    ", cuts=" + to_string(result.num_cuts));
}

// Test basic functionality
TestResult TestBasic() {
  // Simple case: 10 elements, k=2
  uint64_t keys[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  size_t n = 10;
  size_t k = 2;

  auto result = SolveMinMaxSegmentation(n, keys, k, n);

  if (result.max_segment_size > n) {
    return TestResult(false, "max_segment_size exceeds n");
  }

  // With 2 cuts, we should get at most ceil(10/3) = 4 in the worst segment
  size_t expected_upper = (n + k) / (k + 1) + 1;
  if (result.max_segment_size > expected_upper) {
    return TestResult(false, "Result too large: " +
                      to_string(result.max_segment_size));
  }

  return VerifyResult(n, keys, k, result);
}

// Test edge cases
TestResult TestEmptyInput() {
  auto result = SolveMinMaxSegmentation<uint64_t>(0, nullptr, 10, 100);
  if (result.max_segment_size == 0 && result.num_cuts == 0) {
    return TestResult(true, "Empty input handled correctly");
  }
  return TestResult(false, "Empty input not handled correctly");
}

TestResult TestSingleElement() {
  uint64_t keys[] = {42};
  auto result = SolveMinMaxSegmentation(1, keys, 10, 1);
  if (result.max_segment_size == 1) {
    return TestResult(true, "Single element handled correctly");
  }
  return TestResult(false, "Single element should have max_segment_size=1");
}

TestResult TestZeroCuts() {
  uint64_t keys[] = {10, 20, 30, 40, 50};
  auto result = SolveMinMaxSegmentation(5, keys, 0, 5);
  if (result.max_segment_size == 5 && result.num_cuts == 0) {
    return TestResult(true, "Zero cuts handled correctly");
  }
  return TestResult(false, "Zero cuts: expected max=5, cuts=0");
}

// Test with various distributions
TestResult TestDistribution(size_t n, size_t k, Distribution dist) {
  uint64_t* keys = GenerateSortedKeys(n, dist);
  auto result = SolveMinMaxSegmentation(n, keys, k, n);
  auto test_result = VerifyResult(n, keys, k, result);
  free(keys);
  return test_result;
}

// Performance test
void RunPerformanceTest(size_t n, size_t k) {
  cout << "\n=== Performance Test ===" << endl;
  cout << "n = " << n << ", k = " << k << endl;

  uint64_t* keys = GenerateSortedKeys(n, Distribution::UNIFORM);

  auto start = chrono::high_resolution_clock::now();

  auto result = SolveMinMaxSegmentation(n, keys, k, n);

  auto end = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

  cout << "Time: " << duration.count() << " ms" << endl;
  cout << "Result: max_segment=" << result.max_segment_size
       << ", x=" << (int)result.x
       << ", t=" << result.t
       << ", cuts=" << result.num_cuts << endl;

  // Theoretical bound
  size_t lower_bound = (n + k) / (k + 1);
  double ratio = (double)result.max_segment_size / lower_bound;
  cout << "Theoretical lower bound: " << lower_bound << endl;
  cout << "Ratio to lower bound: " << ratio << endl;

  free(keys);
}

// Brute force solution for small inputs (for comparison)
size_t BruteForceSolve(size_t n, const uint64_t* keys, size_t k) {
  if (n == 0) return 0;
  if (k == 0) return n;

  size_t best_max = n;
  uint64_t max_key = keys[n - 1];

  // Try all x values
  for (uint8_t x = 0; x < 64; x++) {
    uint64_t bucket_size = 1ULL << x;
    if (bucket_size > max_key) break;

    uint64_t max_t = (max_key >> x) + 1;

    // Try all starting positions t
    for (uint64_t t = 0; t <= max_t; t++) {
      // Try all number of cuts from 1 to k
      for (size_t num_cuts = 1; num_cuts <= k; num_cuts++) {
        // Count segment sizes
        vector<size_t> segment_counts(num_cuts + 1, 0);

        for (size_t i = 0; i < n; i++) {
          uint64_t key = keys[i];
          size_t segment = 0;

          for (size_t j = 0; j < num_cuts; j++) {
            uint64_t cut = (t + j) << x;
            if (key >= cut) {
              segment = j + 1;
            }
          }
          segment_counts[segment]++;
        }

        size_t max_seg = *max_element(segment_counts.begin(), segment_counts.end());
        best_max = min(best_max, max_seg);
      }
    }
  }

  return best_max;
}

TestResult TestAgainstBruteForce(size_t n, size_t k) {
  uint64_t* keys = GenerateSortedKeys(n, Distribution::UNIFORM, 12345);

  auto result = SolveMinMaxSegmentation(n, keys, k, n);
  size_t brute_force_result = BruteForceSolve(n, keys, k);

  free(keys);

  if (result.max_segment_size <= brute_force_result) {
    return TestResult(true, "Algorithm: " + to_string(result.max_segment_size) +
                      ", BruteForce: " + to_string(brute_force_result));
  }

  return TestResult(false, "Algorithm: " + to_string(result.max_segment_size) +
                    " > BruteForce: " + to_string(brute_force_result));
}

int main(int argc, char* argv[]) {
  size_t n = 10000;      // 10K
  size_t k = n / 10;     // 1K

  if (argc > 1) n = stoull(argv[1]);
  if (argc > 2) k = stoull(argv[2]);

  cout << "=== MinMaxSegmentation Tests ===" << endl;

  int total_tests = 0;
  int passed_tests = 0;

  // Basic tests
  {
    auto result = TestBasic();
    PrintTestResult("TestBasic", result);
    total_tests++;
    if (result.passed) passed_tests++;
  }

  {
    auto result = TestEmptyInput();
    PrintTestResult("TestEmptyInput", result);
    total_tests++;
    if (result.passed) passed_tests++;
  }

  {
    auto result = TestSingleElement();
    PrintTestResult("TestSingleElement", result);
    total_tests++;
    if (result.passed) passed_tests++;
  }

  {
    auto result = TestZeroCuts();
    PrintTestResult("TestZeroCuts", result);
    total_tests++;
    if (result.passed) passed_tests++;
  }

  // Test against brute force (small inputs)
  cout << "\n--- Brute Force Comparison (small inputs) ---" << endl;
  for (size_t test_n : {20, 50, 100}) {
    for (size_t test_k : {2, 5, 10}) {
      auto result = TestAgainstBruteForce(test_n, test_k);
      PrintTestResult("BruteForce n=" + to_string(test_n) + " k=" + to_string(test_k), result);
      total_tests++;
      if (result.passed) passed_tests++;
    }
  }

  // Test various distributions
  cout << "\n--- Distribution Tests ---" << endl;
  vector<Distribution> distributions = {
    Distribution::UNIFORM,
    Distribution::EXPONENTIAL,
    Distribution::CLUSTERED,
    Distribution::DENSE,
    Distribution::SPARSE
  };

  for (Distribution dist : distributions) {
    auto result = TestDistribution(n, k, dist);
    PrintTestResult(string("Distribution_") + DistributionName(dist), result);
    total_tests++;
    if (result.passed) passed_tests++;
  }

  // Performance test
  if (n >= 100000) {
    RunPerformanceTest(n, k);
  }

  cout << "\n=== Summary ===" << endl;
  cout << "Passed: " << passed_tests << "/" << total_tests << endl;

  return (passed_tests == total_tests) ? 0 : 1;
}
