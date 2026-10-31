#include <fstream>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

#include "common_internal/construction/fraction.h"
#include "common_internal/construction/maximal_covering_builder.h"
#include "common_internal/utility/math.h"
#include "parlay/parallel.h"
#include "util/dataset_generator.h"
#include "timer.hpp"

using namespace std;
using namespace fh_index::internal::segmenting;
using fh_index::internal::Fraction;
using fh_index::util::Distribution;
using fh_index::util::DistributionName;
using fh_index::util::GenerateSortedKeys;

#ifndef TEST_B
#define TEST_B 48
#endif

#ifndef TEST_F
#define TEST_F 8
#endif

static const Fraction TEST_FINGER_DENSITY(1, TEST_F);

// Test result structure
struct TestResult {
  bool passed;
  string message;

  TestResult(bool p, const string& msg) : passed(p), message(msg) {}
};

// Utility function to print test result
void PrintTestResult(const string& test_name, const TestResult& result) {
  cout << "[" << (result.passed ? "PASS" : "FAIL") << "] " << test_name;
  if (!result.message.empty()) {
    cout << ": " << result.message;
  }
  cout << endl;
}

// Compare both matching paths with an independent upper-bound search. Some
// prefixes deliberately have no matching suffix; others match every suffix.
TestResult TestPrefixSuffixMatching() {
  for (size_t n : {size_t(1), size_t(32), size_t(16384)}) {
    auto keys = parlay::tabulate(n, [](size_t i) { return uint64_t(i); });
    auto positions = parlay::tabulate(n, [](size_t i) { return i; });
    MaximalCoveringBuilder<uint64_t> builder(n, keys.data(), 2, Fraction(1, 2));
    for (int pattern = 0; pattern < 3; ++pattern) {
      for (size_t i = 0; i < n; ++i) {
        builder.L[i] = pattern == 0 ? static_cast<int64_t>(2 * i) - 4
                                    : (pattern == 1 ? -1 : 100000);
        builder.R[i] = static_cast<int64_t>(3 * i) + 1;
      }
      auto matches = builder.FindMatchingPrefixSuffixPos(
          parlay::make_slice(positions), parlay::make_slice(positions));
      for (size_t i = 0; i < n; ++i) {
        const auto end = std::upper_bound(builder.R, builder.R + n, builder.L[i]);
        const size_t expected = end == builder.R
            ? std::numeric_limits<size_t>::max()
            : static_cast<size_t>(end - builder.R - 1);
        if (matches[i] != expected) {
          return TestResult(false, "Prefix/suffix match differs from upper_bound");
        }
      }
    }
  }
  return TestResult(true, "Empty, partial, and full matching ranges agree");
}

// Test 1: Verify BuildX constructs valid X values
TestResult TestBuildX(MaximalCoveringBuilder<uint64_t>& builder) {
  builder.BuildX();

  size_t n = builder.n;
  size_t invalid_count = 0;

  for (size_t i = 0; i < n; i++) {
    if (builder.X[i] >= kMaxX) {
      invalid_count++;
      if (invalid_count <= 5) {
        cout << "  X[" << i << "] = " << (int)builder.X[i]
             << " exceeds max value" << endl;
      }
    }
  }

  if (invalid_count > 0) {
    return TestResult(false,
        "Found " + to_string(invalid_count) + " invalid X values");
  }

  return TestResult(true, "All X values are valid");
}

// Test 2: Verify BuildRPos constructs valid r_pos values
TestResult TestBuildRPos(MaximalCoveringBuilder<uint64_t>& builder) {
  builder.BuildX();
  builder.BuildRPos();

  size_t n = builder.n;

  // r_pos should be monotonically non-decreasing
  size_t violation_count = 0;
  for (size_t i = 1; i < n; i++) {
    if (builder.r_pos[i] < builder.r_pos[i - 1]) {
      violation_count++;
      if (violation_count <= 5) {
        cout << "  r_pos[" << i << "] = " << builder.r_pos[i]
             << " < r_pos[" << i - 1 << "] = " << builder.r_pos[i - 1] << endl;
      }
    }
  }

  if (violation_count > 0) {
    return TestResult(false,
        "Found " + to_string(violation_count) + " monotonicity violations in r_pos");
  }

  // r_pos[i] should be >= min(i + B, n) (minimum segment size, capped at n)
  size_t min_violation_count = 0;
  for (size_t i = 0; i < n; i++) {
    size_t min_r = std::min(i + builder.B, n);
    if (builder.r_pos[i] < min_r) {
      min_violation_count++;
      if (min_violation_count <= 5) {
        cout << "  r_pos[" << i << "] = " << builder.r_pos[i]
             << " < min(i + B, n) = " << min_r << endl;
      }
    }
  }

  if (min_violation_count > 0) {
    return TestResult(false,
        "Found " + to_string(min_violation_count) + " minimum size violations in r_pos");
  }

  return TestResult(true, "r_pos values are valid");
}

// Test 3: Verify BuildRPosPrefixMax (parallel vs serial)
TestResult TestBuildRPosPrefixMax(MaximalCoveringBuilder<uint64_t>& builder) {
  builder.BuildX();

  // Initialize r_pos with some test values
  size_t n = builder.n;
  parlay::parallel_for(0, n, [&](size_t i) {
    builder.r_pos[i] = std::min(i + builder.B * 2, n);
  });

  // Make a copy for serial test
  std::vector<size_t> r_pos_serial(n);
  for (size_t i = 0; i < n; i++) {
    r_pos_serial[i] = builder.r_pos[i];
  }

  // Run parallel version
  builder.BuildRPosPrefixMax(0, n, builder.r_pos);

  // Run serial version on copy
  for (size_t i = 1; i < n; i++) {
    r_pos_serial[i] = std::max(r_pos_serial[i - 1], r_pos_serial[i]);
  }

  // Compare results
  size_t mismatch_count = 0;
  for (size_t i = 0; i < n; i++) {
    if (builder.r_pos[i] != r_pos_serial[i]) {
      mismatch_count++;
      if (mismatch_count <= 5) {
        cout << "  r_pos[" << i << "]: parallel=" << builder.r_pos[i]
             << ", serial=" << r_pos_serial[i] << endl;
      }
    }
  }

  if (mismatch_count > 0) {
    return TestResult(false,
        "Found " + to_string(mismatch_count) + " mismatches between parallel and serial");
  }

  return TestResult(true, "BuildRPosPrefixMax parallel matches serial");
}

// Test 4: Full Build test
TestResult TestFullBuild(MaximalCoveringBuilder<uint64_t>& builder) {
  // Build without internal verification (which may have strict requirements)
  builder.BuildX();
  builder.BuildRPos();
  builder.MaximalCoveringFromRPos();

  // Verify segments are valid
  if (builder.segments.empty()) {
    return TestResult(false, "No segments generated");
  }

  size_t segment_count = builder.segments.size();

  // Check segment properties
  size_t invalid_segments = 0;
  for (size_t i = 0; i < segment_count; i++) {
    const auto& seg = builder.segments[i];

    // l should be < r
    if (seg.l >= seg.r) {
      invalid_segments++;
      if (invalid_segments <= 5) {
        cout << "  Segment " << i << ": l=" << seg.l << " >= r=" << seg.r << endl;
      }
    }

    // Segment size should be at least B
    if (seg.r - seg.l < builder.B) {
      invalid_segments++;
      if (invalid_segments <= 5) {
        cout << "  Segment " << i << ": size=" << (seg.r - seg.l)
             << " < B=" << builder.B << endl;
      }
    }
  }

  if (invalid_segments > 0) {
    return TestResult(false,
        "Found " + to_string(invalid_segments) + " invalid segments");
  }

  // Check segments cover the full range and overlap properly
  // First segment should start at 0
  if (builder.segments[0].l != 0) {
    return TestResult(false,
        "First segment doesn't start at 0, starts at " + to_string(builder.segments[0].l));
  }

  // Last segment should end at n
  if (builder.segments[segment_count - 1].r != builder.n) {
    return TestResult(false,
        "Last segment doesn't end at n=" + to_string(builder.n) +
        ", ends at " + to_string(builder.segments[segment_count - 1].r));
  }

  // Consecutive segments should overlap
  size_t gap_count = 0;
  for (size_t i = 1; i < segment_count; i++) {
    if (builder.segments[i].l > builder.segments[i - 1].r) {
      gap_count++;
      if (gap_count <= 5) {
        cout << "  Gap between segment " << (i - 1) << " (r=" << builder.segments[i - 1].r
             << ") and segment " << i << " (l=" << builder.segments[i].l << ")" << endl;
      }
    }
  }

  if (gap_count > 0) {
    return TestResult(false,
        "Found " + to_string(gap_count) + " gaps between consecutive segments");
  }

  return TestResult(true,
      "Build completed with " + to_string(segment_count) + " valid segments");
}

// Run all tests for a given distribution
void RunTestsForDistribution(size_t n, Distribution dist, int& total_tests, int& passed_tests) {
  cout << "\n--- Distribution: " << DistributionName(dist) << " ---" << endl;

  auto keys = GenerateSortedKeys(n, dist);

  // Test 1: BuildX
  {
    MaximalCoveringBuilder<uint64_t> builder(
        n, keys.data(), TEST_B, TEST_FINGER_DENSITY);
    auto result = TestBuildX(builder);
    PrintTestResult("TestBuildX", result);
    total_tests++;
    if (result.passed) passed_tests++;
  }

  // Test 2: BuildRPos
  {
    MaximalCoveringBuilder<uint64_t> builder(
        n, keys.data(), TEST_B, TEST_FINGER_DENSITY);
    auto result = TestBuildRPos(builder);
    PrintTestResult("TestBuildRPos", result);
    total_tests++;
    if (result.passed) passed_tests++;
  }

  // Test 3: BuildRPosPrefixMax
  {
    MaximalCoveringBuilder<uint64_t> builder(
        n, keys.data(), TEST_B, TEST_FINGER_DENSITY);
    auto result = TestBuildRPosPrefixMax(builder);
    PrintTestResult("TestBuildRPosPrefixMax", result);
    total_tests++;
    if (result.passed) passed_tests++;
  }

  // Test 4: Full Build
  {
    MaximalCoveringBuilder<uint64_t> builder(
        n, keys.data(), TEST_B, TEST_FINGER_DENSITY);
    auto result = TestFullBuild(builder);
    PrintTestResult("TestFullBuild", result);
    total_tests++;
    if (result.passed) passed_tests++;
  }
}

int main(int argc, char* argv[]) {
  size_t n = 1000000;  // Default test size

  if (argc > 1) {
    n = std::stoull(argv[1]);
  }

  cout << "=== MaximalCoveringBuilder Tests ===" << endl;
  cout << "n = " << n << ", B = " << TEST_B << ", F = " << TEST_F << endl;

  int total_tests = 0;
  int passed_tests = 0;

  const auto matching_result = TestPrefixSuffixMatching();
  PrintTestResult("TestPrefixSuffixMatching", matching_result);
  ++total_tests;
  if (matching_result.passed) ++passed_tests;

  // This valid small-bucket input previously read suffix_positions[-1].
  auto small_keys = parlay::tabulate(20, [](size_t i) { return uint64_t(i * i + 2); });
  MaximalCoveringBuilder<uint64_t> small_builder(
      small_keys.size(), small_keys.data(), 2, Fraction(1, 2));
  const auto small_result = TestFullBuild(small_builder);
  PrintTestResult("SmallBucketConstruction", small_result);
  ++total_tests;
  if (small_result.passed) ++passed_tests;

  // Test all distributions
  vector<Distribution> distributions = {
    Distribution::UNIFORM,
    Distribution::EXPONENTIAL,
    Distribution::NORMAL,
    Distribution::CLUSTERED,
    Distribution::DENSE,
    Distribution::SPARSE,
    Distribution::MIXED
  };

  for (Distribution dist : distributions) {
    RunTestsForDistribution(n, dist, total_tests, passed_tests);
  }

  cout << endl;
  cout << "=== Summary ===" << endl;
  cout << "Passed: " << passed_tests << "/" << total_tests << endl;
  cout << "Distributions tested: " << distributions.size() << endl;

  return (passed_tests == total_tests) ? 0 : 1;
}
