#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "common_internal/construction/fraction.h"
#include "common_internal/construction/single_level_segmentor.h"
#include "common_internal/representation/finger_group_descriptor.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "timer.hpp"
#include "util/io_utility.h"

using namespace std;
using namespace fh_index::internal::segmenting;
using fh_index::internal::Fraction;

#ifndef B
#define B 48
#endif

#ifndef F
#define F 8
#endif

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

// Generate sorted random keys using io_utility
uint64_t* GenerateSortedKeys(size_t n) {
  auto keys_vec = fh_index::util::UniformRandomKeys(n, 0, true);

  constexpr size_t kAlignment = 64;
  const size_t bytes = sizeof(uint64_t) * n;
  const size_t aligned_bytes =
      ((bytes + kAlignment - 1) / kAlignment) * kAlignment;
  uint64_t* keys =
      static_cast<uint64_t*>(aligned_alloc(kAlignment, aligned_bytes));
  assert(keys != nullptr);
  parlay::parallel_for(0, n, [&](size_t i) { keys[i] = keys_vec[i]; });

  return keys;
}

//==============================================================================
// Verification functions - operate on already-run Segmentor
//==============================================================================

// Verify MaximalCovering results
TestResult VerifyMaximalCovering(const Segmentor<uint64_t>& segmentor, size_t B_param) {
  auto& segments = segmentor.segments;
  size_t segment_count = segments.size();
  size_t n = segmentor.n;

  if (segment_count == 0) {
    return TestResult(false, "No segments generated");
  }

  // Verify segments cover [0, n)
  if (segments[0].l != 0) {
    return TestResult(false, "First segment doesn't start at 0");
  }

  // Verify segments are ordered and overlap
  for (size_t i = 1; i < segment_count; i++) {
    if (segments[i].l >= segments[i - 1].r) {
      return TestResult(false, "Gap between segments at index " + to_string(i));
    }
    if (segments[i].l <= segments[i - 1].l) {
      return TestResult(false,
                        "Segments not ordered by left boundary at " + to_string(i));
    }
  }

  // Verify each segment size >= 2*B (except last segment which may be smaller)
  for (size_t i = 0; i + 1 < segment_count; i++) {
    size_t seg_size = segments[i].r - segments[i].l;
    if (seg_size < 2 * B_param) {
      return TestResult(false, "Segment " + to_string(i) + " size " +
                                    to_string(seg_size) + " < 2*B=" +
                                    to_string(2 * B_param));
    }
  }

  return TestResult(true, to_string(segment_count) + " overlapping segments");
}

// Verify SegmentSelection results (called after SegmentSelection, before SegmentAdjustment)
TestResult VerifySegmentSelection(const Segmentor<uint64_t>& segmentor) {
  auto& valid_segments = segmentor.valid_segments;
  size_t valid_count = valid_segments.size();
  size_t n = segmentor.n;

  if (valid_count == 0) {
    return TestResult(false, "No valid segments selected");
  }

  // Verify selection covers [0, n) - first segment starts at 0 or covers it
  if (valid_segments[0].l > 0) {
    return TestResult(false, "Selected segments don't cover start");
  }

  // Verify selected segments overlap (before adjustment)
  for (size_t i = 1; i < valid_count; i++) {
    if (valid_segments[i].l >= valid_segments[i - 1].r) {
      return TestResult(false,
                        "Gap in selected segments at index " + to_string(i));
    }
  }

  // Verify last segment covers to the end
  if (valid_segments[valid_count - 1].r < n) {
    return TestResult(false, "Selected segments don't cover end");
  }

  size_t original_count = segmentor.segments.size();
  double reduction = 100.0 * (1.0 - (double)valid_count / original_count);

  return TestResult(true, to_string(valid_count) + "/" +
                              to_string(original_count) + " segments selected (" +
                              to_string((int)reduction) + "% reduction)");
}

// Verify SegmentAdjustment results
TestResult VerifySegmentAdjustment(const Segmentor<uint64_t>& segmentor) {
  auto& valid_segments = segmentor.valid_segments;
  size_t valid_count = valid_segments.size();
  size_t n = segmentor.n;

  // Verify no overlap after adjustment
  for (size_t i = 1; i < valid_count; i++) {
    if (valid_segments[i].l < valid_segments[i - 1].r) {
      return TestResult(false, "Overlap remains at index " + to_string(i) +
                                    ": prev.r=" +
                                    to_string(valid_segments[i - 1].r) +
                                    ", curr.l=" + to_string(valid_segments[i].l));
    }
  }

  // Verify segments form a partition of [0, n)
  // First segment should start at 0
  if (valid_segments[0].l != 0) {
    return TestResult(false, "First adjusted segment doesn't start at 0");
  }

  // Consecutive segments should be adjacent
  for (size_t i = 1; i < valid_count; i++) {
    if (valid_segments[i].l != valid_segments[i - 1].r) {
      return TestResult(false,
                        "Gap after adjustment at index " + to_string(i));
    }
  }

  // Last segment should end at n
  if (valid_segments[valid_count - 1].r != n) {
    return TestResult(false, "Last segment doesn't end at n");
  }

  return TestResult(true, to_string(valid_count) + " non-overlapping segments");
}

// Verify FGD generation results
TestResult VerifyFGDGeneration(const Segmentor<uint64_t>& segmentor) {
  size_t fgd_count = segmentor.fgd_size;

  if (fgd_count == 0) {
    return TestResult(false, "No FGDs generated");
  }

  // Verify fgd_keys (left-side at this point) are sorted
  for (size_t i = 1; i < fgd_count; i++) {
    if (segmentor.fgd_keys[i] < segmentor.fgd_keys[i - 1]) {
      return TestResult(false, "FGD keys not sorted at index " + to_string(i));
    }
  }

  // Verify AP parameters are valid
  size_t total_ap_count = 0;
  for (size_t i = 0; i < fgd_count; i++) {
    size_t ap_size = segmentor.fgd_values[i].GetAPSize();
    total_ap_count += ap_size;

    // AP step should be a power of 2 (stored as exponent)
    uint16_t ap_step = segmentor.fgd_values[i].GetAPStep();
    if (ap_step > 63) {
      return TestResult(false, "Invalid AP step at FGD " + to_string(i));
    }
  }

  return TestResult(true, to_string(fgd_count) + " FGDs, " +
                              to_string(total_ap_count) + " total APs");
}

// Verify finger key generation results
TestResult VerifyFingerKeys(const Segmentor<uint64_t>& segmentor) {
  auto& finger_keys = segmentor.finger_keys;
  size_t finger_count = finger_keys.size();

  if (finger_count == 0) {
    return TestResult(false, "No finger keys generated");
  }

  // Verify finger keys are sorted
  for (size_t i = 1; i < finger_count; i++) {
    if (finger_keys[i] < finger_keys[i - 1]) {
      return TestResult(false,
                        "Finger keys not sorted at index " + to_string(i));
    }
  }

  // Verify finger_offsets are consistent
  auto& finger_offsets = segmentor.finger_offsets;
  size_t fgd_count = segmentor.fgd_size;

  for (size_t i = 0; i < fgd_count; i++) {
    size_t expected_fingers = segmentor.fgd_values[i].GetAPSize() + 1;
    size_t actual_fingers;
    if (i + 1 < fgd_count) {
      actual_fingers = finger_offsets[i + 1] - finger_offsets[i];
    } else {
      actual_fingers = finger_count - finger_offsets[i];
    }

    if (actual_fingers != expected_fingers) {
      return TestResult(false, "Finger count mismatch at FGD " + to_string(i));
    }
  }

  return TestResult(true, to_string(finger_count) + " finger keys generated");
}

// Verify finger position computation results
TestResult VerifyFingerPositions(const Segmentor<uint64_t>& segmentor, const uint64_t* keys, size_t B_param) {
  auto& finger_indexes = segmentor.finger_indexes;
  auto& finger_keys = segmentor.finger_keys;
  size_t finger_count = finger_keys.size();
  size_t n = segmentor.n;

  // Verify finger positions are valid
  for (size_t i = 0; i < finger_count; i++) {
    size_t pos = finger_indexes[i];
    if (pos > n) {
      return TestResult(false, "Finger position " + to_string(i) +
                                    " = " + to_string(pos) + " exceeds n=" + to_string(n));
    }

    // Verify lower_bound property: keys[pos-1] < finger_key <= keys[pos]
    if (pos > 0 && keys[pos - 1] >= finger_keys[i]) {
      return TestResult(false, "Lower bound property violated at finger " +
                                    to_string(i));
    }
    if (pos < n && keys[pos] < finger_keys[i]) {
      return TestResult(false, "Lower bound property violated at finger " +
                                    to_string(i));
    }
  }

  // Verify positions are non-decreasing and distance between fingers <= B
  for (size_t i = 1; i < finger_count; i++) {
    if (finger_indexes[i] < finger_indexes[i - 1]) {
      return TestResult(false,
                        "Finger positions not monotonic at " + to_string(i));
    }

    size_t distance = finger_indexes[i] - finger_indexes[i - 1];
    if (distance > B_param) {
      return TestResult(false, "Distance between finger " + to_string(i - 1) +
                                    " and " + to_string(i) + " is " +
                                    to_string(distance) + " > B=" + to_string(B_param));
    }
  }

  // Verify last finger to end distance <= B
  if (finger_count > 0) {
    size_t last_distance = n - finger_indexes[finger_count - 1];
    if (last_distance > B_param) {
      return TestResult(false, "Distance from last finger to end is " +
                                    to_string(last_distance) + " > B=" + to_string(B_param));
    }
  }

  return TestResult(true, "All " + to_string(finger_count) +
                              " finger positions verified");
}

// Verify full pipeline coverage
TestResult VerifyFullPipeline(const Segmentor<uint64_t>& segmentor) {
  size_t fgd_count = segmentor.fgd_size;
  size_t finger_count = segmentor.finger_keys.size();
  size_t n = segmentor.n;

  if (fgd_count == 0) {
    return TestResult(false, "Run() produced no FGDs");
  }

  if (finger_count == 0) {
    return TestResult(false, "Run() produced no fingers");
  }

  // Verify finger coverage
  auto& finger_indexes = segmentor.finger_indexes;

  // All keys should be covered by some finger range
  // Check that finger ranges cover [0, n)
  vector<bool> covered(n, false);

  for (size_t i = 0; i < finger_count; i++) {
    size_t start = finger_indexes[i];
    size_t end = (i + 1 < finger_count) ? finger_indexes[i + 1] : n;
    for (size_t j = start; j < end && j < n; j++) {
      covered[j] = true;
    }
  }

  size_t uncovered = 0;
  for (size_t i = 0; i < n; i++) {
    if (!covered[i]) {
      uncovered++;
    }
  }

  if (uncovered > 0) {
    return TestResult(false, to_string(uncovered) + " keys not covered");
  }

  double keys_per_finger = (double)n / finger_count;

  return TestResult(true, "FGDs=" + to_string(fgd_count) +
                              ", fingers=" + to_string(finger_count) +
                              ", keys/finger=" + to_string(keys_per_finger));
}

//==============================================================================
// Run all tests - interleaves segmentor steps with verification
//==============================================================================
bool RunAllTests(size_t n, size_t B_param, size_t F_param) {
  cout << "\n=== Testing n=" << n << ", B=" << B_param << ", F=" << F_param
       << " ===" << endl;

  uint64_t* keys = GenerateSortedKeys(n);
  Fraction covering_finger_density(1, F_param);

  Segmentor<uint64_t> segmentor(
      n, keys, B_param, covering_finger_density);

  int passed = 0;
  int total = 0;

  auto run_test = [&](const string& name, TestResult result) {
    PrintTestResult(name, result);
    total++;
    if (result.passed) passed++;
  };

  // Step 1: BuildMaximalCovering then verify
  segmentor.BuildMaximalCovering();
  run_test("MaximalCovering", VerifyMaximalCovering(segmentor, B_param));

  // Step 2: SegmentSelection then verify
  segmentor.SegmentSelection();
  run_test("SegmentSelection", VerifySegmentSelection(segmentor));

  // Step 3: SegmentAdjustment then verify
  segmentor.SegmentAdjustment();
  run_test("SegmentAdjustment", VerifySegmentAdjustment(segmentor));

  // Step 4: Generate FGD (left-side keys + values)
  segmentor.GenerateFGD();
  run_test("FGDGeneration", VerifyFGDGeneration(segmentor));

  // Step 5: GenerateFingerKeys then verify
  segmentor.GenerateFingerKeys();
  run_test("FingerKeys", VerifyFingerKeys(segmentor));

  // Step 6: ComputeFingerPositions then verify
  segmentor.ComputeFingerPositions();
  run_test("FingerPositions", VerifyFingerPositions(segmentor, keys, B_param));

  // Step 7: Convert to right-side keys and verify full pipeline
  segmentor.ConvertFGDKeysToRightSide();
  run_test("FullPipeline", VerifyFullPipeline(segmentor));

  cout << "--- Result: " << passed << "/" << total << " passed ---" << endl;

  free(keys);
  return passed == total;
}


//==============================================================================
// Main
//==============================================================================
int main(int argc, char* argv[]) {
  cout << "=== Segmenting Component Tests ===" << endl;
  cout << "Default B=" << B << ", F=" << F << endl;

  size_t test_n = 1000;  // Default test size (small for fast testing)
  if (argc > 1) test_n = stoull(argv[1]);

  // Main tests with default parameters
  bool passed = RunAllTests(test_n, B, F);

  // Edge case tests: tiny inputs
  cout << "\n=== Edge Case Tests ===" << endl;
  passed = RunAllTests(1, B, F) && passed;
  passed = RunAllTests(2, B, F) && passed;
  passed = RunAllTests(3, B, F) && passed;

  // Large input test
  passed = RunAllTests(1000000, B, F) && passed;

  cout << "\n=== Summary ===" << endl;
  cout << "All test scenarios completed!" << endl;

  return passed ? 0 : 1;
}
