#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "common_internal/construction/fraction.h"
#include "common_internal/construction/multi_level_segmentor.h"
#include "common_internal/representation/finger_group_descriptor.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "util/io_utility.h"

using namespace std;
using namespace fh_index::internal::segmenting;
using fh_index::internal::FingerGroupDescriptorValue;
using fh_index::internal::Fraction;

// Friend class to access private methods for unit testing
class MultiLevelSegmentorTest {
 public:
  static std::pair<FingerGroupDescriptorValue, FingerGroupDescriptorValue>
  SplitFGDValue(const FingerGroupDescriptorValue& val, uint64_t K) {
    return MultiLevelSegmentor<uint64_t>::SplitFGDValue(val, K);
  }

  static std::pair<parlay::sequence<uint64_t>, parlay::sequence<FingerGroupDescriptorValue>>
  MergeKeysWithFGDValues(
      MultiLevelSegmentor<uint64_t>& seg,
      const parlay::sequence<uint64_t>& fgd_keys,
      const parlay::sequence<FingerGroupDescriptorValue>& fgd_values,
      const parlay::sequence<uint64_t>& finger_keys) {
    return seg.MergeKeysWithFGDValues(fgd_keys, fgd_values, finger_keys);
  }
};

// Default parameters
static constexpr size_t LEAF_B = 48;
static constexpr size_t LEAF_F = 4;
static constexpr size_t INTERNAL_B = 32;
static constexpr size_t INTERNAL_F = 2;

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
  if (!result.passed) {
    // Fail fast for easier debugging
    exit(1);
  }
}

static int total_tests = 0;
static int passed_tests = 0;

void RunTest(const string& name, TestResult result) {
  PrintTestResult(name, result);
  total_tests++;
  if (result.passed) passed_tests++;
}

// =============================================================================
// Unit tests for SplitFGDValue
// =============================================================================

// Helper: collect all AP cut keys from a FGDValue
vector<uint64_t> CollectAPKeys(const FingerGroupDescriptorValue& val) {
  vector<uint64_t> keys;
  for (size_t j = 0; j < val.GetAPSize(); j++) {
    keys.push_back(val.GetAPKey(j));
  }
  return keys;
}

TestResult TestSplitBasic() {
  // AP: start=100, size=4, step=2 => cuts at 100, 104, 108, 112
  FingerGroupDescriptorValue val(100, 4, 2, nullptr);

  // Split at K=106: left should have cuts {100, 104}, right {108, 112}
  auto [left, right] = MultiLevelSegmentorTest::SplitFGDValue(val, 106);

  if (left.GetAPSize() + right.GetAPSize() != val.GetAPSize()) {
    return TestResult(false, "ap_size not preserved: left=" +
        to_string(left.GetAPSize()) + " right=" + to_string(right.GetAPSize()));
  }

  auto left_keys = CollectAPKeys(left);
  auto right_keys = CollectAPKeys(right);
  for (auto k : left_keys) {
    if (k >= 106) return TestResult(false, "left cut " + to_string(k) + " >= K=106");
  }
  for (auto k : right_keys) {
    if (k < 106) return TestResult(false, "right cut " + to_string(k) + " < K=106");
  }

  if (left.GetAPSize() != 2) {
    return TestResult(false, "expected left.ap_size=2, got " + to_string(left.GetAPSize()));
  }

  return TestResult(true, "left.ap_size=2, right.ap_size=2");
}

TestResult TestSplitBeforeAll() {
  // AP: start=100, size=3, step=0 => cuts at 100, 101, 102
  FingerGroupDescriptorValue val(100, 3, 0, nullptr);

  // Split at K=50: all cuts >= K, so left.ap_size=0
  auto [left, right] = MultiLevelSegmentorTest::SplitFGDValue(val, 50);

  if (left.GetAPSize() != 0) {
    return TestResult(false, "expected left.ap_size=0, got " + to_string(left.GetAPSize()));
  }
  if (right.GetAPSize() != 3) {
    return TestResult(false, "expected right.ap_size=3, got " + to_string(right.GetAPSize()));
  }

  return TestResult(true, "all cuts in right");
}

TestResult TestSplitAfterAll() {
  // AP: start=100, size=3, step=0 => cuts at 100, 101, 102
  FingerGroupDescriptorValue val(100, 3, 0, nullptr);

  // Split at K=200: all cuts < K, so right.ap_size=0
  auto [left, right] = MultiLevelSegmentorTest::SplitFGDValue(val, 200);

  if (left.GetAPSize() != 3) {
    return TestResult(false, "expected left.ap_size=3, got " + to_string(left.GetAPSize()));
  }
  if (right.GetAPSize() != 0) {
    return TestResult(false, "expected right.ap_size=0, got " + to_string(right.GetAPSize()));
  }

  return TestResult(true, "all cuts in left");
}

TestResult TestSplitOnCut() {
  // AP: start=100, size=4, step=2 => cuts at 100, 104, 108, 112
  FingerGroupDescriptorValue val(100, 4, 2, nullptr);

  // Split at K=108: cut[2]=108 >= K, so left gets {100, 104}, right gets {108, 112}
  auto [left, right] = MultiLevelSegmentorTest::SplitFGDValue(val, 108);

  if (left.GetAPSize() != 2) {
    return TestResult(false, "expected left.ap_size=2, got " + to_string(left.GetAPSize()));
  }
  if (right.GetAPSize() != 2) {
    return TestResult(false, "expected right.ap_size=2, got " + to_string(right.GetAPSize()));
  }

  // Verify right starts at 108
  if (right.GetAPKey(0) != 108) {
    return TestResult(false, "expected right first cut=108, got " + to_string(right.GetAPKey(0)));
  }

  return TestResult(true, "split exactly on cut");
}

TestResult TestSplitEmpty() {
  // ap_size = 0
  FingerGroupDescriptorValue val(100, 0, 0, nullptr);

  auto [left, right] = MultiLevelSegmentorTest::SplitFGDValue(val, 50);

  if (left.GetAPSize() != 0 || right.GetAPSize() != 0) {
    return TestResult(false, "expected both ap_size=0");
  }

  return TestResult(true, "both empty");
}

TestResult TestSplitPreservesUnion() {
  // AP: start=1000, size=8, step=3 => cuts at 1000,1008,1016,...,1056
  FingerGroupDescriptorValue val(1000, 8, 3, nullptr);
  auto original_keys = CollectAPKeys(val);

  // Split at various points and verify union
  uint64_t split_points[] = {999, 1000, 1004, 1008, 1030, 1056, 1057, 2000};
  for (auto K : split_points) {
    auto [left, right] = MultiLevelSegmentorTest::SplitFGDValue(val, K);

    // Collect all cuts from left + right
    auto left_keys = CollectAPKeys(left);
    auto right_keys = CollectAPKeys(right);

    vector<uint64_t> merged;
    merged.insert(merged.end(), left_keys.begin(), left_keys.end());
    merged.insert(merged.end(), right_keys.begin(), right_keys.end());

    if (merged.size() != original_keys.size()) {
      return TestResult(false, "union size mismatch at K=" + to_string(K) +
          ": expected " + to_string(original_keys.size()) + ", got " + to_string(merged.size()));
    }

    for (size_t i = 0; i < merged.size(); i++) {
      if (merged[i] != original_keys[i]) {
        return TestResult(false, "union key mismatch at K=" + to_string(K) + " index " + to_string(i));
      }
    }
  }

  return TestResult(true, "union preserved for all split points");
}

void RunSplitTests() {
  cout << "\n=== SplitFGDValue Unit Tests ===" << endl;
  RunTest("Split.Basic", TestSplitBasic());
  RunTest("Split.BeforeAll", TestSplitBeforeAll());
  RunTest("Split.AfterAll", TestSplitAfterAll());
  RunTest("Split.OnCut", TestSplitOnCut());
  RunTest("Split.Empty", TestSplitEmpty());
  RunTest("Split.PreservesUnion", TestSplitPreservesUnion());
}

// =============================================================================
// Unit tests for MergeKeysWithFGDValues
// =============================================================================

TestResult TestMergeEmpty() {
  // No finger keys to merge
  parlay::sequence<uint64_t> fgd_keys = {0, 100, 200};
  parlay::sequence<FingerGroupDescriptorValue> fgd_values = {
    FingerGroupDescriptorValue(10, 1, 0, nullptr),
    FingerGroupDescriptorValue(110, 1, 0, nullptr),
    FingerGroupDescriptorValue(210, 1, 0, nullptr),
  };
  parlay::sequence<uint64_t> finger_keys;

  // Need an instance to call non-static method
  LevelConfig leaf_config(LEAF_B, Fraction(1, LEAF_F));
  LevelConfig internal_config(INTERNAL_B, Fraction(1, INTERNAL_F));
  uint64_t dummy_key = 0;
  MultiLevelSegmentor<uint64_t> seg(1, &dummy_key, leaf_config, internal_config, 1, 0);

  auto [mk, mv] = MultiLevelSegmentorTest::MergeKeysWithFGDValues(seg, fgd_keys, fgd_values, finger_keys);

  if (mk.size() != fgd_keys.size()) {
    return TestResult(false, "expected same size, got " + to_string(mk.size()));
  }

  return TestResult(true, "empty merge preserves input");
}

TestResult TestMergeBasic() {
  // 3 segments: [0, 100), [100, 200), [200, ...)
  // Each with 1 AP cut
  parlay::sequence<uint64_t> fgd_keys = {0, 100, 200};
  parlay::sequence<FingerGroupDescriptorValue> fgd_values = {
    FingerGroupDescriptorValue(50, 1, 0, nullptr),   // cut at 50
    FingerGroupDescriptorValue(150, 1, 0, nullptr),  // cut at 150
    FingerGroupDescriptorValue(250, 1, 0, nullptr),  // cut at 250
  };
  // Insert finger key 60 into first segment (0, 100)
  parlay::sequence<uint64_t> finger_keys = {60};

  LevelConfig leaf_config(LEAF_B, Fraction(1, LEAF_F));
  LevelConfig internal_config(INTERNAL_B, Fraction(1, INTERNAL_F));
  uint64_t dummy_key = 0;
  MultiLevelSegmentor<uint64_t> seg(1, &dummy_key, leaf_config, internal_config, 1, 0);

  auto [mk, mv] = MultiLevelSegmentorTest::MergeKeysWithFGDValues(seg, fgd_keys, fgd_values, finger_keys);

  // Should have 4 entries: [0, 60, 100, 200]
  if (mk.size() != 4) {
    return TestResult(false, "expected 4 keys, got " + to_string(mk.size()));
  }
  if (mv.size() != 4) {
    return TestResult(false, "expected 4 values, got " + to_string(mv.size()));
  }

  // Check keys are sorted
  for (size_t i = 1; i < mk.size(); i++) {
    if (mk[i] <= mk[i - 1]) {
      return TestResult(false, "keys not sorted at index " + to_string(i));
    }
  }

  // Check original fgd_keys present
  bool has_0 = false, has_100 = false, has_200 = false, has_60 = false;
  for (auto k : mk) {
    if (k == 0) has_0 = true;
    if (k == 60) has_60 = true;
    if (k == 100) has_100 = true;
    if (k == 200) has_200 = true;
  }
  if (!has_0 || !has_60 || !has_100 || !has_200) {
    return TestResult(false, "missing expected keys");
  }

  // Check split: segment [0, 100) had cut at 50. Split at 60:
  // left part: cuts < 60 => {50}, right part: cuts >= 60 => {}
  if (mv[0].GetAPSize() != 1) {
    return TestResult(false, "expected first sub-segment ap_size=1, got " + to_string(mv[0].GetAPSize()));
  }
  if (mv[1].GetAPSize() != 0) {
    return TestResult(false, "expected second sub-segment ap_size=0, got " + to_string(mv[1].GetAPSize()));
  }

  return TestResult(true, "merge with split correct");
}

TestResult TestMergeDuplicate() {
  // finger_key == fgd_key should be skipped (falls outside (fgd_keys[i], fgd_keys[i+1]))
  parlay::sequence<uint64_t> fgd_keys = {0, 100, 200};
  parlay::sequence<FingerGroupDescriptorValue> fgd_values = {
    FingerGroupDescriptorValue(50, 1, 0, nullptr),
    FingerGroupDescriptorValue(150, 1, 0, nullptr),
    FingerGroupDescriptorValue(250, 1, 0, nullptr),
  };
  parlay::sequence<uint64_t> finger_keys = {100};  // == fgd_keys[1]

  LevelConfig leaf_config(LEAF_B, Fraction(1, LEAF_F));
  LevelConfig internal_config(INTERNAL_B, Fraction(1, INTERNAL_F));
  uint64_t dummy_key = 0;
  MultiLevelSegmentor<uint64_t> seg(1, &dummy_key, leaf_config, internal_config, 1, 0);

  auto [mk, mv] = MultiLevelSegmentorTest::MergeKeysWithFGDValues(seg, fgd_keys, fgd_values, finger_keys);

  // finger_key 100 == fgd_keys[1], upper_bound(100) > lower_bound(100), so it's excluded
  // Should remain 3 entries
  if (mk.size() != 3) {
    return TestResult(false, "expected 3 keys (dup skipped), got " + to_string(mk.size()));
  }

  return TestResult(true, "duplicate key skipped");
}

TestResult TestMergeMultiple() {
  // Insert multiple finger keys into one segment
  parlay::sequence<uint64_t> fgd_keys = {0, 1000};
  parlay::sequence<FingerGroupDescriptorValue> fgd_values = {
    FingerGroupDescriptorValue(200, 4, 0, nullptr),  // cuts at 200,201,202,203
    FingerGroupDescriptorValue(1100, 0, 0, nullptr),
  };
  parlay::sequence<uint64_t> finger_keys = {300, 500, 700};

  LevelConfig leaf_config(LEAF_B, Fraction(1, LEAF_F));
  LevelConfig internal_config(INTERNAL_B, Fraction(1, INTERNAL_F));
  uint64_t dummy_key = 0;
  MultiLevelSegmentor<uint64_t> seg(1, &dummy_key, leaf_config, internal_config, 1, 0);

  auto [mk, mv] = MultiLevelSegmentorTest::MergeKeysWithFGDValues(seg, fgd_keys, fgd_values, finger_keys);

  // Segment [0, 1000) gets 3 finger keys => 4 sub-segments
  // Segment [1000, ...) gets 0 => 1
  // Total: 5
  if (mk.size() != 5) {
    return TestResult(false, "expected 5 keys, got " + to_string(mk.size()));
  }

  // Keys should be sorted
  for (size_t i = 1; i < mk.size(); i++) {
    if (mk[i] <= mk[i - 1]) {
      return TestResult(false, "keys not sorted at index " + to_string(i));
    }
  }

  // Total ap cuts across sub-segments of original segment[0] should equal original ap_size=4
  size_t total_ap = 0;
  for (size_t i = 0; i < 4; i++) {  // 4 sub-segments from segment [0,1000)
    total_ap += mv[i].GetAPSize();
  }
  if (total_ap != 4) {
    return TestResult(false, "ap cuts not preserved: expected 4, got " + to_string(total_ap));
  }

  return TestResult(true, "multiple finger keys merge correct");
}

void RunMergeTests() {
  cout << "\n=== MergeKeysWithFGDValues Unit Tests ===" << endl;
  RunTest("Merge.Empty", TestMergeEmpty());
  RunTest("Merge.Basic", TestMergeBasic());
  RunTest("Merge.Duplicate", TestMergeDuplicate());
  RunTest("Merge.Multiple", TestMergeMultiple());
}

// =============================================================================
// Integration tests for Run()
// =============================================================================

// Verify a single level's internal consistency
TestResult VerifyLevel(const LevelData<uint64_t>& level, size_t level_idx,
                       size_t B_limit, size_t lower_size,
                       const string& prefix) {
  size_t fgd_size = level.size();
  if (fgd_size == 0) {
    return TestResult(false, prefix + " empty level");
  }

  // fgd_keys sorted (right-side definition)
  for (size_t i = 1; i < fgd_size; i++) {
    if (level.fgd_keys[i] <= level.fgd_keys[i - 1]) {
      return TestResult(false, prefix + " fgd_keys not sorted at " + to_string(i));
    }
  }

  // fgd_values count matches fgd_keys count
  if (level.fgd_values.size() != fgd_size) {
    return TestResult(false, prefix + " fgd_values size mismatch");
  }

  // Last fgd_key should be max_key (UINT64_MAX)
  if (level.fgd_keys[fgd_size - 1] != numeric_limits<uint64_t>::max()) {
    return TestResult(false, prefix + " last fgd_key != max_key, got " +
        to_string(level.fgd_keys[fgd_size - 1]));
  }

  // finger_offsets consistency
  if (level.finger_offsets.size() != fgd_size + 1) {
    return TestResult(false, prefix + " finger_offsets size mismatch: expected " +
        to_string(fgd_size + 1) + ", got " + to_string(level.finger_offsets.size()));
  }
  if (level.finger_offsets[0] != 0) {
    return TestResult(false, prefix + " finger_offsets[0] != 0");
  }
  for (size_t j = 0; j < fgd_size; j++) {
    size_t expected = level.fgd_values[j].GetAPSize() + 1;
    size_t actual = level.finger_offsets[j + 1] - level.finger_offsets[j];
    if (actual != expected) {
      return TestResult(false, prefix + " finger_offsets mismatch at FGD " + to_string(j) +
          ": expected " + to_string(expected) + ", got " + to_string(actual));
    }
  }

  // finger_indexes count
  size_t total_fingers = level.finger_offsets[fgd_size];
  if (level.finger_indexes.size() != total_fingers) {
    return TestResult(false, prefix + " finger_indexes size mismatch");
  }

  // finger_indexes non-decreasing
  for (size_t i = 1; i < total_fingers; i++) {
    if (level.finger_indexes[i] < level.finger_indexes[i - 1]) {
      return TestResult(false, prefix + " finger_indexes not monotonic at " + to_string(i));
    }
  }

  // Finger sub-interval sizes <= B
  for (size_t j = 0; j < fgd_size; j++) {
    size_t fo_start = level.finger_offsets[j];
    size_t fo_end = level.finger_offsets[j + 1];
    for (size_t k = fo_start; k < fo_end; k++) {
      size_t fi_start = level.finger_indexes[k];
      size_t fi_end;
      if (k + 1 < total_fingers) {
        fi_end = level.finger_indexes[k + 1];
      } else {
        fi_end = lower_size;
      }
      size_t interval = fi_end - fi_start;
      if (interval > B_limit) {
        return TestResult(false, prefix + " sub-interval size " + to_string(interval) +
            " > B=" + to_string(B_limit) + " at FGD " +
            to_string(j) + " finger " + to_string(k));
      }
    }
  }

  return TestResult(true, prefix + " fgd_size=" + to_string(fgd_size) +
      ", fingers=" + to_string(total_fingers));
}

// Run integration test with given parameters
void RunIntegrationTest(const string& label, size_t n, const uint64_t* keys,
                        LevelConfig leaf_config, LevelConfig internal_config,
                        size_t height_limit, size_t top_level_threshold) {
  cout << "\n--- " << label << " (n=" << n
       << ", height_limit=" << height_limit
       << ", threshold=" << top_level_threshold << ") ---" << endl;

  MultiLevelSegmentor<uint64_t> seg(n, keys, leaf_config, internal_config,
                                     height_limit, top_level_threshold);
  seg.Run();

  size_t num_levels = seg.GetLevelCount();
  cout << "  levels=" << num_levels << endl;

  // Verify stop condition
  if (height_limit > 0) {
    RunTest(label + ".LevelCount",
        TestResult(num_levels == height_limit,
            "expected " + to_string(height_limit) + " levels, got " + to_string(num_levels)));
  } else {
    size_t top_fgd_size = seg.GetLevel(num_levels - 1).size();
    RunTest(label + ".TopThreshold",
        TestResult(top_fgd_size <= top_level_threshold,
            "top fgd_size=" + to_string(top_fgd_size) +
            ", threshold=" + to_string(top_level_threshold)));
  }

  // Verify each level
  for (size_t i = 0; i < num_levels; i++) {
    size_t B_limit = (i == 0) ? leaf_config.B : internal_config.B;
    size_t lower_size = (i == 0) ? n : seg.GetLevel(i - 1).size();
    string prefix = "Level" + to_string(i);
    RunTest(label + "." + prefix,
            VerifyLevel(seg.GetLevel(i), i, B_limit, lower_size, prefix));
  }

  // Cross-level: upper level finger_indexes point into valid lower level range
  for (size_t i = 1; i < num_levels; i++) {
    const auto& upper = seg.GetLevel(i);
    const auto& lower = seg.GetLevel(i - 1);
    size_t lower_size = lower.size();

    bool valid = true;
    string err_msg;
    for (size_t f = 0; f < upper.finger_indexes.size(); f++) {
      if (upper.finger_indexes[f] > lower_size) {
        valid = false;
        err_msg = "finger_indexes[" + to_string(f) + "]=" +
            to_string(upper.finger_indexes[f]) + " > lower_size=" + to_string(lower_size);
        break;
      }
    }
    RunTest(label + ".CrossLevel" + to_string(i),
        TestResult(valid, valid ? "all finger_indexes within lower level" : err_msg));
  }
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char* argv[]) {
  cout << "=== MultiLevelSegmentor Tests ===" << endl;

  // --- Unit tests (no dataset needed) ---
  RunSplitTests();
  RunMergeTests();

  // --- Integration tests (dataset required) ---
  string dataset_path;
  size_t load_limit = 100000;

  if (argc > 1) {
    dataset_path = argv[1];
  } else {
    dataset_path = "../../datasets_final/fb_100K_uint64.bin";
  }
  if (argc > 2) {
    load_limit = stoull(argv[2]);
  }

  vector<uint64_t> keys_vec;
  if (dataset_path == "GEN") {
    cout << "\n=== Generating deterministic non-uniform dataset ===" << endl;
    keys_vec.reserve(load_limit);
    uint64_t key = 0;
    for (size_t i = 0; i < load_limit; ++i) {
      keys_vec.push_back(key);
      key += 1 + (i % 100);
    }
  } else {
    cout << "\n=== Loading dataset: " << dataset_path << " ===" << endl;
    keys_vec = fh_index::util::LoadKeysFromBinary(
        dataset_path, load_limit, true, true, false);
    keys_vec = fh_index::util::DeduplicateSortedKeys(keys_vec);
  }
  size_t n = keys_vec.size();
  cout << "Loaded " << n << " unique sorted keys" << endl;

  const uint64_t* keys = keys_vec.data();

  LevelConfig leaf_config(LEAF_B, Fraction(1, LEAF_F));
  LevelConfig internal_config(INTERNAL_B, Fraction(1, INTERNAL_F));

  // height_limit mode
  RunIntegrationTest("HeightLimit1", n, keys, leaf_config, internal_config, 1, 0);
  RunIntegrationTest("HeightLimit2", n, keys, leaf_config, internal_config, 2, 0);
  RunIntegrationTest("HeightLimit3", n, keys, leaf_config, internal_config, 3, 0);

  // top_level_threshold mode
  RunIntegrationTest("Threshold64", n, keys, leaf_config, internal_config, 0, 64);
  RunIntegrationTest("Threshold16", n, keys, leaf_config, internal_config, 0, 16);
  RunIntegrationTest("Threshold4", n, keys, leaf_config, internal_config, 0, 4);

  // Edge cases: small inputs
  cout << "\n=== Edge Cases ===" << endl;
  auto small_keys_vec = fh_index::util::UniformRandomKeys(100, 42, true);
  small_keys_vec = fh_index::util::DeduplicateSortedKeys(small_keys_vec);
  RunIntegrationTest("SmallInput", small_keys_vec.size(), small_keys_vec.data(),
                     leaf_config, internal_config, 1, 0);

  // Very small: single level should suffice
  uint64_t tiny_keys[] = {10, 20, 30, 40, 50};
  RunIntegrationTest("TinyInput", 5, tiny_keys, leaf_config, internal_config, 1, 0);

  // Different B/F configurations
  LevelConfig big_leaf(96, Fraction(1, 16));
  LevelConfig big_internal(64, Fraction(1, 8));
  RunIntegrationTest("BigBlocks", n, keys, big_leaf, big_internal, 2, 0);

  LevelConfig small_leaf(16, Fraction(1, 2));
  LevelConfig small_internal(16, Fraction(1, 2));
  RunIntegrationTest("SmallBlocks", n, keys, small_leaf, small_internal, 2, 0);

  // Exercise non-dyadic resource bounds through the complete hierarchy path.
  LevelConfig fractional_leaf(48, Fraction(3, 20));
  LevelConfig fractional_internal(32, Fraction(1, 3));
  RunIntegrationTest("FractionalDensity", n, keys, fractional_leaf,
                     fractional_internal, 2, 0);

  // Summary
  cout << "\n=== Summary ===" << endl;
  cout << passed_tests << "/" << total_tests << " tests passed" << endl;

  return (passed_tests == total_tests) ? 0 : 1;
}
