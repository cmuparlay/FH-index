#include <fstream>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

#include "fh_index_ro/storage/kv_layer_ro.h"
#include "parlay/parallel.h"
#include "util/io_utility.h"
#include "timer.hpp"

using namespace std;
using namespace fh_index_ro;

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

// Utility function to print test result
void PrintTestResult(const string& test_name, const TestResult& result) {
  cout << "[" << (result.passed ? "PASS" : "FAIL") << "] " << test_name;
  if (!result.message.empty()) {
    cout << ": " << result.message;
  }
  cout << endl;
}

// Test 1: Verify BulkLoad works correctly
TestResult TestBulkLoad(const vector<uint64_t>& keys_vec) {
  size_t n = keys_vec.size();

  // Round up to cacheline boundary
  constexpr size_t kNumKeyPerCacheline = 64 / sizeof(uint64_t);
  size_t array_size = ((n + kNumKeyPerCacheline - 1) / kNumKeyPerCacheline) * kNumKeyPerCacheline;

  // Allocate and initialize keys
  uint64_t* keys = new uint64_t[array_size];
  uint64_t* values = new uint64_t[array_size];

  parlay::parallel_for(0, n, [&](size_t i) {
    keys[i] = keys_vec[i];
    values[i] = i;  // Simple value: just the index
  });

  // Fill padding with max value
  parlay::parallel_for(n, array_size, [&](size_t i) {
    keys[i] = std::numeric_limits<uint64_t>::max();
    values[i] = 0;
  });

  try {
    KVLayerRO<uint64_t, uint64_t> kv_layer;
    kv_layer.BulkLoad(0, n, array_size, std::move(keys), std::move(values), B,
                      fh_index::internal::Fraction(1, F));

    if (kv_layer.GetSize() != n) {
      return TestResult(false, "Size mismatch: expected " + to_string(n) + ", got " + to_string(kv_layer.GetSize()));
    }

    return TestResult(true, "BulkLoad successful with " + to_string(n) + " keys");

  } catch (const exception& e) {
    return TestResult(false, string("Exception: ") + e.what());
  }
}

// Test 2: Verify LowerBound methods return correct results
TestResult TestLowerBound(const vector<uint64_t>& keys_vec, const string& variant) {
  size_t n = keys_vec.size();

  // Round up to cacheline boundary
  constexpr size_t kNumKeyPerCacheline = 64 / sizeof(uint64_t);
  size_t array_size = ((n + kNumKeyPerCacheline - 1) / kNumKeyPerCacheline) * kNumKeyPerCacheline;

  // Allocate and initialize keys
  uint64_t* keys = new uint64_t[array_size];
  uint64_t* values = new uint64_t[array_size];

  parlay::parallel_for(0, n, [&](size_t i) {
    keys[i] = keys_vec[i];
    values[i] = i;
  });

  parlay::parallel_for(n, array_size, [&](size_t i) {
    keys[i] = std::numeric_limits<uint64_t>::max();
    values[i] = 0;
  });

  try {
    KVLayerRO<uint64_t, uint64_t> kv_layer;
    kv_layer.BulkLoad(0, n, array_size, std::move(keys), std::move(values), B,
                      fh_index::internal::Fraction(1, F));

    // Test searches for various keys
    size_t num_tests = min(n, (size_t)100);
    size_t errors = 0;

    for (size_t i = 0; i < num_tests; i++) {
      size_t idx = i * n / num_tests;
      uint64_t search_key = keys_vec[idx];

      // Search the entire range (in cacheline units)
      size_t left_filter = 0;
      size_t right_filter = array_size / kNumKeyPerCacheline;

      size_t result;
      if (variant == "AVX2") {
#ifdef __AVX2__
        result = kv_layer.LowerBoundOverKVArrayAvx2(search_key, left_filter * kNumKeyPerCacheline, right_filter * kNumKeyPerCacheline);
#else
        return TestResult(true, "AVX2 not supported, skipped");
#endif
      } else if (variant == "AVX512") {
#ifdef __AVX512F__
        result = kv_layer.LowerBoundOverKVArrayAvx512(search_key, left_filter * kNumKeyPerCacheline, right_filter * kNumKeyPerCacheline);
#else
        return TestResult(true, "AVX512 not supported, skipped");
#endif
      } else if (variant == "Naive") {
        result = kv_layer.LowerBoundOverKVArrayNaive(search_key, left_filter * kNumKeyPerCacheline, right_filter * kNumKeyPerCacheline);
      } else {
        return TestResult(false, "Unknown variant: " + variant);
      }

      // Verify result
      if (result != idx) {
        errors++;
        if (errors <= 5) {
          cout << "  Search for key " << search_key << " returned index " << result
               << ", expected " << idx << endl;
        }
      }
    }

    if (errors > 0) {
      return TestResult(false, "Found " + to_string(errors) + " search errors");
    }

    return TestResult(true, "All " + to_string(num_tests) + " searches correct");

  } catch (const exception& e) {
    return TestResult(false, string("Exception: ") + e.what());
  }
}

// Test 3: Verify getter methods work correctly
TestResult TestGetters(const vector<uint64_t>& keys_vec) {
  size_t n = keys_vec.size();

  constexpr size_t kNumKeyPerCacheline = 64 / sizeof(uint64_t);
  size_t array_size = ((n + kNumKeyPerCacheline - 1) / kNumKeyPerCacheline) * kNumKeyPerCacheline;

  uint64_t* keys = new uint64_t[array_size];
  uint64_t* values = new uint64_t[array_size];

  parlay::parallel_for(0, n, [&](size_t i) {
    keys[i] = keys_vec[i];
    values[i] = i * 2;  // Value = 2 * index
  });

  parlay::parallel_for(n, array_size, [&](size_t i) {
    keys[i] = std::numeric_limits<uint64_t>::max();
    values[i] = 0;
  });

  try {
    KVLayerRO<uint64_t, uint64_t> kv_layer;
    kv_layer.BulkLoad(0, n, array_size, std::move(keys), std::move(values), B,
                      fh_index::internal::Fraction(1, F));

    // Test GetKV
    for (size_t i = 0; i < min(n, (size_t)10); i++) {
      auto kv = kv_layer.GetKV(i);
      if (kv.first != keys_vec[i] || kv.second != i * 2) {
        return TestResult(false,
            "GetKV(" + to_string(i) + ") returned incorrect value");
      }
    }

    // Test BatchGetKey
    constexpr size_t kBatchSize = 4;
    size_t indices[kBatchSize];
    uint64_t batch_keys[kBatchSize];

    for (size_t i = 0; i < kBatchSize && i < n; i++) {
      indices[i] = i;
    }

    if (n >= kBatchSize) {
      kv_layer.BatchGetKey<kBatchSize>(indices, batch_keys);

      for (size_t i = 0; i < kBatchSize; i++) {
        if (batch_keys[i] != keys_vec[i]) {
          return TestResult(false, "BatchGetKey returned incorrect value at index " + to_string(i));
        }
      }
    }

    return TestResult(true, "All getter methods work correctly");

  } catch (const exception& e) {
    return TestResult(false, string("Exception: ") + e.what());
  }
}

// Run all tests on a dataset
bool RunTests(const string& dataset_name, const vector<uint64_t>& keys_vec) {
  cout << "\n========================================" << endl;
  cout << "Testing dataset: " << dataset_name << endl;
  cout << "Dataset size: " << keys_vec.size() << endl;
  cout << "Parameters: B=" << B << ", F=" << F << endl;
  cout << "========================================\n" << endl;

  auto result_bulkload = TestBulkLoad(keys_vec);
  PrintTestResult("Test 1: BulkLoad", result_bulkload);

  auto result_naive = TestLowerBound(keys_vec, "Naive");
  PrintTestResult("Test 2: LowerBoundNaive", result_naive);

  auto result_avx2 = TestLowerBound(keys_vec, "AVX2");
  PrintTestResult("Test 3: LowerBoundAvx2", result_avx2);

  auto result_avx512 = TestLowerBound(keys_vec, "AVX512");
  PrintTestResult("Test 4: LowerBoundAvx512", result_avx512);

  auto result_getters = TestGetters(keys_vec);
  PrintTestResult("Test 5: Getter methods", result_getters);

  // Summary
  cout << "\n----------------------------------------" << endl;
  const bool passed = result_bulkload.passed && result_naive.passed &&
      result_avx2.passed && result_avx512.passed && result_getters.passed;
  if (passed) {
    cout << "ALL TESTS PASSED ✓" << endl;
  } else {
    cout << "SOME TESTS FAILED ✗" << endl;
  }
  cout << "----------------------------------------\n" << endl;
  return passed;
}

int main(int argc, char* argv[]) {
  init_root_timer();
  timer::default_detail = false;
  timer::print_when_time = false;

  bool passed = true;

  // Test 1: Small uniform random dataset
  {
    vector<uint64_t> keys = fh_index::util::UniformRandomKeys(10000, 0, true);
    passed = RunTests("Small Uniform Random (10K)", keys) && passed;
  }

  // Test 2: Medium uniform random dataset
  {
    vector<uint64_t> keys = fh_index::util::UniformRandomKeys(100000, 0, true);
    passed = RunTests("Medium Uniform Random (100K)", keys) && passed;
  }

  // Test 3: Large uniform random dataset
  {
    vector<uint64_t> keys = fh_index::util::UniformRandomKeys(1000000, 0, true);
    passed = RunTests("Large Uniform Random (1M)", keys) && passed;
  }

  // Test 4: Custom dataset from command line if provided
  if (argc >= 3) {
    string dataset = argv[1];
    uint64_t load_limit = std::stoull(argv[2]);

    vector<uint64_t> keys;
    if (dataset == "GEN") {
      keys = fh_index::util::UniformRandomKeys(load_limit, 0, true);
    } else {
      keys = fh_index::util::LoadKeysFromBinary(dataset, load_limit);
    }

    passed = RunTests(dataset + " (custom)", keys) && passed;
  }

  cout << "\n========================================" << endl;
  cout << "All test scenarios completed!" << endl;
  cout << "========================================" << endl;

  return passed ? 0 : 1;
}
