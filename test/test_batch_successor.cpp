#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#include "common_internal/construction/batch_successor.h"
#include "parlay/parallel.h"

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

// Test 1: Basic correctness test
TestResult TestBasicCorrectness() {
  // Simple test case: keys = [1, 3, 5, 7, 9], queries = [0, 2, 4, 6, 8, 10]
  vector<uint64_t> keys = {1, 3, 5, 7, 9};
  vector<uint64_t> queries = {0, 2, 4, 6, 8, 10};
  vector<size_t> expected = {0, 1, 2, 3, 4, 5};  // successor indices

  size_t n = keys.size();
  size_t m = queries.size();
  vector<size_t> result(m);

  BatchSuccessor(n, keys.data(), m, queries.data(), result.data());

  for (size_t i = 0; i < m; i++) {
    if (result[i] != expected[i]) {
      return TestResult(false, "Mismatch at index " + to_string(i) +
                                   ": expected " + to_string(expected[i]) +
                                   ", got " + to_string(result[i]));
    }
  }

  return TestResult(true, "All basic cases passed");
}

// Test 2: Edge cases
TestResult TestEdgeCases() {
  // Test empty queries
  {
    vector<uint64_t> keys = {1, 2, 3};
    vector<size_t> result;
    BatchSuccessor(keys.size(), keys.data(), 0, (uint64_t*)nullptr,
                   result.data());
  }

  // Test single element
  {
    vector<uint64_t> keys = {5};
    vector<uint64_t> queries = {3, 5, 7};
    vector<size_t> expected = {0, 0, 1};
    vector<size_t> result(3);

    BatchSuccessor(keys.size(), keys.data(), queries.size(), queries.data(),
                   result.data());

    for (size_t i = 0; i < queries.size(); i++) {
      if (result[i] != expected[i]) {
        return TestResult(false, "Single element test failed at index " +
                                     to_string(i));
      }
    }
  }

  // Test all queries smaller than all keys
  {
    vector<uint64_t> keys = {10, 20, 30};
    vector<uint64_t> queries = {1, 5, 9};
    vector<size_t> expected = {0, 0, 0};
    vector<size_t> result(3);

    BatchSuccessor(keys.size(), keys.data(), queries.size(), queries.data(),
                   result.data());

    for (size_t i = 0; i < queries.size(); i++) {
      if (result[i] != expected[i]) {
        return TestResult(false, "All smaller test failed");
      }
    }
  }

  // Test all queries larger than all keys
  {
    vector<uint64_t> keys = {10, 20, 30};
    vector<uint64_t> queries = {31, 40, 50};
    vector<size_t> expected = {3, 3, 3};  // n = 3, means no successor
    vector<size_t> result(3);

    BatchSuccessor(keys.size(), keys.data(), queries.size(), queries.data(),
                   result.data());

    for (size_t i = 0; i < queries.size(); i++) {
      if (result[i] != expected[i]) {
        return TestResult(false, "All larger test failed");
      }
    }
  }

  return TestResult(true, "All edge cases passed");
}

// Test 3: Verify parallel vs serial consistency
TestResult TestParallelSerialConsistency() {
  mt19937_64 rng(42);
  size_t n = 100000;
  size_t m = 50000;

  // Generate sorted keys
  vector<uint64_t> keys(n);
  for (size_t i = 0; i < n; i++) {
    keys[i] = rng() % (n * 10);
  }
  sort(keys.begin(), keys.end());

  // Generate sorted queries
  vector<uint64_t> queries(m);
  for (size_t i = 0; i < m; i++) {
    queries[i] = rng() % (n * 10);
  }
  sort(queries.begin(), queries.end());

  // Run parallel version
  vector<size_t> result_parallel(m);
  BatchSuccessor(n, keys.data(), m, queries.data(), result_parallel.data());

  // Run serial version
  vector<size_t> result_serial(m);
  BatchSuccessorSerial(n, keys.data(), m, queries.data(), result_serial.data());

  // Compare results
  for (size_t i = 0; i < m; i++) {
    if (result_parallel[i] != result_serial[i]) {
      return TestResult(false, "Mismatch at index " + to_string(i) +
                                   ": parallel=" + to_string(result_parallel[i]) +
                                   ", serial=" + to_string(result_serial[i]));
    }
  }

  return TestResult(true, "Parallel and serial results match for " +
                              to_string(m) + " queries");
}

// Test 4: Verify using VerifyBatchSuccessor
TestResult TestVerification() {
  mt19937_64 rng(123);
  size_t n = 10000;
  size_t m = 5000;

  // Generate sorted keys
  vector<uint64_t> keys(n);
  for (size_t i = 0; i < n; i++) {
    keys[i] = rng() % (n * 10);
  }
  sort(keys.begin(), keys.end());

  // Generate sorted queries
  vector<uint64_t> queries(m);
  for (size_t i = 0; i < m; i++) {
    queries[i] = rng() % (n * 10);
  }
  sort(queries.begin(), queries.end());

  // Run BatchSuccessor
  vector<size_t> result(m);
  BatchSuccessor(n, keys.data(), m, queries.data(), result.data());

  // Verify results
  if (!VerifyBatchSuccessor(n, keys.data(), m, queries.data(), result.data())) {
    return TestResult(false, "Verification failed");
  }

  return TestResult(true, "Verification passed for " + to_string(m) +
                              " queries");
}

// Test 5: Large scale test
TestResult TestLargeScale() {
  mt19937_64 rng(999);
  size_t n = 1000000;
  size_t m = 500000;

  // Generate sorted keys
  vector<uint64_t> keys(n);
  for (size_t i = 0; i < n; i++) {
    keys[i] = rng();
  }
  sort(keys.begin(), keys.end());

  // Generate sorted queries
  vector<uint64_t> queries(m);
  for (size_t i = 0; i < m; i++) {
    queries[i] = rng();
  }
  sort(queries.begin(), queries.end());

  // Run BatchSuccessor
  vector<size_t> result(m);
  BatchSuccessor(n, keys.data(), m, queries.data(), result.data());

  // Verify results
  if (!VerifyBatchSuccessor(n, keys.data(), m, queries.data(), result.data())) {
    return TestResult(false, "Large scale verification failed");
  }

  return TestResult(true, "Large scale test passed (" + to_string(n) +
                              " keys, " + to_string(m) + " queries)");
}

// Test 6: Duplicate keys test
TestResult TestDuplicateKeys() {
  // Keys with duplicates
  vector<uint64_t> keys = {1, 1, 2, 2, 2, 3, 5, 5, 5, 5};
  vector<uint64_t> queries = {0, 1, 2, 3, 4, 5, 6};
  // Expected: first occurrence index for each query
  vector<size_t> expected = {0, 0, 2, 5, 6, 6, 10};

  size_t n = keys.size();
  size_t m = queries.size();
  vector<size_t> result(m);

  BatchSuccessor(n, keys.data(), m, queries.data(), result.data());

  for (size_t i = 0; i < m; i++) {
    if (result[i] != expected[i]) {
      return TestResult(false, "Duplicate keys test failed at index " +
                                   to_string(i) + ": expected " +
                                   to_string(expected[i]) + ", got " +
                                   to_string(result[i]));
    }
  }

  return TestResult(true, "Duplicate keys test passed");
}

// Test 7: Parlay sequence interface
TestResult TestParlaySequenceInterface() {
  vector<uint64_t> keys = {10, 20, 30, 40, 50};
  parlay::sequence<uint64_t> queries = {5, 15, 25, 35, 45, 55};

  auto result = BatchSuccessor(keys.size(), keys.data(), queries);

  vector<size_t> expected = {0, 1, 2, 3, 4, 5};
  for (size_t i = 0; i < queries.size(); i++) {
    if (result[i] != expected[i]) {
      return TestResult(false, "Parlay sequence interface failed");
    }
  }

  return TestResult(true, "Parlay sequence interface works correctly");
}

int main() {
  cout << "========================================" << endl;
  cout << "BatchSuccessor Unit Tests" << endl;
  cout << "========================================" << endl;
  cout << endl;

  vector<pair<string, TestResult>> results;

  results.push_back({"Basic Correctness", TestBasicCorrectness()});
  results.push_back({"Edge Cases", TestEdgeCases()});
  results.push_back({"Parallel/Serial Consistency", TestParallelSerialConsistency()});
  results.push_back({"Verification Function", TestVerification()});
  results.push_back({"Large Scale", TestLargeScale()});
  results.push_back({"Duplicate Keys", TestDuplicateKeys()});
  results.push_back({"Parlay Sequence Interface", TestParlaySequenceInterface()});

  int passed = 0;
  int failed = 0;

  for (const auto& [name, result] : results) {
    PrintTestResult(name, result);
    if (result.passed) {
      passed++;
    } else {
      failed++;
    }
  }

  cout << endl;
  cout << "========================================" << endl;
  cout << "Results: " << passed << " passed, " << failed << " failed" << endl;
  if (failed == 0) {
    cout << "ALL TESTS PASSED" << endl;
  } else {
    cout << "SOME TESTS FAILED" << endl;
  }
  cout << "========================================" << endl;

  return failed > 0 ? 1 : 0;
}
