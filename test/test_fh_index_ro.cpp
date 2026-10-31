#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "fh_index/fh_index_ro.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "util/io_utility.h"

using namespace std;

using KeyType = uint64_t;
using ValueType = uint64_t;
using EntryType = std::pair<KeyType, ValueType>;

// Index parameters
constexpr size_t kTopLevelThreshold = 256;

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

#ifndef FINGER_SPARSITY
#define FINGER_SPARSITY 4
#endif

constexpr size_t kBlockSize = BLOCK_SIZE;
constexpr size_t kFingerSparsity = FINGER_SPARSITY;

struct TestResult {
  bool passed;
  string message;

  TestResult(bool p, const string& msg) : passed(p), message(msg) {}
};

void PrintTestResult(const string& test_name, const TestResult& result) {
  cout << (result.passed ? "[PASS] " : "[FAIL] ") << test_name << ": "
       << result.message << endl;
}

// Reference implementation using sorted array + binary search
class ReferenceIndex {
 public:
  void BulkLoad(const vector<KeyType>& keys, const vector<ValueType>& values) {
    size_t n = keys.size();
    entries_.resize(n);
    for (size_t i = 0; i < n; i++) {
      entries_[i] = {keys[i], values[i]};
    }
    sort(entries_.begin(), entries_.end(),
         [](const auto& a, const auto& b) { return a.first < b.first; });
  }

  bool Find(KeyType key, ValueType& value) const {
    auto it = lower_bound(entries_.begin(), entries_.end(), key,
                          [](const auto& e, KeyType k) { return e.first < k; });
    if (it != entries_.end() && it->first == key) {
      value = it->second;
      return true;
    }
    return false;
  }

  // Returns index of first element >= key, or size() if not found
  size_t LowerBound(KeyType key) const {
    auto it = lower_bound(entries_.begin(), entries_.end(), key,
                          [](const auto& e, KeyType k) { return e.first < k; });
    return it - entries_.begin();
  }

  size_t Size() const { return entries_.size(); }

  KeyType GetKey(size_t idx) const { return entries_[idx].first; }
  ValueType GetValue(size_t idx) const { return entries_[idx].second; }

 private:
  vector<EntryType> entries_;
};

// Test Find operation
TestResult TestFind(const vector<KeyType>& keys_vec,
                    const vector<ValueType>& values_vec, size_t num_queries) {
  size_t n = keys_vec.size();

  // Build reference index
  ReferenceIndex ref_index;
  ref_index.BulkLoad(keys_vec, values_vec);

  // Build FHIndexRO
  KeyType* keys = new KeyType[n];
  ValueType* values = new ValueType[n];

  parlay::parallel_for(0, n, [&](size_t i) {
    keys[i] = keys_vec[i];
    values[i] = values_vec[i];
  });

  // Sort for FHIndexRO
  auto entries = parlay::tabulate(n, [&](size_t i) -> EntryType {
    return {keys[i], values[i]};
  });
  parlay::integer_sort_inplace(entries,
                               [](EntryType kv) { return kv.first; });
  parlay::parallel_for(0, n, [&](size_t i) {
    keys[i] = entries[i].first;
    values[i] = entries[i].second;
  });

  fh_index_ro::FHIndexRO<KeyType, ValueType> index(kTopLevelThreshold,
                                                   kBlockSize, kFingerSparsity);
  index.BulkLoad(n, keys, values);

  // Test queries
  size_t errors = 0;
  size_t queries_tested = min(num_queries, n);

  for (size_t i = 0; i < queries_tested; i++) {
    size_t idx = i * n / queries_tested;
    KeyType query_key = ref_index.GetKey(idx);

    // Query reference
    ValueType ref_value;
    bool ref_found = ref_index.Find(query_key, ref_value);

    // Query FHIndexRO
    ValueType fh_value;
    bool fh_found;
    index.Find(query_key, fh_value, fh_found);

    if (ref_found != fh_found) {
      errors++;
      if (errors <= 5) {
        cout << "  Find mismatch: key=" << query_key << " ref_found="
             << ref_found << " fh_found=" << fh_found << endl;
      }
    } else if (ref_found && ref_value != fh_value) {
      errors++;
      if (errors <= 5) {
        cout << "  Value mismatch: key=" << query_key << " ref_value="
             << ref_value << " fh_value=" << fh_value << endl;
      }
    }
  }

  delete[] keys;
  delete[] values;

  if (errors > 0) {
    return TestResult(false, "Found " + to_string(errors) + " errors in " +
                                 to_string(queries_tested) + " queries");
  }
  return TestResult(true, "All " + to_string(queries_tested) +
                              " Find queries correct");
}

// Test LowerBound operation
TestResult TestLowerBound(const vector<KeyType>& keys_vec,
                          const vector<ValueType>& values_vec,
                          size_t num_queries) {
  size_t n = keys_vec.size();

  // Build reference index
  ReferenceIndex ref_index;
  ref_index.BulkLoad(keys_vec, values_vec);

  // Build FHIndexRO
  KeyType* keys = new KeyType[n];
  ValueType* values = new ValueType[n];

  parlay::parallel_for(0, n, [&](size_t i) {
    keys[i] = keys_vec[i];
    values[i] = values_vec[i];
  });

  // Sort for FHIndexRO
  auto entries = parlay::tabulate(n, [&](size_t i) -> EntryType {
    return {keys[i], values[i]};
  });
  parlay::integer_sort_inplace(entries,
                               [](EntryType kv) { return kv.first; });
  parlay::parallel_for(0, n, [&](size_t i) {
    keys[i] = entries[i].first;
    values[i] = entries[i].second;
  });

  fh_index_ro::FHIndexRO<KeyType, ValueType> index(kTopLevelThreshold,
                                                   kBlockSize, kFingerSparsity);
  index.BulkLoad(n, keys, values);

  // Test queries - use both existing keys and non-existing keys
  size_t errors = 0;
  size_t queries_tested = min(num_queries, n);

  // Test with existing keys
  for (size_t i = 0; i < queries_tested / 2; i++) {
    size_t idx = i * n / (queries_tested / 2);
    KeyType query_key = ref_index.GetKey(idx);

    size_t ref_pos = ref_index.LowerBound(query_key);
    size_t fh_pos = index.LowerBound(query_key);

    if (ref_pos != fh_pos) {
      errors++;
      if (errors <= 5) {
        cout << "  LowerBound mismatch (existing key): key=" << query_key
             << " ref_pos=" << ref_pos << " fh_pos=" << fh_pos << endl;
      }
    }
  }

  // Test with non-existing keys (between existing keys)
  for (size_t i = 0; i < queries_tested / 2; i++) {
    size_t idx = i * n / (queries_tested / 2);
    if (idx + 1 >= n) continue;

    KeyType key1 = ref_index.GetKey(idx);
    KeyType key2 = ref_index.GetKey(idx + 1);
    if (key2 - key1 <= 1) continue;  // No gap between keys

    KeyType query_key = key1 + (key2 - key1) / 2;  // Key in the middle

    size_t ref_pos = ref_index.LowerBound(query_key);
    size_t fh_pos = index.LowerBound(query_key);

    if (ref_pos != fh_pos) {
      errors++;
      if (errors <= 5) {
        cout << "  LowerBound mismatch (non-existing key): key=" << query_key
             << " ref_pos=" << ref_pos << " fh_pos=" << fh_pos << endl;
      }
    }
  }

  delete[] keys;
  delete[] values;

  if (errors > 0) {
    return TestResult(false, "Found " + to_string(errors) + " errors");
  }
  return TestResult(true, "All LowerBound queries correct");
}

TestResult TestEmptyBulkLoad() {
  fh_index_ro::FHIndexRO<KeyType, ValueType> index(
      kTopLevelThreshold, kBlockSize, kFingerSparsity);

  try {
    index.BulkLoad(0, nullptr, nullptr);
  } catch (const invalid_argument&) {
    return TestResult(true, "Empty input is rejected");
  }
  return TestResult(false, "Empty input was accepted");
}

// Run all tests
bool RunTests(const string& dataset_name, const vector<KeyType>& keys_vec,
              const vector<ValueType>& values_vec, size_t num_queries) {
  cout << "\n========================================" << endl;
  cout << "Testing dataset: " << dataset_name << endl;
  cout << "Dataset size: " << keys_vec.size() << endl;
  cout << "Num queries: " << num_queries << endl;
  cout << "========================================\n" << endl;

  auto result_find = TestFind(keys_vec, values_vec, num_queries);
  PrintTestResult("Test Find", result_find);

  auto result_lb = TestLowerBound(keys_vec, values_vec, num_queries);
  PrintTestResult("Test LowerBound", result_lb);

  auto result_empty = TestEmptyBulkLoad();
  PrintTestResult("Test Empty BulkLoad", result_empty);

  cout << "\n----------------------------------------" << endl;
  const bool passed = result_find.passed && result_lb.passed &&
                      result_empty.passed;
  if (passed) {
    cout << "ALL TESTS PASSED" << endl;
  } else {
    cout << "SOME TESTS FAILED" << endl;
  }
  cout << "----------------------------------------\n" << endl;
  return passed;
}

int main(int argc, char* argv[]) {
  string dataset;
  size_t load_limit = 0;
  size_t num_queries = 1000;
  size_t seed = 42;

  if (argc < 2) {
    cout << "Usage: " << argv[0]
         << " <dataset|GEN> [load_limit] [num_queries] [seed]" << endl;
    cout << "  dataset: path to binary file or 'GEN' for random data" << endl;
    cout << "  load_limit: max number of keys to load (0 = all)" << endl;
    cout << "  num_queries: number of queries to test (default: 1000)" << endl;
    cout << "  seed: random seed for GEN mode (default: 42)" << endl;
    return 1;
  }

  dataset = argv[1];
  if (argc >= 3) load_limit = stoull(argv[2]);
  if (argc >= 4) num_queries = stoull(argv[3]);
  if (argc >= 5) seed = stoull(argv[4]);

  vector<KeyType> keys_vec;
  vector<ValueType> values_vec;

  if (dataset == "GEN") {
    if (load_limit == 0) load_limit = 100000;  // Default size for GEN
    cout << "Generating " << load_limit << " uniform random keys (seed=" << seed
         << ")" << endl;
    keys_vec = fh_index::util::UniformRandomKeys(load_limit, seed, true);  // sorted
  } else {
    cout << "Loading keys from " << dataset << endl;
    keys_vec = fh_index::util::LoadKeysFromBinary(dataset, load_limit, false,
                                                   false, false);
    // Sort the keys
    parlay::integer_sort_inplace(keys_vec);
  }

  // Deduplicate sorted keys
  auto last = unique(keys_vec.begin(), keys_vec.end());
  keys_vec.erase(last, keys_vec.end());

  size_t n = keys_vec.size();
  cout << "After deduplication: " << n << " unique keys" << endl;
  values_vec.resize(n);

  // Generate values as hash of keys
  parlay::parallel_for(0, n,
                       [&](size_t i) { values_vec[i] = parlay::hash64(keys_vec[i]); });

  string dataset_name = (dataset == "GEN")
                            ? "Uniform Random (" + to_string(n) + ")"
                            : dataset + " (" + to_string(n) + ")";

  return RunTests(dataset_name, keys_vec, values_vec, num_queries) ? 0 : 1;
}
