#include <algorithm>
#include <atomic>
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "fh_index/fh_index_rw.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "util/io_utility.h"
#include "util/dataset_generator.h"

using namespace std;
using fh_index::util::Distribution;
using fh_index::util::DistributionName;
using fh_index::util::GenerateSortedKeys;
using fh_index::util::ParseDistribution;

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
constexpr size_t kInternalBlockSize = 16;
constexpr size_t kInternalFingerSparsity = 1;

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

  // Prepare sorted keys and values for FHIndexRW
  vector<KeyType> sorted_keys = keys_vec;
  vector<ValueType> sorted_values = values_vec;

  // Create entries and sort
  vector<EntryType> entries(n);
  for (size_t i = 0; i < n; i++) {
    entries[i] = {keys_vec[i], values_vec[i]};
  }
  sort(entries.begin(), entries.end(),
       [](const auto& a, const auto& b) { return a.first < b.first; });

  for (size_t i = 0; i < n; i++) {
    sorted_keys[i] = entries[i].first;
    sorted_values[i] = entries[i].second;
  }

  // Build FHIndexRW
  fh_index_rw::FHIndexRW<KeyType, ValueType> index(kTopLevelThreshold,
                                                    kBlockSize, kBlockSize * 2, kFingerSparsity,
                                                    kInternalBlockSize, kInternalBlockSize * 2, kInternalFingerSparsity);
  index.BulkLoad(n, sorted_keys.data(), sorted_values.data());

  // Test queries
  size_t errors = 0;
  size_t queries_tested = min(num_queries, n);

  for (size_t i = 0; i < queries_tested; i++) {
    size_t idx = i * n / queries_tested;
    KeyType query_key = ref_index.GetKey(idx);

    // Query reference
    ValueType ref_value;
    bool ref_found = ref_index.Find(query_key, ref_value);

    // Query FHIndexRW
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

  if (errors > 0) {
    return TestResult(false, "Found " + to_string(errors) + " errors in " +
                                 to_string(queries_tested) + " queries");
  }
  return TestResult(true, "All " + to_string(queries_tested) +
                              " Find queries correct");
}

// Test Insert operation
TestResult TestInsert(const vector<KeyType>& keys_vec,
                      const vector<ValueType>& values_vec,
                      size_t num_inserts) {
  size_t n = keys_vec.size();
  if (n < num_inserts * 2) {
    return TestResult(true, "Dataset too small for insert test");
  }

  // Use first half for initial bulk load
  size_t initial_size = n / 2;
  vector<EntryType> entries(initial_size);
  for (size_t i = 0; i < initial_size; i++) {
    entries[i] = {keys_vec[i], values_vec[i]};
  }
  sort(entries.begin(), entries.end(),
       [](const auto& a, const auto& b) { return a.first < b.first; });

  vector<KeyType> sorted_keys(initial_size);
  vector<ValueType> sorted_values(initial_size);
  for (size_t i = 0; i < initial_size; i++) {
    sorted_keys[i] = entries[i].first;
    sorted_values[i] = entries[i].second;
  }

  // Build FHIndexRW
  fh_index_rw::FHIndexRW<KeyType, ValueType> index(kTopLevelThreshold,
                                                    kBlockSize, kBlockSize * 2, kFingerSparsity,
                                                    kInternalBlockSize, kInternalBlockSize * 2, kInternalFingerSparsity);
  index.BulkLoad(initial_size, sorted_keys.data(), sorted_values.data());

  // Insert keys from second half
  size_t insert_errors = 0;
  size_t find_errors = 0;
  size_t inserts_tested = min(num_inserts, n - initial_size);

  for (size_t i = 0; i < inserts_tested; i++) {
    size_t idx = initial_size + i;
    KeyType key = keys_vec[idx];
    ValueType value = values_vec[idx];

    // Insert
    bool inserted = index.Insert(key, value);
    if (!inserted) {
      insert_errors++;
      if (insert_errors <= 5) {
        cout << "  Insert failed for key=" << key << endl;
      }
      continue;
    }

    // Verify the inserted key can be found
    ValueType found_value;
    bool found;
    index.Find(key, found_value, found);

    if (!found) {
      find_errors++;
      if (find_errors <= 5) {
        cout << "  Find after insert failed for key=" << key << endl;
      }
    } else if (found_value != value) {
      find_errors++;
      if (find_errors <= 5) {
        cout << "  Value mismatch after insert: key=" << key
             << " expected=" << value << " got=" << found_value << endl;
      }
    }
  }

  // Test inserting duplicate key (should fail)
  if (initial_size > 0) {
    KeyType dup_key = sorted_keys[0];
    bool dup_inserted = index.Insert(dup_key, 12345);
    if (dup_inserted) {
      return TestResult(false, "Duplicate key insert should have failed");
    }
  }

  if (insert_errors > 0 || find_errors > 0) {
    return TestResult(false, "Insert errors: " + to_string(insert_errors) +
                                 ", Find errors: " + to_string(find_errors));
  }
  return TestResult(true, "All " + to_string(inserts_tested) +
                              " Insert operations correct");
}

// Test Delete operation
TestResult TestDelete(const vector<KeyType>& keys_vec,
                      const vector<ValueType>& values_vec,
                      size_t num_deletes) {
  size_t n = keys_vec.size();

  // Prepare sorted keys and values
  vector<EntryType> entries(n);
  for (size_t i = 0; i < n; i++) {
    entries[i] = {keys_vec[i], values_vec[i]};
  }
  sort(entries.begin(), entries.end(),
       [](const auto& a, const auto& b) { return a.first < b.first; });

  vector<KeyType> sorted_keys(n);
  vector<ValueType> sorted_values(n);
  for (size_t i = 0; i < n; i++) {
    sorted_keys[i] = entries[i].first;
    sorted_values[i] = entries[i].second;
  }

  // Build FHIndexRW
  fh_index_rw::FHIndexRW<KeyType, ValueType> index(kTopLevelThreshold,
                                                    kBlockSize, kBlockSize * 2, kFingerSparsity,
                                                    kInternalBlockSize, kInternalBlockSize * 2, kInternalFingerSparsity);
  index.BulkLoad(n, sorted_keys.data(), sorted_values.data());

  // Delete some keys
  size_t delete_errors = 0;
  size_t verify_errors = 0;
  size_t deletes_tested = min(num_deletes, n);

  for (size_t i = 0; i < deletes_tested; i++) {
    size_t idx = i * n / deletes_tested;
    KeyType key = sorted_keys[idx];

    // Delete
    bool deleted = index.Delete(key);
    if (!deleted) {
      delete_errors++;
      if (delete_errors <= 5) {
        cout << "  Delete failed for key=" << key << endl;
      }
      continue;
    }

    // Verify the deleted key cannot be found
    ValueType found_value;
    bool found;
    index.Find(key, found_value, found);

    if (found) {
      verify_errors++;
      if (verify_errors <= 5) {
        cout << "  Key still found after delete: key=" << key << endl;
      }
    }
  }

  // Test deleting non-existing key (should fail)
  KeyType non_exist_key = sorted_keys[n - 1] + 1000;
  bool non_exist_deleted = index.Delete(non_exist_key);
  if (non_exist_deleted) {
    return TestResult(false, "Non-existing key delete should have failed");
  }

  if (delete_errors > 0 || verify_errors > 0) {
    return TestResult(false, "Delete errors: " + to_string(delete_errors) +
                                 ", Verify errors: " + to_string(verify_errors));
  }
  return TestResult(true, "All " + to_string(deletes_tested) +
                              " Delete operations correct");
}

// Test copy-on-write Update, including parallel updates to distinct keys.
TestResult TestUpdate(const vector<KeyType>& keys_vec,
                      const vector<ValueType>& values_vec,
                      size_t num_updates) {
  size_t n = keys_vec.size();
  if (n == 0) {
    return TestResult(true, "Dataset is empty");
  }

  vector<EntryType> entries(n);
  for (size_t i = 0; i < n; ++i) {
    entries[i] = {keys_vec[i], values_vec[i]};
  }
  sort(entries.begin(), entries.end(),
       [](const auto& a, const auto& b) { return a.first < b.first; });

  vector<KeyType> sorted_keys(n);
  vector<ValueType> sorted_values(n);
  for (size_t i = 0; i < n; ++i) {
    sorted_keys[i] = entries[i].first;
    sorted_values[i] = entries[i].second;
  }

  fh_index_rw::FHIndexRW<KeyType, ValueType> index(
      kTopLevelThreshold, kBlockSize, kBlockSize * 2, kFingerSparsity,
      kInternalBlockSize, kInternalBlockSize * 2,
      kInternalFingerSparsity);
  index.BulkLoad(n, sorted_keys.data(), sorted_values.data());

  size_t updates_tested = min(num_updates, n);
  parlay::parallel_for(0, updates_tested, [&](size_t i) {
    ValueType new_value = sorted_values[i] ^ 0x9e3779b97f4a7c15ULL;
    bool updated = index.Update(sorted_keys[i], new_value);
    assert(updated && "Existing key must be updated");
  });

  size_t errors = 0;
  for (size_t i = 0; i < updates_tested; ++i) {
    ValueType expected = sorted_values[i] ^ 0x9e3779b97f4a7c15ULL;
    ValueType actual = 0;
    bool found = false;
    index.Find(sorted_keys[i], actual, found);
    if (!found || actual != expected) {
      ++errors;
    }
  }

  KeyType missing_key = sorted_keys.back();
  if (missing_key != numeric_limits<KeyType>::max()) {
    ++missing_key;
    if (index.Update(missing_key, 12345)) {
      return TestResult(false, "Updating a missing key should fail");
    }
  }

  if (errors != 0) {
    return TestResult(false, "Found " + to_string(errors) +
                                 " errors after Update");
  }

  // Exercise multiple packed-version rollovers on the same leaf block.  The
  // writer replaces the block before the 4-bit version can wrap in place.
  ValueType rollover_value = 0;
  bool rollover_found = false;
  index.Find(sorted_keys[0], rollover_value, rollover_found);
  if (!rollover_found) {
    return TestResult(false, "Rollover test key is missing");
  }
  for (size_t round = 0; round < 32; ++round) {
    rollover_value += 0x9e3779b97f4a7c15ULL + round;
    if (!index.Update(sorted_keys[0], rollover_value)) {
      return TestResult(false, "Update failed during packed-version rollover");
    }
  }
  ValueType rollover_actual = 0;
  index.Find(sorted_keys[0], rollover_actual, rollover_found);
  if (!rollover_found || rollover_actual != rollover_value) {
    return TestResult(false, "Packed-version rollover produced a wrong value");
  }

  return TestResult(true, "All " + to_string(updates_tested) +
                              " Update operations and packed-version rollovers correct");
}

// Verify the construction contract for RW internal blocks: boundary
// propagation may add one descriptor, but the materialized block must still
// remain within internal_B.
TestResult TestMaterializedInternalBlockBound() {
  constexpr size_t leaf_B = 8;
  constexpr size_t leaf_B_overflow = 16;
  constexpr size_t leaf_F = 1;
  constexpr size_t internal_B = 5;
  constexpr size_t internal_B_overflow = 10;
  constexpr size_t internal_F = 1;
  constexpr size_t root_threshold = 2;

  vector<KeyType> keys;
  vector<ValueType> values;
  keys.reserve(10000);
  values.reserve(10000);

  KeyType key = 0;
  for (size_t i = 0; i < 10000; ++i) {
    keys.push_back(key);
    values.push_back(i);
    key += 1 + (i % 100);
  }

  fh_index_rw::FHIndexRW<KeyType, ValueType> index(
      root_threshold, leaf_B, leaf_B_overflow, leaf_F,
      internal_B, internal_B_overflow, internal_F);
  index.BulkLoad(keys.size(), keys.data(), values.data());

  if (index.GetLevelCount() < 2) {
    return TestResult(false, "Test setup did not produce an internal level");
  }

  size_t block_count = 0;
  size_t max_block_size = 0;
  for (size_t level = 0; level + 1 < index.GetLevelCount(); ++level) {
    auto finger_arrays = index.GetInternalFingerArrays(level);
    for (auto* finger_array : finger_arrays) {
      auto* fingers = finger_array->fingers<fh_index_rw::FingerRW>();
      for (size_t i = 0; i < finger_array->size; ++i) {
        uint64_t data = fingers[i].GetData();
        size_t block_size = fh_index_rw::FingerRW::UnpackSize(data);
        ++block_count;
        max_block_size = max(max_block_size, block_size);
        if (block_size == 0 || block_size > internal_B) {
          return TestResult(
              false, "Materialized internal block size " +
                         to_string(block_size) + " exceeds internal_B=" +
                         to_string(internal_B));
        }
      }
    }
  }

  return TestResult(true, "Checked " + to_string(block_count) +
                              " internal blocks; max size=" +
                              to_string(max_block_size));
}

// Test MarkSubtreeRebuilding and CollectKeyValuePairs
// Verifies no duplicates and no missing entries
TestResult TestMarkAndCollect(const vector<KeyType>& keys_vec,
                               const vector<ValueType>& values_vec) {
  size_t n = keys_vec.size();

  // Prepare sorted keys and values
  vector<EntryType> entries(n);
  for (size_t i = 0; i < n; i++) {
    entries[i] = {keys_vec[i], values_vec[i]};
  }
  sort(entries.begin(), entries.end(),
       [](const auto& a, const auto& b) { return a.first < b.first; });

  vector<KeyType> sorted_keys(n);
  vector<ValueType> sorted_values(n);
  for (size_t i = 0; i < n; i++) {
    sorted_keys[i] = entries[i].first;
    sorted_values[i] = entries[i].second;
  }

  // Build FHIndexRW
  fh_index_rw::FHIndexRW<KeyType, ValueType> index(kTopLevelThreshold,
                                                    kBlockSize, kBlockSize * 2, kFingerSparsity,
                                                    kInternalBlockSize, kInternalBlockSize * 2, kInternalFingerSparsity);
  index.BulkLoad(n, sorted_keys.data(), sorted_values.data());

  // Collect all key-value pairs from all leaf finger arrays
  vector<KeyType> collected_keys;
  vector<ValueType> collected_values;

  const auto& leaf_arrays = index.GetLeafFingerArrays();
  for (size_t i = 0; i < leaf_arrays.size(); i++) {
    auto leaf_fingers = index.MarkSubtreeRebuilding(leaf_arrays[i], 0);
    auto [size, keys, values] = index.CollectKeyValuePairs(leaf_fingers);
    collected_keys.insert(collected_keys.end(), keys, keys + size);
    collected_values.insert(collected_values.end(), values, values + size);
    delete[] keys;
    delete[] values;
  }

  // Check no duplicates: collected keys should all be unique
  vector<KeyType> collected_keys_sorted = collected_keys;
  sort(collected_keys_sorted.begin(), collected_keys_sorted.end());
  auto dup_it = adjacent_find(collected_keys_sorted.begin(), collected_keys_sorted.end());
  if (dup_it != collected_keys_sorted.end()) {
    return TestResult(false, "Found duplicate key: " + to_string(*dup_it));
  }

  // Check no missing: collected count should equal original count
  if (collected_keys.size() != n) {
    return TestResult(false, "Key count mismatch: expected " + to_string(n) +
                             ", got " + to_string(collected_keys.size()));
  }

  // Verify all keys and values match original
  // Create a map from sorted keys for verification
  size_t value_errors = 0;
  for (size_t i = 0; i < collected_keys.size(); i++) {
    KeyType key = collected_keys[i];
    ValueType value = collected_values[i];

    // Find key in original sorted array
    auto it = lower_bound(sorted_keys.begin(), sorted_keys.end(), key);
    if (it == sorted_keys.end() || *it != key) {
      return TestResult(false, "Collected key not in original: " + to_string(key));
    }
    size_t idx = it - sorted_keys.begin();
    if (sorted_values[idx] != value) {
      value_errors++;
    }
  }

  if (value_errors > 0) {
    return TestResult(false, "Value mismatches: " + to_string(value_errors));
  }

  return TestResult(true, "All " + to_string(n) + " key-value pairs collected correctly");
}

// Test MarkSubtreeRebuilding with internal levels
// Tests multi-layer descent and verifies no duplicates/missing
TestResult TestMarkSubtreeWithInternalLevels(const vector<KeyType>& keys_vec,
                                              const vector<ValueType>& values_vec) {
  size_t n = keys_vec.size();

  // Prepare sorted keys and values
  vector<EntryType> entries(n);
  for (size_t i = 0; i < n; i++) {
    entries[i] = {keys_vec[i], values_vec[i]};
  }
  sort(entries.begin(), entries.end(),
       [](const auto& a, const auto& b) { return a.first < b.first; });

  vector<KeyType> sorted_keys(n);
  vector<ValueType> sorted_values(n);
  for (size_t i = 0; i < n; i++) {
    sorted_keys[i] = entries[i].first;
    sorted_values[i] = entries[i].second;
  }

  // Build FHIndexRW with small B to force more segments, and small threshold for internal levels
  size_t small_threshold = 2;
  size_t small_B = 16;
  size_t small_F = 1;
  size_t small_internal_B = 16;
  size_t small_internal_F = 1;
  fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
                                                    small_B, small_B * 2, small_F,
                                                    small_internal_B, small_internal_B * 2, small_internal_F);
  index.BulkLoad(n, sorted_keys.data(), sorted_values.data());

  size_t level_count = index.GetLevelCount();

  // If no internal levels (only leaf), skip this test
  if (level_count <= 1) {
    return TestResult(true, "No internal levels to test (n=" + to_string(n) +
                            ", segments=" + to_string(index.GetLeafFingerArrays().size()) + ")");
  }

  // Test from the highest internal level
  // Get a finger array from the highest internal level
  // level_count - 2 converts to internal level index (0-based)
  const auto& internal_arrays = index.GetInternalFingerArrays(level_count - 2);
  if (internal_arrays.empty()) {
    return TestResult(true, "No internal finger arrays to test");
  }

  // Mark subtree starting from first internal finger array
  // layer_id = level_count - 1 means we're at the highest layer below root
  auto leaf_fingers = index.MarkSubtreeRebuilding(internal_arrays[0], level_count - 1);
  auto [size, keys_ptr, values_ptr] = index.CollectKeyValuePairs(leaf_fingers);

  // Check no duplicates
  vector<KeyType> keys_sorted(keys_ptr, keys_ptr + size);
  sort(keys_sorted.begin(), keys_sorted.end());
  auto dup_it = adjacent_find(keys_sorted.begin(), keys_sorted.end());
  if (dup_it != keys_sorted.end()) {
    delete[] keys_ptr;
    delete[] values_ptr;
    return TestResult(false, "Found duplicate key in multi-level descent: " + to_string(*dup_it));
  }

  // Verify all collected keys exist in original
  for (size_t i = 0; i < size; i++) {
    auto it = lower_bound(sorted_keys.begin(), sorted_keys.end(), keys_ptr[i]);
    if (it == sorted_keys.end() || *it != keys_ptr[i]) {
      delete[] keys_ptr;
      delete[] values_ptr;
      return TestResult(false, "Collected key not in original: " + to_string(keys_ptr[i]));
    }
  }

  size_t collected_size = size;
  delete[] keys_ptr;
  delete[] values_ptr;

  return TestResult(true, "Multi-level descent: " + to_string(collected_size) +
                          " keys collected from " + to_string(level_count) +
                          " levels (including leaf), no duplicates");
}

// Test MarkNextLayer specifically
TestResult TestMarkNextLayer(const vector<KeyType>& keys_vec,
                              const vector<ValueType>& values_vec) {
  size_t n = keys_vec.size();

  // Prepare sorted keys and values
  vector<EntryType> entries(n);
  for (size_t i = 0; i < n; i++) {
    entries[i] = {keys_vec[i], values_vec[i]};
  }
  sort(entries.begin(), entries.end(),
       [](const auto& a, const auto& b) { return a.first < b.first; });

  vector<KeyType> sorted_keys(n);
  vector<ValueType> sorted_values(n);
  for (size_t i = 0; i < n; i++) {
    sorted_keys[i] = entries[i].first;
    sorted_values[i] = entries[i].second;
  }

  // Build FHIndexRW with small B to force more segments, and small threshold for internal levels
  size_t small_threshold = 4;
  size_t small_B = 16;
  size_t small_F = 1;
  size_t small_internal_B = 16;
  size_t small_internal_F = 1;
  fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
                                                    small_B, small_B * 2, small_F,
                                                    small_internal_B, small_internal_B * 2, small_internal_F);
  index.BulkLoad(n, sorted_keys.data(), sorted_values.data());

  size_t level_count = index.GetLevelCount();
  if (level_count <= 1) {
    return TestResult(true, "No internal levels to test MarkNextLayer");
  }

  // Get fingers from highest internal level
  // level_count - 2 converts to internal level index (0-based)
  const auto& internal_arrays = index.GetInternalFingerArrays(level_count - 2);
  if (internal_arrays.empty()) {
    return TestResult(true, "No internal finger arrays");
  }

  // Manually mark first layer and call MarkNextLayer
  using FingerRW = fh_index_rw::FingerRW;
  using FingerArray = fh_index::internal::FingerArray;

  FingerArray* first_array = internal_arrays[0];
  vector<FingerRW> first_layer_fingers;

  FingerRW* fingers = first_array->fingers<FingerRW>();
  size_t finger_count = first_array->size;

  uint64_t last_block_ptr = 0;
  for (size_t i = 0; i < finger_count; i++) {
    fingers[i].SetRebuilding();
    uint64_t data = fingers[i].GetData();
    uint64_t block_ptr = data & FingerRW::kPtrMask;
    if (block_ptr != 0 && block_ptr != last_block_ptr) {
      first_layer_fingers.push_back(fingers[i]);
      last_block_ptr = block_ptr;
    }
  }

  // Call MarkNextLayer
  auto next_layer_fingers = index.MarkNextLayer(first_layer_fingers);

  // Check no duplicates in next layer (by block pointer)
  vector<uint64_t> block_ptrs;
  for (const auto& finger : next_layer_fingers) {
    uint64_t data = finger.GetData();
    uint64_t block_ptr = data & FingerRW::kPtrMask;
    block_ptrs.push_back(block_ptr);
  }

  // Block pointers should be unique (after dedup logic)
  vector<uint64_t> block_ptrs_sorted = block_ptrs;
  sort(block_ptrs_sorted.begin(), block_ptrs_sorted.end());
  auto dup_it = adjacent_find(block_ptrs_sorted.begin(), block_ptrs_sorted.end());
  if (dup_it != block_ptrs_sorted.end()) {
    return TestResult(false, "Found duplicate block pointer in MarkNextLayer output");
  }

  return TestResult(true, "MarkNextLayer: " + to_string(first_layer_fingers.size()) +
                          " input fingers -> " + to_string(next_layer_fingers.size()) +
                          " output fingers, no duplicates");
}

// Test Insert then Delete (round-trip)
TestResult TestInsertDelete(const vector<KeyType>& keys_vec,
                            const vector<ValueType>& values_vec,
                            size_t num_ops) {
  size_t n = keys_vec.size();
  if (n < num_ops * 2) {
    return TestResult(true, "Dataset too small for insert-delete test");
  }

  // Use first half for initial bulk load
  size_t initial_size = n / 2;
  vector<EntryType> entries(initial_size);
  for (size_t i = 0; i < initial_size; i++) {
    entries[i] = {keys_vec[i], values_vec[i]};
  }
  sort(entries.begin(), entries.end(),
       [](const auto& a, const auto& b) { return a.first < b.first; });

  vector<KeyType> sorted_keys(initial_size);
  vector<ValueType> sorted_values(initial_size);
  for (size_t i = 0; i < initial_size; i++) {
    sorted_keys[i] = entries[i].first;
    sorted_values[i] = entries[i].second;
  }

  // Build FHIndexRW
  fh_index_rw::FHIndexRW<KeyType, ValueType> index(kTopLevelThreshold,
                                                    kBlockSize, kBlockSize * 2, kFingerSparsity,
                                                    kInternalBlockSize, kInternalBlockSize * 2, kInternalFingerSparsity);
  index.BulkLoad(initial_size, sorted_keys.data(), sorted_values.data());

  size_t errors = 0;
  size_t ops_tested = min(num_ops, n - initial_size);

  for (size_t i = 0; i < ops_tested; i++) {
    size_t idx = initial_size + i;
    KeyType key = keys_vec[idx];
    ValueType value = values_vec[idx];

    // Insert
    bool inserted = index.Insert(key, value);
    if (!inserted) {
      errors++;
      continue;
    }

    // Verify insert
    ValueType found_value;
    bool found;
    index.Find(key, found_value, found);
    if (!found || found_value != value) {
      errors++;
      continue;
    }

    // Delete
    bool deleted = index.Delete(key);
    if (!deleted) {
      errors++;
      continue;
    }

    // Verify delete
    index.Find(key, found_value, found);
    if (found) {
      errors++;
      continue;
    }
  }

  if (errors > 0) {
    return TestResult(false, "Found " + to_string(errors) +
                                 " errors in insert-delete round-trip");
  }
  return TestResult(true, "All " + to_string(ops_tested) +
                              " Insert-Delete round-trips correct");
}

// Test Find with non-existing keys (should return not found)
TestResult TestFindNotExists(const vector<KeyType>& keys_vec,
                             const vector<ValueType>& values_vec,
                             size_t num_queries) {
  size_t n = keys_vec.size();

  // Prepare sorted keys and values for FHIndexRW
  vector<EntryType> entries(n);
  for (size_t i = 0; i < n; i++) {
    entries[i] = {keys_vec[i], values_vec[i]};
  }
  sort(entries.begin(), entries.end(),
       [](const auto& a, const auto& b) { return a.first < b.first; });

  vector<KeyType> sorted_keys(n);
  vector<ValueType> sorted_values(n);
  for (size_t i = 0; i < n; i++) {
    sorted_keys[i] = entries[i].first;
    sorted_values[i] = entries[i].second;
  }

  // Build FHIndexRW
  fh_index_rw::FHIndexRW<KeyType, ValueType> index(kTopLevelThreshold,
                                                    kBlockSize, kBlockSize * 2, kFingerSparsity,
                                                    kInternalBlockSize, kInternalBlockSize * 2, kInternalFingerSparsity);
  index.BulkLoad(n, sorted_keys.data(), sorted_values.data());

  // Test queries with non-existing keys
  size_t errors = 0;
  size_t queries_tested = 0;

  for (size_t i = 0; i < min(num_queries, n - 1); i++) {
    size_t idx = i * (n - 1) / num_queries;
    if (idx + 1 >= n) continue;

    KeyType key1 = sorted_keys[idx];
    KeyType key2 = sorted_keys[idx + 1];
    if (key2 - key1 <= 1) continue;  // No gap between keys

    KeyType query_key = key1 + (key2 - key1) / 2;  // Key in the middle
    queries_tested++;

    // Query FHIndexRW - should return not found
    ValueType fh_value;
    bool fh_found;
    index.Find(query_key, fh_value, fh_found);

    if (fh_found) {
      errors++;
      if (errors <= 5) {
        cout << "  False positive: key=" << query_key
             << " was found but shouldn't exist" << endl;
      }
    }
  }

  if (queries_tested == 0) {
    return TestResult(true, "No gaps in keys to test non-existing queries");
  }

  if (errors > 0) {
    return TestResult(false, "Found " + to_string(errors) + " false positives in " +
                                 to_string(queries_tested) + " queries");
  }
  return TestResult(true, "All " + to_string(queries_tested) +
                              " non-existing key queries correct");
}

// Test Insert/Delete boundary cases
TestResult TestBoundaryInsertDelete() {
  // Test with small dataset and small block size to hit boundaries easily
  size_t small_B = 16;
  size_t small_B_overflow = 32;
  size_t small_F = 1;
  size_t small_internal_B = 16;
  size_t small_internal_B_overflow = 32;
  size_t small_internal_F = 1;
  size_t small_threshold = 4;

  size_t errors = 0;

  // Test 1: Insert at beginning (key smaller than all existing keys)
  {
    vector<KeyType> keys = {100, 200, 300, 400, 500};
    vector<ValueType> values = {1000, 2000, 3000, 4000, 5000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    // Insert key smaller than all existing
    bool inserted = index.Insert(50, 500);
    if (!inserted) {
      errors++;
      cout << "  Test 1 failed: Insert at beginning failed" << endl;
    } else {
      ValueType found_value;
      bool found;
      index.Find(50, found_value, found);
      if (!found || found_value != 500) {
        errors++;
        cout << "  Test 1 failed: Find after insert at beginning failed" << endl;
      }
    }
  }

  // Test 2: Insert at end (key larger than all existing keys)
  {
    vector<KeyType> keys = {100, 200, 300, 400, 500};
    vector<ValueType> values = {1000, 2000, 3000, 4000, 5000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    // Insert key larger than all existing
    bool inserted = index.Insert(600, 6000);
    if (!inserted) {
      errors++;
      cout << "  Test 2 failed: Insert at end failed" << endl;
    } else {
      ValueType found_value;
      bool found;
      index.Find(600, found_value, found);
      if (!found || found_value != 6000) {
        errors++;
        cout << "  Test 2 failed: Find after insert at end failed" << endl;
      }
    }
  }

  // Test 3: Delete first key
  {
    vector<KeyType> keys = {100, 200, 300, 400, 500};
    vector<ValueType> values = {1000, 2000, 3000, 4000, 5000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    // Delete first key
    bool deleted = index.Delete(100);
    if (!deleted) {
      errors++;
      cout << "  Test 3 failed: Delete first key failed" << endl;
    } else {
      ValueType found_value;
      bool found;
      index.Find(100, found_value, found);
      if (found) {
        errors++;
        cout << "  Test 3 failed: First key still found after delete" << endl;
      }
      // Verify other keys still exist
      index.Find(200, found_value, found);
      if (!found || found_value != 2000) {
        errors++;
        cout << "  Test 3 failed: Other keys affected by delete" << endl;
      }
    }
  }

  // Test 4: Delete last key
  {
    vector<KeyType> keys = {100, 200, 300, 400, 500};
    vector<ValueType> values = {1000, 2000, 3000, 4000, 5000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    // Delete last key
    bool deleted = index.Delete(500);
    if (!deleted) {
      errors++;
      cout << "  Test 4 failed: Delete last key failed" << endl;
    } else {
      ValueType found_value;
      bool found;
      index.Find(500, found_value, found);
      if (found) {
        errors++;
        cout << "  Test 4 failed: Last key still found after delete" << endl;
      }
      // Verify other keys still exist
      index.Find(400, found_value, found);
      if (!found || found_value != 4000) {
        errors++;
        cout << "  Test 4 failed: Other keys affected by delete" << endl;
      }
    }
  }

  // Test 5: Insert key=0 (minimum possible key)
  {
    vector<KeyType> keys = {100, 200, 300};
    vector<ValueType> values = {1000, 2000, 3000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    bool inserted = index.Insert(0, 999);
    if (!inserted) {
      errors++;
      cout << "  Test 5 failed: Insert key=0 failed" << endl;
    } else {
      ValueType found_value;
      bool found;
      index.Find(0, found_value, found);
      if (!found || found_value != 999) {
        errors++;
        cout << "  Test 5 failed: Find key=0 failed" << endl;
      }
    }
  }

  // Test 6: Insert duplicate key (should fail)
  {
    vector<KeyType> keys = {100, 200, 300};
    vector<ValueType> values = {1000, 2000, 3000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    bool inserted = index.Insert(200, 9999);
    if (inserted) {
      errors++;
      cout << "  Test 6 failed: Insert duplicate key should fail" << endl;
    }
  }

  // Test 7: Delete non-existing key (should fail)
  {
    vector<KeyType> keys = {100, 200, 300};
    vector<ValueType> values = {1000, 2000, 3000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    bool deleted = index.Delete(150);
    if (deleted) {
      errors++;
      cout << "  Test 7 failed: Delete non-existing key should fail" << endl;
    }
  }

  // Test 8: Delete same key twice (second should fail)
  {
    vector<KeyType> keys = {100, 200, 300};
    vector<ValueType> values = {1000, 2000, 3000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    bool deleted1 = index.Delete(200);
    if (!deleted1) {
      errors++;
      cout << "  Test 8 failed: First delete should succeed" << endl;
    }
    bool deleted2 = index.Delete(200);
    if (deleted2) {
      errors++;
      cout << "  Test 8 failed: Second delete should fail" << endl;
    }
  }

  // Test 9: Insert then delete then insert same key again
  {
    vector<KeyType> keys = {100, 200, 300};
    vector<ValueType> values = {1000, 2000, 3000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    // Insert new key
    bool inserted1 = index.Insert(150, 1500);
    if (!inserted1) {
      errors++;
      cout << "  Test 9 failed: First insert should succeed" << endl;
    }

    // Delete it
    bool deleted = index.Delete(150);
    if (!deleted) {
      errors++;
      cout << "  Test 9 failed: Delete should succeed" << endl;
    }

    // Insert same key again with different value
    bool inserted2 = index.Insert(150, 1555);
    if (!inserted2) {
      errors++;
      cout << "  Test 9 failed: Re-insert should succeed" << endl;
    } else {
      ValueType found_value;
      bool found;
      index.Find(150, found_value, found);
      if (!found || found_value != 1555) {
        errors++;
        cout << "  Test 9 failed: Re-inserted value incorrect" << endl;
      }
    }
  }

  // Test 10: Sequential inserts filling up a block
  {
    vector<KeyType> keys = {1000};
    vector<ValueType> values = {10000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    // Insert many keys to fill up block(s)
    bool all_ok = true;
    for (KeyType k = 1001; k <= 1000 + small_B * 2; k++) {
      bool inserted = index.Insert(k, k * 10);
      if (!inserted) {
        errors++;
        cout << "  Test 10 failed: Insert key=" << k << " failed" << endl;
        all_ok = false;
        break;
      }
    }

    // Verify all keys can be found
    if (all_ok) {
      for (KeyType k = 1000; k <= 1000 + small_B * 2; k++) {
        ValueType found_value;
        bool found;
        index.Find(k, found_value, found);
        if (!found || found_value != k * 10) {
          errors++;
          cout << "  Test 10 failed: Find key=" << k << " failed after bulk insert" << endl;
          break;
        }
      }
    }
  }

  // Test 11: Sequential deletes emptying a block
  {
    vector<KeyType> keys;
    vector<ValueType> values;
    for (KeyType k = 100; k <= 100 + small_B - 1; k++) {
      keys.push_back(k);
      values.push_back(k * 10);
    }

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    // Delete all keys except one
    for (size_t i = 0; i < keys.size() - 1; i++) {
      bool deleted = index.Delete(keys[i]);
      if (!deleted) {
        errors++;
        cout << "  Test 11 failed: Delete key=" << keys[i] << " failed" << endl;
        break;
      }
    }

    // Last key should still exist
    ValueType found_value;
    bool found;
    index.Find(keys.back(), found_value, found);
    if (!found || found_value != keys.back() * 10) {
      errors++;
      cout << "  Test 11 failed: Last key not found after deleting others" << endl;
    }
  }

  if (errors > 0) {
    return TestResult(false, "Found " + to_string(errors) + " boundary test errors");
  }
  return TestResult(true, "All 11 boundary tests passed");
}

// Test with single element index
TestResult TestSingleElement() {
  size_t small_B = 16;
  size_t small_B_overflow = 32;
  size_t small_F = 1;
  size_t small_internal_B = 16;
  size_t small_internal_B_overflow = 32;
  size_t small_internal_F = 1;
  size_t small_threshold = 4;

  size_t errors = 0;

  // Test 1: Single element index - Find
  {
    vector<KeyType> keys = {500};
    vector<ValueType> values = {5000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    ValueType found_value;
    bool found;
    index.Find(500, found_value, found);
    if (!found || found_value != 5000) {
      errors++;
      cout << "  Single element Find failed" << endl;
    }

    index.Find(499, found_value, found);
    if (found) {
      errors++;
      cout << "  Single element Find (non-existing smaller) should not find" << endl;
    }

    index.Find(501, found_value, found);
    if (found) {
      errors++;
      cout << "  Single element Find (non-existing larger) should not find" << endl;
    }
  }

  // Test 2: Single element index - Insert before
  {
    vector<KeyType> keys = {500};
    vector<ValueType> values = {5000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    bool inserted = index.Insert(100, 1000);
    if (!inserted) {
      errors++;
      cout << "  Single element Insert before failed" << endl;
    } else {
      ValueType found_value;
      bool found;
      index.Find(100, found_value, found);
      if (!found || found_value != 1000) {
        errors++;
        cout << "  Single element Find after Insert before failed" << endl;
      }
    }
  }

  // Test 3: Single element index - Insert after
  {
    vector<KeyType> keys = {500};
    vector<ValueType> values = {5000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    bool inserted = index.Insert(900, 9000);
    if (!inserted) {
      errors++;
      cout << "  Single element Insert after failed" << endl;
    } else {
      ValueType found_value;
      bool found;
      index.Find(900, found_value, found);
      if (!found || found_value != 9000) {
        errors++;
        cout << "  Single element Find after Insert after failed" << endl;
      }
    }
  }

  // Test 4: Single element index - Delete the only element
  {
    vector<KeyType> keys = {500};
    vector<ValueType> values = {5000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    bool deleted = index.Delete(500);
    if (!deleted) {
      errors++;
      cout << "  Single element Delete failed" << endl;
    } else {
      ValueType found_value;
      bool found;
      index.Find(500, found_value, found);
      if (found) {
        errors++;
        cout << "  Single element Find after Delete should not find" << endl;
      }
    }
  }

  if (errors > 0) {
    return TestResult(false, "Found " + to_string(errors) + " single element test errors");
  }
  return TestResult(true, "All 4 single element tests passed");
}

// Test continuous inserts at boundaries (max and min)
TestResult TestContinuousBoundaryInserts() {
  size_t small_B = 16;
  size_t small_B_overflow = 32;
  size_t small_F = 1;
  size_t small_internal_B = 16;
  size_t small_internal_B_overflow = 32;
  size_t small_internal_F = 1;
  size_t small_threshold = 4;

  size_t errors = 0;
  size_t num_inserts = 100;  // Number of inserts at each boundary

  // Test 1: Continuous inserts at max boundary (ascending)
  {
    vector<KeyType> keys = {1000};
    vector<ValueType> values = {10000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    KeyType max_key = 1000;
    for (size_t i = 0; i < num_inserts; i++) {
      KeyType new_key = max_key + 1 + i;  // Always larger than current max
      ValueType new_value = new_key * 10;

      bool inserted = index.Insert(new_key, new_value);
      if (!inserted) {
        errors++;
        cout << "  Test 1 failed: Insert at max boundary failed, key=" << new_key << endl;
        break;
      }

      // Verify the inserted key
      ValueType found_value;
      bool found;
      index.Find(new_key, found_value, found);
      if (!found || found_value != new_value) {
        errors++;
        cout << "  Test 1 failed: Find after max insert failed, key=" << new_key << endl;
        break;
      }
    }

    // Verify all keys can still be found
    for (size_t i = 0; i <= num_inserts; i++) {
      KeyType key = 1000 + i;
      ValueType expected_value = key * 10;
      ValueType found_value;
      bool found;
      index.Find(key, found_value, found);
      if (!found || found_value != expected_value) {
        errors++;
        cout << "  Test 1 failed: Verify all keys failed, key=" << key << endl;
        break;
      }
    }
  }

  // Test 2: Continuous inserts at min boundary (descending)
  {
    vector<KeyType> keys = {10000};
    vector<ValueType> values = {100000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    for (size_t i = 0; i < num_inserts; i++) {
      KeyType new_key = 10000 - 1 - i;  // Always smaller than current min
      ValueType new_value = new_key * 10;

      bool inserted = index.Insert(new_key, new_value);
      if (!inserted) {
        errors++;
        cout << "  Test 2 failed: Insert at min boundary failed, key=" << new_key << endl;
        break;
      }

      // Verify the inserted key
      ValueType found_value;
      bool found;
      index.Find(new_key, found_value, found);
      if (!found || found_value != new_value) {
        errors++;
        cout << "  Test 2 failed: Find after min insert failed, key=" << new_key << endl;
        break;
      }
    }

    // Verify all keys can still be found
    for (size_t i = 0; i <= num_inserts; i++) {
      KeyType key = 10000 - i;
      ValueType expected_value = key * 10;
      ValueType found_value;
      bool found;
      index.Find(key, found_value, found);
      if (!found || found_value != expected_value) {
        errors++;
        cout << "  Test 2 failed: Verify all keys failed, key=" << key << endl;
        break;
      }
    }
  }

  // Test 3: Alternating inserts at both boundaries
  {
    vector<KeyType> keys = {5000};
    vector<ValueType> values = {50000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    KeyType min_key = 5000;
    KeyType max_key = 5000;

    for (size_t i = 0; i < num_inserts; i++) {
      // Insert at max boundary
      KeyType new_max = max_key + 1;
      bool inserted_max = index.Insert(new_max, new_max * 10);
      if (!inserted_max) {
        errors++;
        cout << "  Test 3 failed: Insert at max failed, key=" << new_max << endl;
        break;
      }
      max_key = new_max;

      // Insert at min boundary
      KeyType new_min = min_key - 1;
      bool inserted_min = index.Insert(new_min, new_min * 10);
      if (!inserted_min) {
        errors++;
        cout << "  Test 3 failed: Insert at min failed, key=" << new_min << endl;
        break;
      }
      min_key = new_min;

      // Verify both inserted keys
      ValueType found_value;
      bool found;

      index.Find(new_max, found_value, found);
      if (!found || found_value != new_max * 10) {
        errors++;
        cout << "  Test 3 failed: Find max key failed, key=" << new_max << endl;
        break;
      }

      index.Find(new_min, found_value, found);
      if (!found || found_value != new_min * 10) {
        errors++;
        cout << "  Test 3 failed: Find min key failed, key=" << new_min << endl;
        break;
      }
    }

    // Verify all keys in range [min_key, max_key]
    size_t total_keys = 1 + 2 * num_inserts;  // original + 2 per iteration
    size_t verified = 0;
    for (KeyType k = min_key; k <= max_key; k++) {
      ValueType found_value;
      bool found;
      index.Find(k, found_value, found);
      if (!found || found_value != k * 10) {
        errors++;
        cout << "  Test 3 failed: Final verify failed, key=" << k << endl;
        break;
      }
      verified++;
    }
    if (verified != total_keys) {
      errors++;
      cout << "  Test 3 failed: Expected " << total_keys << " keys, verified " << verified << endl;
    }
  }

  // Test 4: Large scale boundary inserts (stress test)
  {
    size_t large_num_inserts = 500;
    vector<KeyType> keys = {1000000};
    vector<ValueType> values = {10000000};

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(small_threshold,
        small_B, small_B_overflow, small_F,
        small_internal_B, small_internal_B_overflow, small_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    // Insert at max boundary
    for (size_t i = 0; i < large_num_inserts; i++) {
      KeyType new_key = 1000001 + i;
      bool inserted = index.Insert(new_key, new_key * 10);
      if (!inserted) {
        errors++;
        cout << "  Test 4 failed: Large scale max insert failed at i=" << i << endl;
        break;
      }
    }

    // Insert at min boundary
    for (size_t i = 0; i < large_num_inserts; i++) {
      KeyType new_key = 999999 - i;
      bool inserted = index.Insert(new_key, new_key * 10);
      if (!inserted) {
        errors++;
        cout << "  Test 4 failed: Large scale min insert failed at i=" << i << endl;
        break;
      }
    }

    // Spot check some keys
    vector<KeyType> check_keys = {
        999999 - large_num_inserts + 1,  // min
        999999,                            // near min
        1000000,                           // original
        1000001,                           // near max
        1000000 + large_num_inserts        // max
    };

    for (KeyType k : check_keys) {
      ValueType found_value;
      bool found;
      index.Find(k, found_value, found);
      if (!found || found_value != k * 10) {
        errors++;
        cout << "  Test 4 failed: Spot check failed, key=" << k << endl;
      }
    }
  }

  if (errors > 0) {
    return TestResult(false, "Found " + to_string(errors) + " continuous boundary insert errors");
  }
  return TestResult(true, "All 4 continuous boundary insert tests passed (" +
                          to_string(num_inserts) + " inserts each, plus 500 stress test)");
}

// Test that actually triggers rebuild by causing internal block overflow
TestResult TestRebuildTrigger() {
  // The min-max segmentation algorithm creates multiple segments based on data distribution.
  // With sorted uniform data, it typically creates only 1 segment.
  // To force multiple segments, we need non-uniform or larger data.

  size_t errors = 0;
  size_t height_change_count = 0;

  // Test: Use large dataset with very small thresholds to force multi-level tree
  // and then insert to trigger overflow
  {
    // Parameters designed to create multi-level tree
    size_t tiny_B = 8;
    size_t tiny_B_overflow = 16;
    size_t tiny_F = 1;
    // B=9 preserves the former fitting capacity of 8 while reserving one
    // materialization slot for a propagated descriptor.
    size_t tiny_internal_B = 9;
    size_t tiny_internal_B_overflow = 18;
    size_t tiny_internal_F = 1;
    size_t tiny_threshold = 4;  // threshold for root level FGD count

    // Create large dataset to force multiple segments
    // The min-max algorithm should create segments based on error bounds
    size_t initial_size = 10000;
    vector<KeyType> keys;
    vector<ValueType> values;

    // Use non-uniform distribution (exponential gaps) to force segmentation
    KeyType current_key = 0;
    for (size_t i = 0; i < initial_size; i++) {
      keys.push_back(current_key);
      values.push_back(current_key * 10);
      // Exponential-like gaps to create non-uniform distribution
      current_key += 1 + (i % 100);
    }

    fh_index_rw::FHIndexRW<KeyType, ValueType, true> index(tiny_threshold,
        tiny_B, tiny_B_overflow, tiny_F,
        tiny_internal_B, tiny_internal_B_overflow, tiny_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    auto initial_root = index.GetRootMetadata();
    size_t initial_level_count = initial_root.height;
    cout << "  Test 1: Initial level_count: " << initial_level_count << endl;

    // Insert keys that will cause blocks to grow
    size_t num_inserts = 5000;
    size_t inserted_count = 0;
    for (size_t i = 0; i < num_inserts; i++) {
      // Insert keys in gaps
      KeyType new_key = keys[i % keys.size()] + 1;

      // Check if key already exists
      ValueType found_value;
      bool found;
      index.Find(new_key, found_value, found);
      if (found) continue;

      bool inserted = index.Insert(new_key, new_key * 10);
      if (inserted) {
        inserted_count++;
        // Verify
        index.Find(new_key, found_value, found);
        if (!found) {
          errors++;
          cout << "  Find after insert failed, key=" << new_key << endl;
          break;
        }
      }
    }

    const auto subtree_kind =
        static_cast<size_t>(fh_index_rw::RebuildKind::kSubtree);
    // Fill gaps in this already multi-level tree until an ordinary subtree
    // rebuild occurs. Uniform keys in Test 2 can stay single-level even after
    // dense insertions, exercising only top-level and root rebuilds.
    for (size_t i = 0;
         i + 1 < keys.size() && i < 1000 &&
             index.GetRetrainStats().by_kind[subtree_kind].count == 0;
         ++i) {
      for (KeyType candidate = keys[i] + 1;
           candidate < keys[i + 1]; ++candidate) {
        if (index.Insert(candidate, candidate * 10)) ++inserted_count;
        if (index.GetRetrainStats().by_kind[subtree_kind].count != 0) break;
      }
    }
    const auto stats = index.GetRetrainStats();
    cout << "  Classified ordinary subtree rebuilds: "
         << stats.by_kind[subtree_kind].count << endl;
    if (stats.by_kind[subtree_kind].count == 0) {
      return TestResult(false, "Multi-level insertions did not record ordinary subtree rebuilds");
    }

    auto final_root = index.GetRootMetadata();
    size_t final_level_count = final_root.height;
    cout << "  Test 1: Final level_count: " << final_level_count
         << " (inserted " << inserted_count << " keys)" << endl;

    if (final_level_count != initial_level_count) {
      height_change_count++;
      cout << "  level_count changed from " << initial_level_count
           << " to " << final_level_count << endl;
    }

    // Spot check some original keys
    for (size_t i = 0; i < min(initial_size, (size_t)100); i += 10) {
      KeyType key = keys[i];
      ValueType expected_value = key * 10;
      ValueType found_value;
      bool found;
      index.Find(key, found_value, found);
      if (!found || found_value != expected_value) {
        errors++;
        cout << "  Original key lost after inserts, key=" << key << endl;
        break;
      }
    }
  }

  // Test 2: Very small B_overflow to guarantee overflow detection
  {
    size_t micro_B = 4;
    size_t micro_B_overflow = 8;  // Very small
    size_t micro_F = 1;
    // B=5 leaves a fitting bound of 4 after reserving one propagation slot.
    size_t micro_internal_B = 5;
    size_t micro_internal_B_overflow = 10;
    size_t micro_internal_F = 1;
    size_t micro_threshold = 2;

    // Start with small dataset
    size_t initial_size = 1000;
    vector<KeyType> keys;
    vector<ValueType> values;
    for (size_t i = 0; i < initial_size; i++) {
      keys.push_back(i * 10);
      values.push_back(i * 100);
    }

    fh_index_rw::FHIndexRW<KeyType, ValueType> index(micro_threshold,
        micro_B, micro_B_overflow, micro_F,
        micro_internal_B, micro_internal_B_overflow, micro_internal_F);
    index.BulkLoad(keys.size(), keys.data(), values.data());

    auto initial_root = index.GetRootMetadata();
    size_t initial_level_count = initial_root.height;
    cout << "  Test 2: Initial level_count: " << initial_level_count << endl;

    // Insert many keys in gaps
    size_t inserted_count = 0;
    for (size_t i = 0; i < initial_size; i++) {
      for (int gap = 1; gap < 10; gap++) {
        KeyType new_key = i * 10 + gap;
        ValueType found_value;
        bool found;
        index.Find(new_key, found_value, found);
        if (!found) {
          bool inserted = index.Insert(new_key, new_key * 10);
          if (inserted) inserted_count++;
        }
      }
    }

    auto final_root = index.GetRootMetadata();
    size_t final_level_count = final_root.height;
    cout << "  Test 2: Final level_count: " << final_level_count
         << " (inserted " << inserted_count << " keys)" << endl;

    if (final_level_count != initial_level_count) {
      height_change_count++;
    }

    // Verify some keys
    for (size_t i = 0; i < min(initial_size, (size_t)100); i += 10) {
      KeyType key = i * 10;
      ValueType found_value;
      bool found;
      index.Find(key, found_value, found);
      if (!found) {
        errors++;
        cout << "  Test 2: Original key lost, key=" << key << endl;
        break;
      }
    }
  }

  if (errors > 0) {
    return TestResult(false, "Found " + to_string(errors) + " errors in rebuild trigger test");
  }

  string msg = "Rebuild trigger test completed, height_changes=" +
               to_string(height_change_count);
  return TestResult(true, msg);
}

// B=80 would ordinarily select an overflow threshold of 160, but the packed
// finger state can represent sizes and capacities only through 127. Verify
// that the capped threshold remains usable and that crossing it rebuilds the
// leaf subtree without losing records.
TestResult TestPackedFingerMaximumOverflow() {
  using Index = fh_index_rw::FHIndexRW<KeyType, ValueType, true>;

  constexpr size_t leaf_B = 80;
  constexpr size_t leaf_overflow = fh_index_rw::FingerRW::kSizeMask;
  constexpr size_t initial_size = 80;
  constexpr size_t inserted_size = 48;

  vector<KeyType> keys(initial_size);
  vector<ValueType> values(initial_size);
  for (size_t i = 0; i < initial_size; ++i) {
    keys[i] = static_cast<KeyType>(2 * i);
    values[i] = static_cast<ValueType>(keys[i] * 10);
  }

  Index index(/*top_level_threshold=*/256, leaf_B, leaf_overflow,
              fh_index::internal::Fraction(1, 3),
              /*internal_B=*/16, /*internal_B_overflow=*/32,
              fh_index::internal::Fraction(1, 1));
  index.BulkLoad(keys.size(), keys.data(), values.data());
  index.ResetRetrainStats();

  for (size_t i = 0; i < inserted_size; ++i) {
    const KeyType key = static_cast<KeyType>(2 * i + 1);
    if (!index.Insert(key, key * 10)) {
      return TestResult(false, "Insert failed while crossing size 127");
    }
  }

  const auto stats = index.GetRetrainStats();
  if (stats.count == 0) {
    return TestResult(false, "Crossing size 127 did not trigger a rebuild");
  }

  constexpr size_t total_size = initial_size + inserted_size;
  vector<KeyType> expected_keys = keys;
  for (size_t i = 0; i < inserted_size; ++i) {
    expected_keys.push_back(static_cast<KeyType>(2 * i + 1));
  }
  sort(expected_keys.begin(), expected_keys.end());

  vector<pair<KeyType, ValueType>> scan_result(total_size);
  const size_t scanned =
      index.Scan(/*lower=*/0, total_size, scan_result.data());
  if (scanned != total_size) {
    return TestResult(false, "Post-rebuild scan returned the wrong size");
  }
  for (size_t i = 0; i < total_size; ++i) {
    const KeyType expected_key = expected_keys[i];
    if (scan_result[i].first != expected_key ||
        scan_result[i].second != expected_key * 10) {
      return TestResult(false, "Post-rebuild scan returned incorrect data");
    }
  }

  constexpr KeyType deleted_key = 95;
  if (!index.Delete(deleted_key)) {
    return TestResult(false, "Post-rebuild delete failed");
  }
  ValueType value = 0;
  bool found = true;
  index.Find(deleted_key, value, found);
  if (found) {
    return TestResult(false, "Deleted post-rebuild key remains visible");
  }

  return TestResult(true, "B=80 rebuilt safely at the packed size limit");
}

// Concurrent readers must observe one self-consistent immutable root snapshot
// while complete-tree rebuilds publish newer snapshots.
TestResult TestAtomicRootSnapshotPublication() {
  using Index = fh_index_rw::FHIndexRW<KeyType, ValueType, true>;

  constexpr size_t n = 4000;
  vector<KeyType> keys(n);
  vector<ValueType> values(n);
  KeyType key = 0;
  for (size_t i = 0; i < n; i++) {
    key += 1 + (i % 97);
    keys[i] = key;
    values[i] = key * 17 + 3;
  }

  Index index(/*top_level_threshold=*/4,
              /*B=*/8, /*B_overflow=*/16, /*F=*/1,
              /*internal_B=*/9, /*internal_B_overflow=*/18,
              /*internal_F=*/1);
  index.BulkLoad(n, keys.data(), values.data());

  const auto before = index.GetRootMetadata();
  if (before.root_size == 0 || before.height == 0) {
    return TestResult(false, "Initial root metadata is invalid");
  }

  atomic<size_t> successful_rebuilds{0};
  atomic<size_t> read_errors{0};
  atomic<size_t> update_errors{0};

  constexpr size_t kTasks = 64;
  constexpr ValueType kUpdateMask = 0x9e3779b97f4a7c15ULL;
  parlay::parallel_for(0, kTasks, [&](size_t task_id) {
    if (task_id % 8 == 0) {
      auto result = index.RebuildRoot();
      if (result == Index::RebuildResult::kSuccess) {
        successful_rebuilds.fetch_add(1, memory_order_relaxed);
      }
      return;
    }

    if (task_id % 8 == 1) {
      for (size_t i = task_id; i < n; i += kTasks) {
        if (!index.Update(keys[i], values[i] ^ kUpdateMask)) {
          update_errors.fetch_add(1, memory_order_relaxed);
        }
      }
      return;
    }

    for (size_t i = task_id; i < n; i += kTasks) {
      ValueType actual = 0;
      bool found = false;
      index.Find(keys[i], actual, found);
      if (!found || actual != values[i]) {
        read_errors.fetch_add(1, memory_order_relaxed);
      }
    }
  });

  const auto after = index.GetRootMetadata();
  const size_t successes = successful_rebuilds.load(memory_order_relaxed);
  if (successes == 0) {
    return TestResult(false, "No root rebuild published a snapshot");
  }
  if (read_errors.load(memory_order_relaxed) != 0) {
    return TestResult(false, "Concurrent readers observed missing or wrong values");
  }
  if (update_errors.load(memory_order_relaxed) != 0) {
    return TestResult(false, "Concurrent updates failed");
  }
  if (after.root_size == 0 || after.height == 0) {
    return TestResult(false, "Published root metadata is invalid");
  }

  const auto stats = index.GetRetrainStats();
  const auto root_kind = static_cast<size_t>(fh_index_rw::RebuildKind::kRoot);
  if (stats.count != successes || stats.by_kind[root_kind].count != successes) {
    return TestResult(false, "Successful root rebuilds were not classified as root");
  }

  for (size_t i = 0; i < n; i++) {
    ValueType actual = 0;
    bool found = false;
    index.Find(keys[i], actual, found);
    const bool was_updated = (i % kTasks) % 8 == 1;
    const ValueType expected =
        was_updated ? (values[i] ^ kUpdateMask) : values[i];
    if (!found || actual != expected) {
      return TestResult(false, "Post-rebuild verification lost key " +
                                   to_string(keys[i]));
    }
  }

  return TestResult(
      true, to_string(successes) +
                " root snapshots published with concurrent readers and updates");
}

// Scan must cross leaf and FGD boundaries without restarting from the root.
TestResult TestScanAcrossBuckets() {
  using Index = fh_index_rw::FHIndexRW<KeyType, ValueType>;

  constexpr size_t n = 4096;
  vector<KeyType> keys(n);
  vector<ValueType> values(n);
  for (size_t i = 0; i < n; ++i) {
    keys[i] = i * 4 + 2;
    values[i] = keys[i] * 19 + 7;
  }

  Index index(/*top_level_threshold=*/8,
              /*B=*/8, /*B_overflow=*/16, /*F=*/1,
              /*internal_B=*/9, /*internal_B_overflow=*/18,
              /*internal_F=*/1);
  index.BulkLoad(n, keys.data(), values.data());

  struct ScanCase {
    KeyType lower;
    size_t length;
  };
  const ScanCase cases[] = {
      {keys.front(), 1},
      {keys[37] + 1, 127},
      {keys[511], 513},
      {keys[n - 10], 50},
      {keys.back() + 1, 32},
  };

  vector<EntryType> result(n);
  for (const ScanCase& scan_case : cases) {
    const size_t begin = static_cast<size_t>(
        lower_bound(keys.begin(), keys.end(), scan_case.lower) - keys.begin());
    const size_t expected_count = min(scan_case.length, n - begin);
    const size_t actual_count =
        index.Scan(scan_case.lower, scan_case.length, result.data());
    if (actual_count != expected_count) {
      return TestResult(false, "Scan returned an incorrect result count");
    }
    for (size_t i = 0; i < actual_count; ++i) {
      if (result[i].first != keys[begin + i] ||
          result[i].second != values[begin + i]) {
        return TestResult(false, "Scan returned an incorrect key-value pair");
      }
    }
  }

  return TestResult(true, "cross-bucket scans match lower_bound semantics");
}

// In-place insert/delete shifts may overlap optimistic scans. The scan accepts
// a copied bucket only when its finger remains unchanged. Writers own disjoint
// keys; readers require each accepted bucket snapshot to remain sorted and
// internally valid.
TestResult TestConcurrentOptimisticScan() {
  using Index = fh_index_rw::FHIndexRW<KeyType, ValueType>;

  constexpr size_t n = 4096;
  vector<KeyType> keys(n);
  vector<ValueType> values(n);
  for (size_t i = 0; i < n; ++i) {
    keys[i] = i * 4 + 2;
    values[i] = keys[i] * 19 + 7;
  }

  Index index(/*top_level_threshold=*/8,
              /*B=*/8, /*B_overflow=*/16, /*F=*/1,
              /*internal_B=*/9, /*internal_B_overflow=*/18,
              /*internal_F=*/1);
  index.BulkLoad(n, keys.data(), values.data());

  atomic<size_t> scan_errors{0};
  atomic<size_t> mutation_errors{0};
  constexpr size_t kTasks = 32;
  constexpr size_t kWriters = 4;
  constexpr size_t kIterations = 4000;
  constexpr size_t kScanLength = 96;

  parlay::parallel_for(0, kTasks, [&](size_t task_id) {
    if (task_id < kWriters) {
      for (size_t round = 0; round < kIterations; ++round) {
        const size_t key_index =
            128 + task_id * 256 + (round % 128);
        const KeyType key = keys[key_index];
        if (!index.Delete(key)) {
          mutation_errors.fetch_add(1, memory_order_relaxed);
          continue;
        }
        if (!index.Insert(key, values[key_index])) {
          mutation_errors.fetch_add(1, memory_order_relaxed);
        }
      }
      return;
    }

    vector<EntryType> result(kScanLength);
    for (size_t round = 0; round < kIterations; ++round) {
      const size_t start_index =
          (task_id * 131 + round * 17) % (n - kScanLength - 1);
      const KeyType lower = keys[start_index] + (round & 1);
      const size_t count = index.Scan(lower, kScanLength, result.data());
      if (count == 0 || count > kScanLength || result[0].first < lower) {
        scan_errors.fetch_add(1, memory_order_relaxed);
        continue;
      }
      for (size_t i = 0; i < count; ++i) {
        const KeyType key = result[i].first;
        if (result[i].second != key * 19 + 7 ||
            (i != 0 && result[i - 1].first >= key)) {
          scan_errors.fetch_add(1, memory_order_relaxed);
          break;
        }
      }
    }
  });

  if (mutation_errors.load(memory_order_relaxed) != 0) {
    return TestResult(false, "Concurrent scan writers failed");
  }
  if (scan_errors.load(memory_order_relaxed) != 0) {
    return TestResult(false, "Optimistic scan observed an invalid snapshot");
  }
  return TestResult(true,
                    "optimistic scans remained valid during in-place shifts");
}

// A root produced exactly at the construction threshold is valid and must not
// be rebuilt forever before every update.
TestResult TestRootThresholdEquality() {
  using Index = fh_index_rw::FHIndexRW<KeyType, ValueType>;

  constexpr size_t n = 1000;
  vector<KeyType> keys(n);
  vector<ValueType> values(n);
  for (size_t i = 0; i < n; i++) {
    keys[i] = i * 10;
    values[i] = keys[i] + 7;
  }

  Index index(/*top_level_threshold=*/1,
              /*B=*/8, /*B_overflow=*/16, /*F=*/1,
              /*internal_B=*/9, /*internal_B_overflow=*/18,
              /*internal_F=*/1);
  index.BulkLoad(n, keys.data(), values.data());

  const auto root = index.GetRootMetadata();
  if (root.root_size != 1) {
    return TestResult(false, "Test setup did not construct a full root");
  }

  constexpr KeyType inserted_key = 101;
  constexpr ValueType inserted_value = 1234567;
  if (!index.Insert(inserted_key, inserted_value)) {
    return TestResult(false, "Insert failed at root threshold equality");
  }

  ValueType actual = 0;
  bool found = false;
  index.Find(inserted_key, actual, found);
  if (!found || actual != inserted_value) {
    return TestResult(false, "Inserted value is missing at root threshold equality");
  }

  return TestResult(true, "root_size == threshold accepts updates");
}

// Rebuilding a leaf subtree stored directly in the root block must preserve
// the untouched root descriptors and their FingerArrays.
TestResult TestTopLevelRebuildStructuralSharing() {
  using Index = fh_index_rw::FHIndexRW<KeyType, ValueType, true>;
  using FingerArray = fh_index::internal::FingerArray;

  constexpr size_t n = 4000;
  vector<KeyType> keys(n);
  vector<ValueType> values(n);
  KeyType key = 0;
  for (size_t i = 0; i < n; i++) {
    keys[i] = key;
    values[i] = key * 13 + 5;
    key += 1 + (i % 97);
  }

  // A large root threshold keeps all model descriptors in one leaf-level
  // hierarchy, so leaf overflow exercises RebuildTopLevel directly.
  Index index(/*top_level_threshold=*/1024,
              /*B=*/8, /*B_overflow=*/16, /*F=*/1,
              /*internal_B=*/9, /*internal_B_overflow=*/18,
              /*internal_F=*/1);
  index.BulkLoad(n, keys.data(), values.data());

  const auto before_root = index.GetRootMetadata();
  vector<FingerArray*> before_arrays = index.GetLeafFingerArrays();
  if (before_root.height != 1 || before_arrays.size() < 2) {
    return TestResult(false, "Test setup did not produce multiple root descriptors");
  }
  unordered_set<FingerArray*> before_set(
      before_arrays.begin(), before_arrays.end());

  vector<pair<KeyType, ValueType>> inserted;
  bool rebuilt = false;
  for (size_t i = 0; i + 1 < n && !rebuilt; i++) {
    for (KeyType candidate = keys[i] + 1;
         candidate < keys[i + 1]; candidate++) {
      ValueType candidate_value = candidate * 13 + 5;
      if (index.Insert(candidate, candidate_value)) {
        inserted.push_back({candidate, candidate_value});
      }
      for (FingerArray* finger_array : index.GetLeafFingerArrays()) {
        if (before_set.count(finger_array) == 0) {
          rebuilt = true;
          break;
        }
      }
      if (rebuilt) break;
    }
  }
  if (!rebuilt) {
    return TestResult(false, "Top-level subtree rebuild was not triggered");
  }

  const auto stats = index.GetRetrainStats();
  const auto top_kind =
      static_cast<size_t>(fh_index_rw::RebuildKind::kTopLevelSubtree);
  if (stats.count == 0 || stats.by_kind[top_kind].count != stats.count) {
    return TestResult(false, "Top-level rebuilds were not classified as top_level_subtree");
  }

  const auto after_root = index.GetRootMetadata();
  vector<FingerArray*> after_arrays = index.GetLeafFingerArrays();
  size_t shared_count = 0;
  for (FingerArray* finger_array : after_arrays) {
    shared_count += before_set.count(finger_array);
  }

  if (after_root.height != before_root.height) {
    return TestResult(false, "Top-level rebuild changed the hierarchy height");
  }
  if (shared_count + 1 != before_arrays.size()) {
    return TestResult(
        false, "Top-level rebuild did not replace exactly one root subtree");
  }

  for (size_t i = 0; i < n; i++) {
    ValueType actual = 0;
    bool found = false;
    index.Find(keys[i], actual, found);
    if (!found || actual != values[i]) {
      return TestResult(false, "Original key lost after top-level rebuild");
    }
  }

  for (const auto& [inserted_key, expected_value] : inserted) {
    ValueType actual = 0;
    bool found = false;
    index.Find(inserted_key, actual, found);
    if (!found || actual != expected_value) {
      return TestResult(false, "Inserted key lost after top-level rebuild");
    }
  }

  return TestResult(
      true, to_string(shared_count) + " of " +
                to_string(before_arrays.size()) +
                " pre-existing root subtrees were shared");
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
  PrintTestResult("Test Find (existing keys)", result_find);

  auto result_not_exists = TestFindNotExists(keys_vec, values_vec, num_queries);
  PrintTestResult("Test Find (non-existing keys)", result_not_exists);

  auto result_insert = TestInsert(keys_vec, values_vec, num_queries);
  PrintTestResult("Test Insert", result_insert);

  auto result_delete = TestDelete(keys_vec, values_vec, num_queries);
  PrintTestResult("Test Delete", result_delete);

  auto result_update = TestUpdate(keys_vec, values_vec, num_queries);
  PrintTestResult("Test Update", result_update);

  auto result_internal_bound = TestMaterializedInternalBlockBound();
  PrintTestResult("Test Materialized Internal Block Bound",
                  result_internal_bound);

  auto result_insert_delete = TestInsertDelete(keys_vec, values_vec, num_queries);
  PrintTestResult("Test Insert-Delete round-trip", result_insert_delete);

  auto result_mark_collect = TestMarkAndCollect(keys_vec, values_vec);
  PrintTestResult("Test MarkSubtreeRebuilding + CollectKeyValuePairs", result_mark_collect);

  auto result_mark_internal = TestMarkSubtreeWithInternalLevels(keys_vec, values_vec);
  PrintTestResult("Test MarkSubtreeRebuilding with internal levels", result_mark_internal);

  auto result_mark_next = TestMarkNextLayer(keys_vec, values_vec);
  PrintTestResult("Test MarkNextLayer", result_mark_next);

  auto result_boundary = TestBoundaryInsertDelete();
  PrintTestResult("Test Boundary Insert/Delete", result_boundary);

  auto result_single = TestSingleElement();
  PrintTestResult("Test Single Element", result_single);

  auto result_continuous = TestContinuousBoundaryInserts();
  PrintTestResult("Test Continuous Boundary Inserts", result_continuous);

  auto result_rebuild = TestRebuildTrigger();
  PrintTestResult("Test Rebuild Trigger", result_rebuild);

  auto result_packed_overflow = TestPackedFingerMaximumOverflow();
  PrintTestResult("Test Packed Finger Maximum Overflow",
                  result_packed_overflow);

  auto result_root_snapshot = TestAtomicRootSnapshotPublication();
  PrintTestResult("Test Atomic Root Snapshot Publication",
                  result_root_snapshot);

  auto result_scan = TestScanAcrossBuckets();
  PrintTestResult("Test Scan Across Buckets", result_scan);

  auto result_concurrent_scan = TestConcurrentOptimisticScan();
  PrintTestResult("Test Concurrent Optimistic Scan", result_concurrent_scan);

  auto result_root_threshold = TestRootThresholdEquality();
  PrintTestResult("Test Root Threshold Equality", result_root_threshold);

  auto result_top_level_sharing = TestTopLevelRebuildStructuralSharing();
  PrintTestResult("Test Top-Level Rebuild Structural Sharing",
                  result_top_level_sharing);

  cout << "\n----------------------------------------" << endl;
  const bool passed = result_find.passed && result_not_exists.passed &&
      result_insert.passed && result_delete.passed && result_insert_delete.passed &&
      result_update.passed && result_internal_bound.passed &&
      result_mark_collect.passed && result_mark_internal.passed && result_mark_next.passed &&
      result_boundary.passed && result_single.passed && result_continuous.passed &&
      result_rebuild.passed && result_packed_overflow.passed &&
      result_root_snapshot.passed &&
      result_scan.passed && result_concurrent_scan.passed &&
      result_root_threshold.passed && result_top_level_sharing.passed;
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

  // Check if dataset is a distribution name
  Distribution dist = ParseDistribution(dataset);
  bool is_distribution = (dataset == "GEN" || dataset == "GAPPED" ||
                          dataset == "UNIFORM" || dataset == "EXPONENTIAL" ||
                          dataset == "NORMAL" || dataset == "CLUSTERED" ||
                          dataset == "DENSE" || dataset == "SPARSE" ||
                          dataset == "MIXED");

  if (is_distribution) {
    if (load_limit == 0) load_limit = 100000;  // Default size
    cout << "Generating " << load_limit << " keys with distribution: "
         << DistributionName(dist) << " (seed=" << seed << ")" << endl;
    keys_vec = GenerateSortedKeys(load_limit, dist, seed);
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

  string dataset_name;
  if (is_distribution) {
    dataset_name = string(DistributionName(dist)) + " (" + to_string(n) + ")";
  } else {
    dataset_name = dataset + " (" + to_string(n) + ")";
  }

  return RunTests(dataset_name, keys_vec, values_vec, num_queries) ? 0 : 1;
}
