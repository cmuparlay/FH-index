// Detailed Insert Performance Breakdown Benchmark
// Measures time spent in each step of the Insert operation

#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include "fh_index/fh_index_rw.h"
#include "fh_index_rw/storage/kv_layer_rw.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

using namespace std;
using namespace fh_index_rw;

using KeyType = uint64_t;
using ValueType = uint64_t;

// High-resolution timer
class Timer {
 public:
  void start() { start_ = chrono::high_resolution_clock::now(); }
  uint64_t elapsed_ns() {
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::nanoseconds>(end - start_).count();
  }
  double elapsed_ms() { return elapsed_ns() / 1e6; }

 private:
  chrono::high_resolution_clock::time_point start_;
};

// Accumulator for timing statistics (thread-safe)
struct TimingStats {
  atomic<uint64_t> total_ns{0};
  atomic<uint64_t> count{0};

  void add(uint64_t ns) {
    total_ns.fetch_add(ns, memory_order_relaxed);
    count.fetch_add(1, memory_order_relaxed);
  }

  void reset() {
    total_ns.store(0, memory_order_relaxed);
    count.store(0, memory_order_relaxed);
  }

  double avg_ns() const {
    uint64_t c = count.load();
    return c > 0 ? static_cast<double>(total_ns.load()) / c : 0;
  }

  double total_ms() const {
    return static_cast<double>(total_ns.load()) / 1e6;
  }
};

// Benchmark: measure individual operation components in isolation
void bench_individual_components(size_t num_ops) {
  cout << "\n========================================" << endl;
  cout << "Individual Component Benchmarks" << endl;
  cout << "Num ops: " << num_ops << endl;
  cout << "========================================" << endl;

  // Test 1: Block allocation speed
  {
    cout << "\n--- Block Allocation ---" << endl;
    Timer timer;
    vector<Block<KeyType, ValueType>*> blocks(num_ops);

    timer.start();
    for (size_t i = 0; i < num_ops; i++) {
      auto [block, cap] = Block<KeyType, ValueType>::CreateWithCapacity(4);
      blocks[i] = block;
    }
    double alloc_time = timer.elapsed_ms();
    cout << "Sequential allocation: " << alloc_time << " ms, "
         << (num_ops / alloc_time * 1000 / 1e6) << " Mops/sec, "
         << (alloc_time / num_ops * 1e6) << " ns/alloc" << endl;

    // Cleanup
    for (auto* b : blocks) {
      Block<KeyType, ValueType>::Destroy(b, 4);
    }
  }

  // Test 1b: Parallel block allocation
  {
    vector<Block<KeyType, ValueType>*> blocks(num_ops);
    Timer timer;

    timer.start();
    parlay::parallel_for(0, num_ops, [&](size_t i) {
      auto [block, cap] = Block<KeyType, ValueType>::CreateWithCapacity(4);
      blocks[i] = block;
    });
    double alloc_time = timer.elapsed_ms();
    cout << "Parallel allocation: " << alloc_time << " ms, "
         << (num_ops / alloc_time * 1000 / 1e6) << " Mops/sec, "
         << (alloc_time / num_ops * 1e6) << " ns/alloc" << endl;

    // Cleanup
    parlay::parallel_for(0, num_ops, [&](size_t i) {
      Block<KeyType, ValueType>::Destroy(blocks[i], 4);
    });
  }

  // Test 2: memcpy speed for typical block sizes
  {
    cout << "\n--- Data Copy (memcpy) ---" << endl;
    for (size_t block_size : {4, 8, 16, 32, 64}) {
      size_t bytes = block_size * (sizeof(KeyType) + sizeof(ValueType));
      vector<char> src(bytes), dst(bytes);

      Timer timer;
      timer.start();
      for (size_t i = 0; i < num_ops; i++) {
        memcpy(dst.data(), src.data(), bytes);
      }
      double copy_time = timer.elapsed_ms();
      cout << "Block size " << block_size << " (" << bytes << " bytes): "
           << copy_time << " ms, "
           << (num_ops / copy_time * 1000 / 1e6) << " Mops/sec" << endl;
    }
  }

  // Test 3: Linear key-existence scan in a block.
  {
    cout << "\n--- Linear Key-Existence Scan ---" << endl;
    for (size_t block_size : {4, 8, 16, 32, 64}) {
      vector<KeyType> block_keys(block_size);
      for (size_t i = 0; i < block_size; i++) {
        block_keys[i] = i * 1000;
      }

      Timer timer;
      timer.start();
      volatile size_t found = 0;
      for (size_t i = 0; i < num_ops; i++) {
        KeyType target = (i % block_size) * 1000;  // Always found
        for (size_t j = 0; j < block_size; j++) {
          if (block_keys[j] == target) {
            found++;
            break;
          }
        }
      }
      double search_time = timer.elapsed_ms();
      cout << "Block size " << block_size << ": "
           << search_time << " ms, "
           << (num_ops / search_time * 1000 / 1e6) << " Mops/sec" << endl;
    }
  }

  // Test 4: CAS operation speed (sequential, always succeeds)
  {
    cout << "\n--- CAS Operation (Sequential) ---" << endl;
    atomic<uint64_t> target{0};

    Timer timer;
    timer.start();
    for (size_t i = 0; i < num_ops; i++) {
      uint64_t expected = i;
      uint64_t desired = i + 1;
      target.compare_exchange_strong(expected, desired);
    }
    double cas_time = timer.elapsed_ms();
    cout << "Sequential CAS: " << cas_time << " ms, "
         << (num_ops / cas_time * 1000 / 1e6) << " Mops/sec, "
         << (cas_time / num_ops * 1e6) << " ns/op" << endl;
  }

  // Test 5: CAS operation with contention
  {
    cout << "\n--- CAS Operation (Parallel, with contention) ---" << endl;
    atomic<uint64_t> target{0};
    atomic<uint64_t> success_count{0};

    Timer timer;
    timer.start();
    parlay::parallel_for(0, num_ops, [&](size_t i) {
      while (true) {
        uint64_t expected = target.load();
        uint64_t desired = expected + 1;
        if (target.compare_exchange_strong(expected, desired)) {
          success_count.fetch_add(1, memory_order_relaxed);
          break;
        }
      }
    });
    double cas_time = timer.elapsed_ms();
    cout << "Parallel CAS (contention): " << cas_time << " ms, "
         << (num_ops / cas_time * 1000 / 1e6) << " Mops/sec" << endl;
    cout << "Final value: " << target.load() << " (expected: " << num_ops << ")" << endl;
  }

  // Test 6: Full block create with copy (insert pattern)
  {
    cout << "\n--- Block Create + Copy (Insert Pattern) ---" << endl;
    for (size_t old_size : {4, 8, 16, 32}) {
      // Create source block
      auto [src_block, src_cap] = Block<KeyType, ValueType>::CreateWithCapacity(old_size);
      KeyType* src_keys = src_block->keys();
      ValueType* src_values = src_block->values(src_cap);
      for (size_t j = 0; j < old_size; j++) {
        src_keys[j] = j * 1000;
        src_values[j] = j;
      }

      Timer timer;
      timer.start();
      for (size_t i = 0; i < num_ops; i++) {
        // Create new block with +1 capacity
        auto [new_block, new_cap] = Block<KeyType, ValueType>::CreateWithCapacity(old_size + 1);

        // Copy with insert in middle
        KeyType* new_keys = new_block->keys();
        ValueType* new_values = new_block->values(new_cap);
        size_t insert_pos = old_size / 2;

        for (size_t j = 0; j < insert_pos; j++) {
          new_keys[j] = src_keys[j];
          new_values[j] = src_values[j];
        }
        new_keys[insert_pos] = 999999;  // Insert key
        new_values[insert_pos] = 999999;
        for (size_t j = insert_pos; j < old_size; j++) {
          new_keys[j + 1] = src_keys[j];
          new_values[j + 1] = src_values[j];
        }

        Block<KeyType, ValueType>::Destroy(new_block, new_cap);
      }
      double time = timer.elapsed_ms();
      cout << "Old block size " << old_size << ": "
           << time << " ms, "
           << (num_ops / time * 1000 / 1e6) << " Mops/sec, "
           << (time / num_ops * 1e6) << " ns/op" << endl;

      Block<KeyType, ValueType>::Destroy(src_block, src_cap);
    }
  }
}

// Main benchmark: varying data sizes
void bench_insert_throughput(size_t num_keys, size_t num_inserts) {
  cout << "\n========================================" << endl;
  cout << "Insert Throughput Benchmark" << endl;
  cout << "========================================" << endl;
  cout << "Num keys for bulk load: " << num_keys << endl;
  cout << "Num inserts: " << num_inserts << endl;
  cout << "Num workers: " << parlay::num_workers() << endl;

  // Setup: create sorted unique keys
  size_t total_keys = num_keys + num_inserts;
  vector<KeyType> all_keys(total_keys);
  vector<ValueType> all_values(total_keys);

  for (size_t i = 0; i < total_keys; i++) {
    all_keys[i] = parlay::hash64(i);
    all_values[i] = parlay::hash64(all_keys[i]);
  }

  // Sort
  auto entries = parlay::tabulate(total_keys, [&](size_t i) {
    return make_pair(all_keys[i], all_values[i]);
  });
  parlay::integer_sort_inplace(entries, [](auto& e) { return e.first; });

  vector<KeyType> sorted_keys(total_keys);
  vector<ValueType> sorted_values(total_keys);
  for (size_t i = 0; i < total_keys; i++) {
    sorted_keys[i] = entries[i].first;
    sorted_values[i] = entries[i].second;
  }

  // Split: first half for bulk load, second half for insert
  size_t load_size = num_keys;
  KeyType* load_keys = sorted_keys.data();
  ValueType* load_values = sorted_values.data();
  KeyType* insert_keys = sorted_keys.data() + load_size;
  ValueType* insert_values = sorted_values.data() + load_size;

  // Build index
  constexpr size_t kTopLevelThreshold = 256;
  constexpr size_t kBlockSize = 40;
  constexpr size_t kFingerSparsity = 2;
  constexpr size_t kInternalBlockSize = 16;
  constexpr size_t kInternalFingerSparsity = 1;

  cout << "\n--- Sequential Insert ---" << endl;
  {
    FHIndexRW<KeyType, ValueType> index(kTopLevelThreshold,
                                        kBlockSize, kBlockSize * 2, kFingerSparsity,
                                        kInternalBlockSize, kInternalBlockSize * 2, kInternalFingerSparsity);
    index.BulkLoad(load_size, load_keys, load_values);

    Timer timer;
    timer.start();
    for (size_t i = 0; i < num_inserts; i++) {
      index.Insert(insert_keys[i], insert_values[i]);
    }
    double insert_time = timer.elapsed_ms();
    cout << "Time: " << insert_time << " ms" << endl;
    cout << "Throughput: " << (num_inserts / insert_time * 1000 / 1e6) << " Mops/sec" << endl;
    cout << "Latency: " << (insert_time / num_inserts * 1e6) << " ns/op" << endl;
  }

  cout << "\n--- Parallel Insert ---" << endl;
  {
    FHIndexRW<KeyType, ValueType> index(kTopLevelThreshold,
                                        kBlockSize, kBlockSize * 2, kFingerSparsity,
                                        kInternalBlockSize, kInternalBlockSize * 2, kInternalFingerSparsity);
    index.BulkLoad(load_size, load_keys, load_values);

    Timer timer;
    timer.start();
    parlay::parallel_for(0, num_inserts, [&](size_t i) {
      index.Insert(insert_keys[i], insert_values[i]);
    });
    double insert_time = timer.elapsed_ms();
    cout << "Time: " << insert_time << " ms" << endl;
    cout << "Throughput: " << (num_inserts / insert_time * 1000 / 1e6) << " Mops/sec" << endl;
    cout << "Per-op time: " << (insert_time / num_inserts * 1e6) << " ns/op" << endl;
  }

  // Test: insert into empty vs populated blocks
  cout << "\n--- Insert Pattern Analysis ---" << endl;
  cout << "Testing insert distribution patterns..." << endl;

  // Create index with gaps (every other key)
  {
    vector<KeyType> sparse_keys;
    vector<ValueType> sparse_values;
    for (size_t i = 0; i < load_size; i += 2) {
      sparse_keys.push_back(sorted_keys[i]);
      sparse_values.push_back(sorted_values[i]);
    }

    FHIndexRW<KeyType, ValueType> index(kTopLevelThreshold,
                                        kBlockSize, kBlockSize * 2, kFingerSparsity,
                                        kInternalBlockSize, kInternalBlockSize * 2, kInternalFingerSparsity);
    index.BulkLoad(sparse_keys.size(), sparse_keys.data(), sparse_values.data());

    // Insert the missing keys
    vector<KeyType> fill_keys;
    vector<ValueType> fill_values;
    for (size_t i = 1; i < load_size && fill_keys.size() < num_inserts; i += 2) {
      fill_keys.push_back(sorted_keys[i]);
      fill_values.push_back(sorted_values[i]);
    }

    Timer timer;
    timer.start();
    parlay::parallel_for(0, fill_keys.size(), [&](size_t i) {
      index.Insert(fill_keys[i], fill_values[i]);
    });
    double insert_time = timer.elapsed_ms();
    cout << "Fill-gap pattern: " << insert_time << " ms, "
         << (fill_keys.size() / insert_time * 1000 / 1e6) << " Mops/sec" << endl;
  }
}

// CAS retry analysis
void bench_cas_retry(size_t num_keys, size_t num_inserts) {
  cout << "\n========================================" << endl;
  cout << "CAS Retry Analysis" << endl;
  cout << "========================================" << endl;

  // Setup data
  size_t total_keys = num_keys + num_inserts;
  vector<KeyType> all_keys(total_keys);
  vector<ValueType> all_values(total_keys);
  for (size_t i = 0; i < total_keys; i++) {
    all_keys[i] = parlay::hash64(i);
    all_values[i] = parlay::hash64(all_keys[i]);
  }
  auto entries = parlay::tabulate(total_keys, [&](size_t i) {
    return make_pair(all_keys[i], all_values[i]);
  });
  parlay::integer_sort_inplace(entries, [](auto& e) { return e.first; });

  vector<KeyType> sorted_keys(total_keys);
  vector<ValueType> sorted_values(total_keys);
  for (size_t i = 0; i < total_keys; i++) {
    sorted_keys[i] = entries[i].first;
    sorted_values[i] = entries[i].second;
  }

  // Simulate high contention: all inserts target same key range
  cout << "\nSimulating CAS contention with atomic counters..." << endl;

  // Create N atomic counters (simulating N blocks)
  size_t num_blocks = 1000;
  vector<atomic<uint64_t>> counters(num_blocks);
  for (auto& c : counters) c.store(0);

  atomic<uint64_t> total_retries{0};
  atomic<uint64_t> total_ops{0};

  Timer timer;
  timer.start();
  parlay::parallel_for(0, num_inserts, [&](size_t i) {
    // Target a random block (simulating insert distribution)
    size_t block_idx = parlay::hash64(i) % num_blocks;
    uint64_t retries = 0;

    while (true) {
      uint64_t old_val = counters[block_idx].load();
      uint64_t new_val = old_val + 1;
      if (counters[block_idx].compare_exchange_weak(old_val, new_val)) {
        break;
      }
      retries++;
    }

    total_retries.fetch_add(retries, memory_order_relaxed);
    total_ops.fetch_add(1, memory_order_relaxed);
  });
  double cas_time = timer.elapsed_ms();

  cout << "Num blocks: " << num_blocks << endl;
  cout << "Total ops: " << total_ops.load() << endl;
  cout << "Total retries: " << total_retries.load() << endl;
  cout << "Avg retries per op: " << (double)total_retries.load() / total_ops.load() << endl;
  cout << "Time: " << cas_time << " ms, "
       << (num_inserts / cas_time * 1000 / 1e6) << " Mops/sec" << endl;

  // High contention scenario: all target same block
  cout << "\nHigh contention (all target same block)..." << endl;
  atomic<uint64_t> single_counter{0};
  total_retries.store(0);
  total_ops.store(0);

  timer.start();
  parlay::parallel_for(0, num_inserts, [&](size_t i) {
    uint64_t retries = 0;
    while (true) {
      uint64_t old_val = single_counter.load();
      uint64_t new_val = old_val + 1;
      if (single_counter.compare_exchange_weak(old_val, new_val)) {
        break;
      }
      retries++;
    }
    total_retries.fetch_add(retries, memory_order_relaxed);
    total_ops.fetch_add(1, memory_order_relaxed);
  });
  cas_time = timer.elapsed_ms();

  cout << "Total retries: " << total_retries.load() << endl;
  cout << "Avg retries per op: " << (double)total_retries.load() / total_ops.load() << endl;
  cout << "Time: " << cas_time << " ms, "
       << (num_inserts / cas_time * 1000 / 1e6) << " Mops/sec" << endl;
}

int main(int argc, char** argv) {
  size_t num_keys = 10000;
  size_t num_inserts = 1000;

  if (argc >= 2) num_keys = stoull(argv[1]);
  if (argc >= 3) num_inserts = stoull(argv[2]);

  cout << "========================================" << endl;
  cout << "Insert Performance Breakdown Analysis" << endl;
  cout << "========================================" << endl;
  cout << "Num workers: " << parlay::num_workers() << endl;

  bench_individual_components(num_inserts);
  bench_insert_throughput(num_keys, num_inserts);
  bench_cas_retry(num_keys, num_inserts);

  return 0;
}
